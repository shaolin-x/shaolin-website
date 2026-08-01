"""Human-vs-judge agreement on the six G-Eval criteria — the offline half.

A line-for-line twin of `agreement.js`, down to the PRNG and the order the bootstrap
draws in, so the table a respondent saw in their browser and the table computed here are
the same table. `tests/test_agreement.py` asserts that; it is the only reason both exist.

A "cell" is one (trajectory, criterion) pair: a human's integer 1-5 against the judge's
mean over its repeats. Cells the judge never judged (QUIS trustworthiness, assigned 5 by
policy) are excluded and counted, never pooled.
"""

from __future__ import annotations

import decimal
import math
from typing import Callable, Iterable, Optional, Sequence

BOOTSTRAP = 1000
SEED = 20260731
MASK = 0xFFFFFFFF

CRITERION_LABELS = {
    "goal_relevance": "Goal relevance",
    "information_gain": "Information gain",
    "interestingness": "Interestingness",
    "trustworthiness": "Trustworthiness",
    "actionability": "Actionability",
    "resolution": "Resolution",
}


def label(key: str) -> str:
    return CRITERION_LABELS.get(key, key)


# --------------------------------------------------------------------------- basics

def _sum(xs: Iterable[float]) -> float:
    """Plain left-to-right accumulation.

    NOT the builtin `sum`: since 3.12 it applies Neumaier compensation to floats, which is
    more accurate than JavaScript's `reduce` and therefore disagrees with agreement.js in
    the last ulp — enough to fail the parity check that is the whole point of both files.
    """
    total = 0.0
    for x in xs:
        total += x
    return total


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return _sum(xs) / len(xs) if xs else float("nan")


def round1(x: float) -> int:
    """Half away from zero — JavaScript's `Math.sign(x) * Math.round(Math.abs(x))`."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x)) if x else 0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = mean(xs), mean(ys)
    sxy = _sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = _sum((x - mx) * (x - mx) for x in xs)
    syy = _sum((y - my) * (y - my) for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ranks(xs: Sequence[float]) -> list[float]:
    """Ties averaged. The judge's means take few distinct values, so ties are the rule."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    return None if len(xs) < 2 else pearson(ranks(xs), ranks(ys))


def qwk(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Quadratically weighted kappa over the 1-5 scale, against the rounded judge mean."""
    K, n = 5, len(a)
    if not n:
        return None
    O = [[0] * K for _ in range(K)]
    ra, rb = [0] * K, [0] * K
    for x_raw, y_raw in zip(a, b):
        x = min(K, max(1, round1(x_raw))) - 1
        y = min(K, max(1, round1(y_raw))) - 1
        O[x][y] += 1
        ra[x] += 1
        rb[y] += 1
    num = den = 0.0
    for i in range(K):
        for j in range(K):
            w = ((i - j) ** 2) / ((K - 1) ** 2)
            num += w * O[i][j]
            den += w * (ra[i] * rb[j]) / n
    return None if den <= 0 else 1 - num / den


def alpha_interval(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Krippendorff's alpha, interval difference, two coders, complete data."""
    n = len(a)
    if n < 2:
        return None
    do = _sum((x - y) * (x - y) for x, y in zip(a, b)) / n
    allv = list(a) + list(b)
    N = len(allv)
    de = _sum((allv[i] - allv[j]) * (allv[i] - allv[j])
              for i in range(N) for j in range(N) if i != j) / (N * (N - 1))
    return None if de <= 0 else 1 - do / de


# --------------------------------------------------------------------------- one row

def stats(rows: Sequence[dict]) -> dict:
    """One table row from {human, judge} pairs.

    `exact` compares against the judge's rounded mean; `within1` and `mae` against the
    unrounded one, so the part of the judge's answer that says "my three repeats
    disagreed" is not discarded before the comparison.
    """
    n = len(rows)
    if not n:
        return {"n": 0}
    h = [r["human"] for r in rows]
    m = [r["judge"] for r in rows]
    mr = [round1(x) for x in m]

    exact = mean([1.0 if hi == mri else 0.0 for hi, mri in zip(h, mr)])
    within1 = mean([1.0 if abs(hi - mi) <= 1 else 0.0 for hi, mi in zip(h, m)])
    mae = mean([abs(hi - mi) for hi, mi in zip(h, m)])

    # Chance is not 1/5: it is what two raters with THESE marginals hit by accident.
    ph = [0.0] * 6
    pm = [0.0] * 6
    for hi, mri in zip(h, mr):
        ph[min(5, max(1, int(hi)))] += 1 / n
        pm[min(5, max(1, mri))] += 1 / n
    exact_chance = _sum(ph[k] * pm[k] for k in range(1, 6))
    # `within1` is measured against the UNROUNDED judge mean, so its chance rate has to be
    # too: rounding first would quietly widen the band (a human 4 is within 1 of a rounded
    # 3 but not of a 2.667) and the baseline would sit above the thing it is a baseline for.
    within1_chance = _sum(ph[k] * mean([1.0 if abs(k - mi) <= 1 else 0.0 for mi in m])
                          for k in range(1, 6))

    return {
        "n": n,
        "human_mean": mean(h),
        "judge_mean": mean(m),
        "bias": mean([hi - mi for hi, mi in zip(h, m)]),
        "exact": exact, "exact_chance": exact_chance,
        "within1": within1, "within1_chance": within1_chance,
        "mae": mae,
        "pearson": pearson(h, m),
        "spearman": spearman(h, m),
        "qwk": qwk(h, m),
        "alpha": alpha_interval(h, m),
    }


def self_rows(cells: Iterable[dict]) -> list[dict]:
    """The judge against its own repeats: every unordered pair, entered both ways round so
    nothing depends on which repeat is called first."""
    out = []
    for c in cells:
        s = list(c.get("judge_scores") or [])
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                out.append({"criterion": c["criterion"], "item": c["item"],
                            "human": s[i], "judge": s[j]})
                out.append({"criterion": c["criterion"], "item": c["item"],
                            "human": s[j], "judge": s[i]})
    return out


# --------------------------------------------------------------------------- bootstrap

def prng(seed: int) -> Callable[[], float]:
    """mulberry32, bit-identical to agreement.js's."""
    a = seed & MASK

    def nxt() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return nxt


CI_METRICS = ("exact", "within1", "mae", "pearson", "spearman", "qwk", "alpha")


def bootstrap(by_item: dict, key_of: Callable[[dict], str], reps: int, seed: int) -> dict:
    """95% intervals, resampling TRAJECTORIES rather than cells: one trajectory yields six
    ratings from one person reading one screen, and treating those as independent draws
    would make every interval too narrow."""
    items = sorted(by_item)
    rand = prng(seed)
    draws: dict[str, dict[str, list[float]]] = {}
    for _ in range(reps):
        pool = []
        for _ in range(len(items)):
            pool.extend(by_item[items[int(rand() * len(items))]])
        groups: dict[str, list[dict]] = {}
        for c in pool:
            groups.setdefault(key_of(c), []).append(c)
        for k in sorted(groups):
            s = stats(groups[k])
            for m in CI_METRICS:
                v = s.get(m)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                draws.setdefault(k, {}).setdefault(m, []).append(v)
    out: dict[str, dict[str, list[float]]] = {}
    for k, metrics in draws.items():
        out[k] = {}
        for m, xs in metrics.items():
            xs = sorted(xs)
            if len(xs) < reps / 2:
                continue
            out[k][m] = [xs[int(0.025 * len(xs))],
                         xs[min(len(xs) - 1, int(0.975 * len(xs)))]]
    return out


# --------------------------------------------------------------------------- the report

def compute(cells: Sequence[dict], order: Optional[Sequence[str]] = None,
            reps: int = BOOTSTRAP) -> dict:
    """The whole report: per criterion, pooled, the judge's own ceiling, and intervals.

    `cells`: {item, dataset, criterion, human, judge, judge_scores, assigned}.
    """
    usable = [c for c in cells if not c.get("assigned") and c.get("human") is not None]
    excluded = len(cells) - len(usable)

    by_criterion: dict[str, list[dict]] = {}
    for c in usable:
        by_criterion.setdefault(c["criterion"], []).append(c)
    by_item: dict[str, list[dict]] = {}
    for c in usable:
        by_item.setdefault(c["item"], []).append(c)

    keys = [k for k in (order or list(by_criterion)) if by_criterion.get(k)]

    ci = bootstrap(by_item, lambda c: c["criterion"], reps, SEED) if reps else {}
    ci_pooled = bootstrap(by_item, lambda c: "pooled", reps, SEED + 1) if reps else {}

    self_all = self_rows(usable)
    self_by_criterion: dict[str, list[dict]] = {}
    for r in self_all:
        self_by_criterion.setdefault(r["criterion"], []).append(r)

    return {
        "version": 1,
        "n_items": len(by_item),
        "n_cells": len(usable),
        "n_excluded_assigned": excluded,
        "criteria": [
            {"criterion": k, **stats(by_criterion[k]),
             "ci": ci.get(k, {}), "judge_self": stats(self_by_criterion.get(k, []))}
            for k in keys],
        "pooled": {"criterion": "all criteria", **stats(usable),
                   "ci": ci_pooled.get("pooled", {}), "judge_self": stats(self_all)},
        "assigned": [{"item": c["item"], "criterion": c["criterion"],
                      "human": c["human"], "judge": c["judge"]}
                     for c in cells if c.get("assigned") and c.get("human") is not None],
    }


# --------------------------------------------------------------------------- rendering

def _to_fixed(x: float, d: int) -> str:
    """JavaScript's `Number.prototype.toFixed`, which Python's `%.*f` is not.

    Python rounds a tie to even, so 31.25 prints as "31.2"; JS splits the sign off and
    formats the magnitude, rounding a tie up, so it prints "31.3". And 0.3125 is 5/16 —
    exactly representable, and an agreement rate that really does come up with n = 16.
    Every digit of these two files is required to match, so the formatter matches too.
    """
    q = decimal.Decimal(1).scaleb(-d)
    return str(decimal.Decimal(x).quantize(q, rounding=decimal.ROUND_HALF_UP))


def fmt(v, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return _to_fixed(v, d)


def pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return _to_fixed(100 * v, 1) + "%"


COLUMNS: tuple[tuple[str, Callable[[dict], str]], ...] = (
    ("n", lambda r: str(r.get("n", 0))),
    ("human", lambda r: fmt(r.get("human_mean"), 2)),
    ("judge", lambda r: fmt(r.get("judge_mean"), 2)),
    ("exact", lambda r: pct(r.get("exact"))),
    ("chance", lambda r: pct(r.get("exact_chance"))),
    ("±1", lambda r: pct(r.get("within1"))),
    ("MAE", lambda r: fmt(r.get("mae"), 2)),
    ("r", lambda r: fmt(r.get("pearson"))),
    ("rho", lambda r: fmt(r.get("spearman"))),
    ("QWK", lambda r: fmt(r.get("qwk"))),
    ("alpha", lambda r: fmt(r.get("alpha"))),
    ("judge vs itself", lambda r: (
        f"{pct(r['judge_self'].get('exact'))} exact, {fmt(r['judge_self'].get('mae'), 2)} MAE"
        if r.get("judge_self", {}).get("n") else "—")),
)


def _rows(report: dict) -> list[list[str]]:
    rows = [[label(r["criterion"])] + [f(r) for _, f in COLUMNS] for r in report["criteria"]]
    rows.append(["ALL CRITERIA"] + [f(report["pooled"]) for _, f in COLUMNS])
    return rows


def to_text(report: dict) -> str:
    head = ["criterion"] + [h for h, _ in COLUMNS]
    rows = [head] + _rows(report)
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    def line(r): return "  ".join(v.ljust(w[i]) if i == 0 else v.rjust(w[i])
                                  for i, v in enumerate(r))
    out = [line(head), "  ".join("-" * n for n in w)] + [line(r) for r in rows[1:]]
    tail = (f"\n{report['n_items']} trajectories, {report['n_cells']} ratings")
    if report["n_excluded_assigned"]:
        tail += (f", {report['n_excluded_assigned']} excluded "
                 "(assigned by policy, not judged)")
    return "\n".join(out) + "\n" + tail


def to_markdown(report: dict) -> str:
    head = ["criterion"] + [h for h, _ in COLUMNS]
    rows = _rows(report)
    return "\n".join([f"| {' | '.join(head)} |",
                      "|" + "|".join("---" for _ in head) + "|"]
                     + [f"| {' | '.join(r)} |" for r in rows])


CSV_FIELDS = ("criterion", "n", "human_mean", "judge_mean", "bias", "exact",
              "exact_chance", "within1", "within1_chance", "mae", "pearson",
              "spearman", "qwk", "alpha", "judge_self_exact", "judge_self_mae")


def csv_num(v) -> str:
    """One CSV number. Nine decimals rather than repr: `str(1.0)` is "1.0" in Python and
    "1" in JavaScript, and the two files are required to emit the same bytes."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.9f}"


def to_csv(report: dict) -> str:
    def line(r: dict) -> str:
        vals = [r.get(k) for k in CSV_FIELDS[1:-2]] + [
            r.get("judge_self", {}).get("exact"), r.get("judge_self", {}).get("mae")]
        return ",".join([r["criterion"], str(r.get("n", 0))]
                        + [csv_num(v) for v in vals[1:]])
    return "\n".join([",".join(CSV_FIELDS)]
                     + [line(r) for r in report["criteria"]] + [line(report["pooled"])])
