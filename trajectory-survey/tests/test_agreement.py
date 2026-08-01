#!/usr/bin/env python3
"""The agreement scoring's correctness suite.

Four groups, in the order they are worth reading:

  estimators   hand-worked values, plus a cross-check of the tie-corrected rank
               correlation and the weighted kappa against scipy / sklearn where they are
               installed
  invariants   an oracle respondent must score 1.0 exact agreement and 0 MAE; an
               adversarial one must correlate negatively; both must hold at any n
  baselines    hundreds of uniformly random respondents, each metric required to converge
               on its own chance column — which validates a metric and its baseline at
               once, with no ground truth involved
  parity       agreement.js run under node over the same cells, every cell diffed at 1e-9,
               bootstrap intervals included

    python3 trajectory-survey/tests/test_agreement.py
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import agreement as A  # noqa: E402
from synth_responses import make  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name} {detail}")


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b or (a is None and b is None)
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cells(pairs, criterion="goal_relevance", repeats=None):
    """{human, judge} pairs as cells, one per synthetic trajectory."""
    out = []
    for i, (h, m) in enumerate(pairs):
        out.append({"item": f"i{i:03d}", "dataset": "d", "criterion": criterion,
                    "human": h, "judge": m,
                    "judge_scores": repeats[i] if repeats else [round(m)] * 3,
                    "assigned": False})
    return out


# --------------------------------------------------------------------------- estimators

def test_estimators() -> None:
    print("estimators")

    # Pearson on a perfect line, and on its mirror.
    xs = [1, 2, 3, 4, 5]
    check("pearson +1", close(A.pearson(xs, xs), 1.0))
    check("pearson -1", close(A.pearson(xs, [5, 4, 3, 2, 1]), -1.0))
    check("pearson flat is None", A.pearson(xs, [3, 3, 3, 3, 3]) is None)

    # Ties averaged: two 2s share ranks 2 and 3, so both get 2.5.
    check("ranks average ties", A.ranks([1, 2, 2, 4]) == [1.0, 2.5, 2.5, 4.0])

    # Weighted kappa, worked by hand. One rater constant: whatever the other does,
    # observed and expected disagreement are equal, so kappa is exactly 0 — chance
    # agreement is all there is.
    check("qwk against a constant rater is 0", close(A.qwk([1, 1], [1, 3]), 0.0),
          f"{A.qwk([1, 1], [1, 3])}")
    # Two raters that swap adjacent categories: observed disagreement is twice expected.
    check("qwk hand-worked -1", close(A.qwk([1, 2], [2, 1]), -1.0), f"{A.qwk([1, 2], [2, 1])}")
    check("qwk identical is 1", close(A.qwk([1, 2, 3], [1, 2, 3]), 1.0))

    # Krippendorff alpha: identical coders have no disagreement to explain.
    check("alpha identical is 1", close(A.alpha_interval([1, 2, 3, 4], [1, 2, 3, 4]), 1.0))
    # ...and a constant-vs-constant pair has no expected disagreement either, so it is
    # reported as unestimable rather than as perfect.
    check("alpha degenerate is None", A.alpha_interval([2, 2], [2, 2]) is None)
    # Hand-worked: Do = mean squared difference, De from the pooled marginals.
    a, b = [1, 2, 3], [2, 2, 4]
    do = (1 + 0 + 1) / 3
    allv = a + b
    de = sum((x - y) ** 2 for i, x in enumerate(allv) for j, y in enumerate(allv)
             if i != j) / (len(allv) * (len(allv) - 1))
    check("alpha hand-worked", close(A.alpha_interval(a, b), 1 - do / de))

    # round1 is half-away-from-zero, matching JavaScript's Math.round on the absolute value.
    check("round1 half up", [A.round1(x) for x in (2.5, 2.4, 2.6, 3.5)] == [3, 2, 3, 4])

    try:
        from scipy import stats as sps
        rng = random.Random(4)
        xs = [rng.randint(1, 5) for _ in range(200)]
        ys = [rng.randint(1, 5) for _ in range(200)]
        check("spearman vs scipy", close(A.spearman(xs, ys), sps.spearmanr(xs, ys).statistic, 1e-9))
    except ImportError:
        print("  skip  spearman vs scipy (scipy not installed)")

    try:
        from sklearn.metrics import cohen_kappa_score
        rng = random.Random(5)
        xs = [rng.randint(1, 5) for _ in range(300)]
        ys = [max(1, min(5, x + rng.choice([-1, 0, 1]))) for x in xs]
        check("qwk vs sklearn",
              close(A.qwk(xs, ys), cohen_kappa_score(xs, ys, weights="quadratic"), 1e-9))
    except ImportError:
        print("  skip  qwk vs sklearn (sklearn not installed)")


# --------------------------------------------------------------------------- invariants

def test_invariants() -> None:
    print("invariants")
    rng = random.Random(11)
    judge = [rng.choice([1, 1.333, 2, 2.667, 3, 3.333, 4, 4.667, 5]) for _ in range(40)]

    oracle = cells([(max(1, min(5, A.round1(m))), m) for m in judge])
    r = A.compute(oracle, reps=0)["pooled"]
    check("oracle exact = 1", close(r["exact"], 1.0))
    check("oracle MAE <= 1/3", r["mae"] <= 1 / 3 + 1e-12, f"{r['mae']}")
    check("oracle r > .95", r["pearson"] > 0.95, f"{r['pearson']}")

    adver = cells([(max(1, min(5, 6 - A.round1(m))), m) for m in judge])
    r = A.compute(adver, reps=0)["pooled"]
    check("adversarial r < -.9", r["pearson"] < -0.9, f"{r['pearson']}")
    check("adversarial qwk < 0", r["qwk"] < 0, f"{r['qwk']}")
    check("adversarial alpha < 0", r["alpha"] < 0, f"{r['alpha']}")

    # An excluded (assigned) cell must not enter anything, and must be counted.
    mixed = cells([(3, 3.0)] * 5)
    mixed.append({"item": "x", "dataset": "d", "criterion": "trustworthiness",
                  "human": 1, "judge": 5.0, "judge_scores": [5, 5, 5], "assigned": True})
    rep = A.compute(mixed, reps=0)
    check("assigned cells excluded", rep["n_cells"] == 5 and rep["n_excluded_assigned"] == 1)
    check("assigned cells reported apart", len(rep["assigned"]) == 1)


# --------------------------------------------------------------------------- baselines

def test_baselines() -> None:
    print("baselines")
    rng = random.Random(3)
    judge = [rng.choice([1, 1.333, 2, 2.333, 2.667, 3, 3.667, 4, 5]) for _ in range(60)]

    # 400 uniformly random respondents over the same judged trajectories. Each metric is
    # averaged over respondents and must land on the chance rate the table prints beside
    # it — the point being that neither number is derived from the other.
    ex, ex_c, w1, w1_c, ks, al = [], [], [], [], [], []
    for _ in range(400):
        c = cells([(rng.randint(1, 5), m) for m in judge])
        r = A.compute(c, reps=0)["pooled"]
        ex.append(r["exact"]); ex_c.append(r["exact_chance"])
        w1.append(r["within1"]); w1_c.append(r["within1_chance"])
        ks.append(r["qwk"]); al.append(r["alpha"])
    check("random exact ~ chance", abs(A.mean(ex) - A.mean(ex_c)) < 0.02,
          f"{A.mean(ex):.3f} vs {A.mean(ex_c):.3f}")
    check("random ±1 ~ chance", abs(A.mean(w1) - A.mean(w1_c)) < 0.02,
          f"{A.mean(w1):.3f} vs {A.mean(w1_c):.3f}")
    check("random qwk ~ 0", abs(A.mean(ks)) < 0.03, f"{A.mean(ks):.3f}")
    check("random alpha ~ 0", abs(A.mean(al)) < 0.05, f"{A.mean(al):.3f}")

    # The judge's own ceiling: three identical repeats agree perfectly, and repeats that
    # differ by one must not.
    same = cells([(3, 3.0)] * 10, repeats=[[3, 3, 3]] * 10)
    check("judge self-agreement 1 when repeats identical",
          close(A.compute(same, reps=0)["pooled"]["judge_self"]["exact"], 1.0))
    split = cells([(3, 3.333)] * 10, repeats=[[3, 3, 4]] * 10)
    r = A.compute(split, reps=0)["pooled"]["judge_self"]
    check("judge self-agreement 1/3 when one repeat differs", close(r["exact"], 1 / 3),
          f"{r['exact']}")


# --------------------------------------------------------------------------- parity

def js_report(cell_list, order, reps):
    with tempfile.TemporaryDirectory() as tmp:
        cpath = Path(tmp) / "cells.json"
        opath = Path(tmp) / "order.json"
        cpath.write_text(json.dumps(cell_list))
        opath.write_text(json.dumps(order))
        proc = subprocess.run(
            ["node", str(HERE / "run_agreement.mjs"), str(cpath), str(opath), str(reps)],
            capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return json.loads(proc.stdout)


def walk(py, js, path="") -> list[str]:
    """Every leaf of the two reports, compared. Structure differences are failures too."""
    bad = []
    if isinstance(py, dict):
        if not isinstance(js, dict):
            return [f"{path}: dict vs {type(js).__name__}"]
        for k in sorted(set(py) | set(js)):
            if k not in py or k not in js:
                bad.append(f"{path}.{k}: missing on {'python' if k not in py else 'js'}")
                continue
            bad += walk(py[k], js[k], f"{path}.{k}")
    elif isinstance(py, list):
        if not isinstance(js, list) or len(py) != len(js):
            return [f"{path}: length {len(py)} vs {len(js) if isinstance(js, list) else '?'}"]
        for i, (a, b) in enumerate(zip(py, js)):
            bad += walk(a, b, f"{path}[{i}]")
    elif isinstance(py, (int, float)) and not isinstance(py, bool):
        if not isinstance(js, (int, float)) or not close(py, js):
            bad.append(f"{path}: {py} vs {js}")
    else:
        if py != js:
            bad.append(f"{path}: {py!r} vs {js!r}")
    return bad


def test_parity() -> None:
    print("parity (js vs python)")
    if subprocess.run(["node", "-v"], capture_output=True).returncode != 0:
        print("  skip  node not on PATH")
        return

    data = ROOT / "data"
    if not (data / "survey.json").is_file():
        print("  skip  no built bundle (run build_survey.py first)")
        return
    survey = json.loads((data / "survey.json").read_text())
    order = survey["meta"]["criteria"]

    if (data / "truth.json").is_file():
        truth = json.loads((data / "truth.json").read_text())
    else:
        # A collection-only build publishes no key. The two implementations still have to
        # agree, so parity runs against a synthetic one rather than quietly not running —
        # the estimators do not care where a judge score came from.
        print("  note  no answer key published; parity runs against a synthetic one")
        rng = random.Random(99)
        truth = {"meta": {"criteria": order}, "items": {}}
        for it in survey["items"]:
            reps = {c: [rng.randint(1, 5) for _ in range(3)] for c in it["criteria"]}
            truth["items"][it["id"]] = {
                "system": "synthetic", "dataset": it["dataset"],
                "scores": {c: {"score": sum(v) / len(v), "scores": v,
                               "reason": "", "assigned": c in (it.get("optional") or [])}
                           for c, v in reps.items()},
            }

    for kind in ("oracle", "noisy", "random", "adversarial"):
        R = make(kind, survey, truth, seed=7)
        cell_list = []
        for ans in R["answers"]:
            t = truth["items"][ans["item"]]
            for crit, value in ans["scores"].items():
                block = t["scores"].get(crit)
                if not block:
                    continue
                cell_list.append({
                    "item": ans["item"], "dataset": ans["dataset"], "criterion": crit,
                    "human": value, "judge": block["score"],
                    "judge_scores": block.get("scores") or [],
                    "assigned": bool(block.get("assigned")),
                })
        py = A.compute(cell_list, order=order, reps=200)
        js = js_report(cell_list, order, 200)
        diffs = walk(py, js["report"])
        check(f"{kind}: every cell equal", not diffs, "; ".join(diffs[:4]))
        check(f"{kind}: rendered table equal", A.to_text(py) == js["text"])
        check(f"{kind}: csv equal", A.to_csv(py) == js["csv"])


def main() -> int:
    test_estimators()
    test_invariants()
    test_baselines()
    test_parity()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
