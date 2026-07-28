#!/usr/bin/env python
"""Score the response files the survey hands back, using the evaluation's own report.

A downloaded response says only "on decision X I picked slot 3". Resolving that to a
candidate needs the shuffle, which lives in `private/decisions.json` and was never
published — so scoring happens here, offline, and a respondent could not have worked out
their own rank from anything the page gave them.

Each response is turned into the session dict `human_trajectory_eval` already knows how to
read, so the tables printed here are that module's `report()`, byte for byte, not a second
implementation of the metrics. On top of it: a pooled report treating every respondent's
picks as one sample, and — with two or more respondents — how often the humans agreed with
each other, which is the ceiling any agreement with D2I should be read against.

    python survey/score_survey.py responses/*.json
    python survey/score_survey.py responses/ -o results/
    python survey/score_survey.py responses/ --per-respondent
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import human_trajectory_eval as hte        # noqa: E402


def load_private(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"no private bundle at {path} — run `python survey/build_survey.py` first "
            "(and keep its output, or the returned responses cannot be scored)")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(response: dict, by_id: dict[str, dict], label: str) -> tuple[list[dict], list[str]]:
    """`(verdicts, warnings)` — the response's answers with `true_index` filled in.

    A slot is 1-based and points into the *shuffled* order, so `order[slot - 1]` is the
    candidate it stood for. Anything that does not line up is dropped with a warning rather
    than raising: one malformed row should not cost the whole respondent.
    """
    out, warn = [], []
    for v in response.get("verdicts") or []:
        d = by_id.get(v.get("decision"))
        if d is None:
            warn.append(f"{label}: unknown decision {v.get('decision')!r} — skipped")
            continue
        slot = v.get("slot")
        true_index = None
        if slot is not None:
            try:
                slot = int(slot)
            except (TypeError, ValueError):
                warn.append(f"{label}: {d['id']} has a non-numeric slot {v.get('slot')!r} — skipped")
                continue
            if not 1 <= slot <= len(d["order"]):
                warn.append(f"{label}: {d['id']} slot {slot} out of range — skipped")
                continue
            true_index = d["order"][slot - 1]
        out.append({
            "decision": d["id"],
            "terminate": v.get("terminate"),
            "slot": slot,
            "true_index": true_index,
            "choice": str(v.get("choice") or ""),
            "note": str(v.get("note") or "").strip(),
            "ts": v.get("ts") or "",
        })
    return out, warn


def session_for(private: dict, verdicts: list[dict], judge: str | None, stamp: str) -> dict:
    """The private bundle with one respondent's verdicts attached — a session `report()` reads."""
    session = dict(private)
    session["judge"] = judge
    session["stamp"] = stamp
    session["verdicts"] = verdicts
    return session


def duplicate_free(verdicts: list[dict], label: str) -> tuple[list[dict], list[str]]:
    """Last answer wins per decision. The page cannot produce a duplicate, but a response
    edited by hand or stitched from two sittings can, and `report()` would count both."""
    seen: dict[str, dict] = {}
    for v in verdicts:
        seen[v["decision"]] = v
    dropped = len(verdicts) - len(seen)
    return list(seen.values()), ([f"{label}: {dropped} duplicate answer(s), kept the last"]
                                 if dropped else [])


def inter_rater(all_verdicts: dict[str, list[dict]]) -> list[str]:
    """Pairwise human-human agreement, on the decisions any two of them both answered.

    Reported alongside agreement with D2I because it is the honest reference point: if two
    analysts only pick the same next question 40% of the time, the model matching a human
    40% of the time is not a shortfall.
    """
    names = sorted(all_verdicts)
    if len(names) < 2:
        return []

    picks = {n: {v["decision"]: v["true_index"] for v in vs if v["true_index"] is not None}
             for n, vs in all_verdicts.items()}
    stops = {n: {v["decision"]: v["terminate"] for v in vs if v.get("terminate") is not None}
             for n, vs in all_verdicts.items()}

    lines = ["", "human vs human", "",
             f"{'pair':>28}  {'same pick':>10}  {'same stop':>10}  {'n':>4}", "-" * 60]
    pick_rates, stop_rates = [], []
    for a, b in itertools.combinations(names, 2):
        shared = sorted(set(picks[a]) & set(picks[b]))
        same = sum(1 for k in shared if picks[a][k] == picks[b][k])
        sshared = sorted(set(stops[a]) & set(stops[b]))
        ssame = sum(1 for k in sshared if stops[a][k] == stops[b][k])
        if shared:
            pick_rates.append(same / len(shared))
        if sshared:
            stop_rates.append(ssame / len(sshared))
        lines.append(
            f"{(a + ' / ' + b)[:28]:>28}  "
            f"{(f'{100 * same / len(shared):.1f}%' if shared else '—'):>10}  "
            f"{(f'{100 * ssame / len(sshared):.1f}%' if sshared else '—'):>10}  "
            f"{len(shared):>4}")
    lines.append("-" * 60)
    if pick_rates:
        lines.append(f"{'mean over pairs':>28}  {100 * sum(pick_rates) / len(pick_rates):>9.1f}%  "
                     f"{(f'{100 * sum(stop_rates) / len(stop_rates):.1f}%' if stop_rates else '—'):>10}  "
                     f"{len(pick_rates):>4}")
    lines += ["(two humans picking the same candidate out of a pool of ~7; this is the ceiling",
              " the D2I agreement rate above should be read against, not 100%.)"]
    return lines


def consensus(all_verdicts: dict[str, list[dict]], by_id: dict[str, dict]) -> list[str]:
    """Where the respondents disagreed with each other, worst first — the decisions worth
    looking at by hand."""
    votes: dict[str, list[int]] = defaultdict(list)
    for vs in all_verdicts.values():
        for v in vs:
            if v["true_index"] is not None:
                votes[v["decision"]].append(v["true_index"])
    split = [(k, v) for k, v in votes.items() if len(v) > 1]
    if not split:
        return []
    lines = ["", "where the respondents split", "",
             f"{'decision':>26}  {'n':>3}  {'modal':>6}  {'agreement':>10}  modal pick", "-" * 78]
    for k, v in sorted(split, key=lambda kv: len(set(kv[1])) / len(kv[1]), reverse=True):
        top = max(set(v), key=v.count)
        d = by_id[k]
        action = d["candidates"][top]["action"] if top < len(d["candidates"]) else "?"
        lines.append(f"{k[:26]:>26}  {len(v):>3}  {v.count(top):>6}  "
                     f"{100 * v.count(top) / len(v):>9.1f}%  [{action}] "
                     f"{d['candidates'][top]['question'][:34]}")
    return lines


def implied_weights(session: dict, seed: int = hte.BOOTSTRAP_SEED) -> list[str]:
    """The score weights that best explain the humans' picks, against D2I's own.

    A conditional logit over the five score terms: each candidate pool is one choice set,
    and the probability of picking candidate *j* is `softmax(beta . x_j)` over that pool.
    The fitted `beta`, rescaled to D2I's own lambda total, is directly comparable to the
    lambdas in `config.json` — so the result reads as "humans behave as if impact were
    weighted higher, and trajectory lower, than D2I weights them".

    Read from the decisions rather than from the precomputed pick rows: the design matrix
    is the whole pool's term values, which would multiply the size of `truth.json` if it
    were published per slot, and the browser does not fit this model anyway.

    Reported as exploratory below ~30 choice sets: five free parameters need more than that
    before the direction of any one of them is worth quoting.
    """
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:                                  # pragma: no cover
        return ["", "(scipy/numpy not installed — implied weights skipped)"]

    shorts = [short for _, short in hte.TERMS]
    keys = [key for key, _ in hte.TERMS]
    by_id = {d["id"]: d for d in session.get("decisions", [])}
    lambdas = next((d.get("lambdas") for d in session.get("decisions", []) if d.get("lambdas")),
                   None)

    # x[i][j][k]: candidate j's value on term k in choice set i; y[i] the chosen index.
    xs, ys = [], []
    for v in session.get("verdicts", []):
        d = by_id.get(v.get("decision"))
        if d is None or v.get("true_index") is None or not d.get("candidates"):
            continue
        rows = []
        for c in d["candidates"]:
            bd = c.get("breakdown") or {}
            vals = [bd.get(k) for k in keys]
            if not all(hte._numeric(x) for x in vals):
                rows = []
                break
            rows.append([float(x) for x in vals])
        if len(rows) > 1:
            xs.append(np.array(rows, dtype=float))
            ys.append(int(v["true_index"]))
    if len(xs) < 4:
        return ["", "(too few pools carry a full term breakdown for the implied-weight fit)"]

    def negloglik(beta):
        total = 0.0
        for x, y in zip(xs, ys):
            u = x @ beta
            u -= u.max()                                 # softmax overflow guard
            total -= u[y] - np.log(np.exp(u).sum())
        return total

    k = xs[0].shape[1]
    fit = minimize(negloglik, np.zeros(k), method="BFGS")
    beta = fit.x

    def stat(sample_idx):
        sx = [xs[i] for i in sample_idx]
        sy = [ys[i] for i in sample_idx]

        def nll(b):
            t = 0.0
            for x, y in zip(sx, sy):
                u = x @ b
                u -= u.max()
                t -= u[y] - np.log(np.exp(u).sum())
            return t
        return minimize(nll, np.zeros(k), method="BFGS").x

    rnd = hte.mulberry32(seed)
    draws = []
    for _ in range(200):                                 # 200: each draw refits the model
        idx = [int(rnd() * len(xs)) % len(xs) for _ in range(len(xs))]
        try:
            draws.append(stat(idx))
        except Exception:                                # pragma: no cover - separation
            continue

    total = sum(abs(v) for v in beta) or 1.0
    scale = (sum(lambdas.values()) if lambdas else 1.0) / total
    lines = ["", "human-implied score weights (conditional logit over the five terms)", "",
             f"{'term':>14}  {'implied':>9}  {'95% CI':>18}  {'D2I λ':>8}", "-" * 56]
    for i, short in enumerate(shorts):
        col = sorted(d[i] * scale for d in draws) if draws else []
        ci = (f"{col[int(0.025 * (len(col) - 1))]:>8.3f} –{col[int(0.975 * (len(col) - 1))]:>8.3f}"
              if len(col) > 20 else f"{'—':>18}")
        lam = f"{lambdas[short]:>8.2f}" if lambdas and short in lambdas else f"{'—':>8}"
        lines.append(f"{short:>14}  {beta[i] * scale:>9.3f}  {ci}  {lam}")
    if not fit.success:
        lines.append("  (the fit did not converge — the terms are near-collinear on this sample)")
    if len(xs) < 30:
        lines.append(f"  (exploratory: {len(xs)} choice sets for {k} free parameters)")
    return lines


def csv_export(picks: list[dict], stops: list[dict], agg: list[dict], out: Path) -> list[str]:
    """One row per pick and per stop call, plus the metric table — for a spreadsheet or a
    re-analysis that does not want to go through this script at all."""
    import csv

    written = []
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "group", "estimator", "value", "chance", "ci_lo", "ci_hi", "n"])
        for r in agg:
            w.writerow([r["label"], r["group"], r["est"], r["value"], r["chance"],
                        r["lo"], r["hi"], r["n"]])
    written.append("metrics.csv")

    if picks:
        cols = ["id", "run", "depth", "cluster", "n", "is_top", "e_top", "attainment",
                "e_attainment", "d_std", "e_d_std", "within_noise", "pool_sd", "top2_gap",
                "flat_pool", "action_hit", "pick_action", "top_action"]
        with (out / "picks.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(picks)
        written.append("picks.csv")

    if stops:
        cols = ["id", "run", "depth", "cluster", "source", "model", "human", "agree",
                "human_continue", "utility", "threshold", "margin"]
        with (out / "stops.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(stops)
        written.append("stops.csv")
    return written


# The table labels are written for a terminal, where Greek and box-drawing characters are
# free. pdfLaTeX — which is what acmart is normally run under — cannot set any of them, so
# the export substitutes math mode rather than shipping raw UTF-8 into a paper.
LATEX_LABELS = {
    "κ_sel (chance-corrected agreement)": r"$\kappa_{\mathrm{sel}}$ (chance-corrected)",
    "  ├ raw agreement rate": r"\quad raw agreement rate",
    "utility attainment [0,1]": r"Utility attainment $A$",
    "standardized regret (pool SDs)": r"Standardized regret $d$",
    "  └ within-noise rate (d<0.5)": r"\quad within-noise rate ($d<0.5$)",
    "action agreement": "Action agreement",
    "margin-weighted agreement": "Margin-weighted agreement",
    "  ├ unweighted (w≡1)": r"\quad unweighted ($w \equiv 1$)",
    "AUROC (utility vs human label)": "AUROC (utility vs.\\ human)",
    "implied human threshold τ̂": r"Implied threshold $\hat{\tau}$",
}


def latex_export(agg: list[dict], out: Path) -> str:
    """The metric table as a booktabs tabular, ready to \\input into the paper."""
    rows = []
    group = None
    for r in agg:
        if r["group"] != group:
            group = r["group"]
            rows.append(r"\midrule" if rows else "")
            rows.append(r"\multicolumn{5}{l}{\textit{%s alignment}} \\" % group.capitalize())

        def cell(x, fmt=r["fmt"]):
            if x is None:
                return "--"
            return f"{100 * x:.1f}\\%" if fmt == "pct" else f"{x:.3f}"

        ci = "--" if r["lo"] is None else f"[{cell(r['lo'])}, {cell(r['hi'])}]"
        label = LATEX_LABELS.get(
            r["label"],
            r["label"].replace("├", "").replace("└", "").replace("_", r"\_").strip())
        rows.append(f"{label} & {cell(r['value'])} & {cell(r['chance'])} & {ci} & {r['n']} \\\\")
    body = "\n".join(x for x in rows if x)
    text = (
        "% generated by survey/score_survey.py -- do not edit by hand\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Metric & Human & Chance & 95\\% CI & $n$ \\\\\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    (out / "metrics.tex").write_text(text, encoding="utf-8")
    return "metrics.tex"


def check_parity(responses: list[tuple[str, dict, dict]]) -> list[str]:
    """Compare each response's browser-computed table against a recomputation here.

    Under the precompute design this should be near-tautological — the page averages rows
    Python wrote — so a failure means a *deployment* problem (a page served against a stale
    truth.json) rather than a formula bug, which is exactly the failure that would otherwise
    go unnoticed until the numbers were already in a paper.
    """
    lines = ["", "parity: browser vs this script", ""]
    for name, response, session in responses:
        got = (response.get("results") or {})
        summary = got.get("summary")
        if not summary:
            lines.append(f"  {name:>28}  no results block (an older page, or --no-reveal)")
            continue
        if got.get("metrics_version") and got["metrics_version"] != hte.METRICS_VERSION:
            lines.append(f"  {name:>28}  built by metrics v{got['metrics_version']}, "
                         f"this is v{hte.METRICS_VERSION} — skipped")
            continue
        mine = hte.aggregate(hte.scored_picks(session), hte.termination_rows(session))
        worst, bad = 0.0, 0
        for a, b in zip(summary, mine):
            for field in ("value", "chance", "lo", "hi"):
                x, y = a.get(field), b.get(field)
                if x is None or y is None:
                    bad += 0 if (x is None and y is None) else 1
                else:
                    worst = max(worst, abs(x - y))
                    bad += 1 if abs(x - y) > 1e-9 else 0
        lines.append(f"  {name:>28}  {'OK' if not bad else f'{bad} cell(s) differ'}"
                     f"  (worst delta {worst:.2e})")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("responses", type=Path, nargs="+",
                   help="response .json files, or directories holding them")
    p.add_argument("--private", type=Path, default=HERE / "private" / "decisions.json",
                   help="the build's private bundle (default: survey/private/decisions.json)")
    p.add_argument("--per-respondent", action="store_true",
                   help="also print each respondent's own report, not just the pooled one")
    p.add_argument("--check-parity", action="store_true",
                   help="verify each response's browser-computed table against this script")
    p.add_argument("--export", action="store_true",
                   help="with -o, also write metrics.csv / picks.csv / stops.csv / metrics.tex")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write the tables here (a directory; one .txt per respondent plus "
                        "pooled.txt)")
    args = p.parse_args()

    private = load_private(args.private)
    by_id = {d["id"]: d for d in private["decisions"]}

    files: list[Path] = []
    for r in args.responses:
        if r.is_dir():
            files += sorted(r.glob("*.json"))
        elif r.is_file():
            files.append(r)
        else:
            print(f"skipped {r} — not a file or directory")
    if not files:
        raise SystemExit("no response files found")

    warnings: list[str] = []
    all_verdicts: dict[str, list[dict]] = {}
    raw_responses: dict[str, dict] = {}       # kept whole for --check-parity
    built = private.get("seed")

    for f in files:
        try:
            response = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            warnings.append(f"{f.name}: unreadable ({e.__class__.__name__}) — skipped")
            continue

        who = ((response.get("respondent") or {}).get("name") or "").strip() or f.stem
        if who in all_verdicts:                       # two files, one name
            who = f"{who} ({f.stem})"

        # A response taken against a different build refers to decisions that no longer
        # exist, or worse, to ids that exist with a different shuffle behind them.
        seed = (response.get("survey") or {}).get("seed")
        if seed is not None and built is not None and seed != built:
            warnings.append(f"{f.name}: built from seed {seed}, this bundle is {built} — skipped")
            continue

        verdicts, warn = resolve(response, by_id, f.name)
        verdicts, warn2 = duplicate_free(verdicts, f.name)
        warnings += warn + warn2
        if not verdicts:
            warnings.append(f"{f.name}: no usable answers — skipped")
            continue
        all_verdicts[who] = verdicts
        raw_responses[who] = response

    if not all_verdicts:
        raise SystemExit("nothing to score" + ("\n  " + "\n  ".join(warnings) if warnings else ""))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    if args.per_respondent:
        for who, verdicts in sorted(all_verdicts.items()):
            lines = hte.report(session_for(private, verdicts, who, private["stamp"]))
            print("\n" + "=" * 72)
            print("\n".join(lines))
            if args.out:
                safe = hte._UNSAFE.sub("-", who) or "respondent"
                (args.out / f"{safe}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pooled: every respondent's answers as one sample. Decision ids repeat across people,
    # which report() handles — it looks each verdict up by id and never assumes uniqueness.
    pooled = [v for vs in all_verdicts.values() for v in vs]
    lines = hte.report(session_for(
        private, pooled, f"{len(all_verdicts)} respondent(s)", private["stamp"]))
    # report() describes one judge over one set of decisions, so its "unjudged" count —
    # decisions minus verdicts — goes negative the moment several people answer the same
    # ones. Every rate above is a mean over verdicts and is unaffected; only that one
    # subtraction is meaningless here, so it is corrected in words rather than by patching
    # a module the CLI shares.
    lines.append(f"(pooled over {len(all_verdicts)} respondent(s) × {len(private['decisions'])} "
                 f"decision(s) = {len(pooled)} answer(s); the 'unjudged' count on the header "
                 "line assumes a single judge and can be ignored.)")
    pooled_session = session_for(private, pooled, f"{len(all_verdicts)} respondent(s)",
                                 private["stamp"])
    lines += implied_weights(pooled_session)
    lines += inter_rater(all_verdicts)
    lines += consensus(all_verdicts, by_id)
    if args.check_parity:
        lines += check_parity([
            (who, raw_responses[who], session_for(private, vs, who, private["stamp"]))
            for who, vs in sorted(all_verdicts.items()) if who in raw_responses])
    if warnings:
        lines += ["", "warnings", ""] + [f"  {w}" for w in warnings]
    lines += ["", "respondents", ""] + [
        f"  {who:>28}  {len(vs):>3} answer(s)" for who, vs in sorted(all_verdicts.items())]

    print("\n" + "=" * 72)
    print("\n".join(lines))
    if args.out:
        (args.out / "pooled.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written = ["pooled.txt"]
        if args.export:
            picks = hte.scored_picks(pooled_session)
            stops = hte.termination_rows(pooled_session)
            agg = hte.aggregate(picks, stops)
            written += csv_export(picks, stops, agg, args.out)
            written.append(latex_export(agg, args.out))
        print(f"\nwrote {', '.join(written)} to {args.out}")


if __name__ == "__main__":
    main()
