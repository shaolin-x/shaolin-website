#!/usr/bin/env python
"""The scoring's correctness tests. Plain asserts, no pytest required.

    python survey/tests/test_metrics.py

Four groups, in increasing order of how much they would catch:

1. estimator units — hand-worked values for kappa/AUROC/Youden/Wilson, plus a cross-check
   of the tie-corrected AUROC against sklearn where it is available.
2. exact invariants — the oracle and adversarial respondents pin every metric to its
   endpoint, which catches sign errors and off-by-ones immediately.
3. the baseline self-check — many random respondents must average out to each metric's own
   chance column. This validates a metric *and* its baseline simultaneously, with no ground
   truth, and is the test that would catch a per-pool expectation computed wrong under ties.
4. JS<->Python parity — `run_metrics.mjs` over the same responses, diffed to 1e-9.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
REPO = SURVEY.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import human_trajectory_eval as hte        # noqa: E402
from synth_responses import respond, session_for        # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILURES.append(label)


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b or (a is None and b is None)
    return abs(a - b) <= tol


# ----------------------------------------------------------------- 1. estimators

def test_estimators() -> None:
    print("\nestimators")
    # kappa: 3 of 4 agree against a 1/2 baseline -> (0.75 - 0.5) / 0.5
    check(close(hte.kappa([1, 1, 1, 0], [0.5] * 4), 0.5), "kappa hand value")
    check(hte.kappa([1, 1], [1.0, 1.0]) is None, "kappa undefined when chance is 1")
    check(close(hte.kappa([0, 0, 0, 0], [0.5] * 4), -1.0), "kappa -1 at total disagreement")

    # AUROC with a tie: one positive at 0.5, negatives at 0.4 and 0.5 -> (1 + 0.5) / 2
    check(close(hte.auroc([0.5, 0.4, 0.5], [1, 0, 0]), 0.75), "auroc half credit for ties")
    check(hte.auroc([0.1, 0.2], [1, 1]) is None, "auroc needs both classes")
    check(close(hte.auroc([1, 0], [1, 0]), 1.0), "auroc perfect")
    check(close(hte.auroc([0, 1], [1, 0]), 0.0), "auroc inverted")

    # weighted mean: agreement only on the heavily-weighted row
    check(close(hte.wmean([1.0, 0.0], [3.0, 1.0]), 0.75), "wmean hand value")
    check(close(hte.wmean([1.0, 0.0], [0.0, 0.0]), 0.5), "wmean falls back when weights are 0")

    # Youden: continue predicted at score >= cut; a clean split at 0.5
    got = hte.youden([0.6, 0.7, 0.2, 0.1], [1, 1, 0, 0])
    check(got is not None and got[1] <= 0.6 <= got[2], "youden brackets the separating cut")

    lo, hi = hte.wilson(7, 10)
    check(lo < 0.7 < hi, "wilson brackets the point estimate")
    # Reference computed from the closed form with scipy's z; matches statsmodels'
    # proportion_confint(7, 10, method="wilson") where that is installed.
    check(close(lo, 0.396778147461, 1e-9) and close(hi, 0.892208732594, 1e-9),
          "wilson reference value 7/10")
    check(hte.wilson(0, 8)[0] == 0.0, "wilson clamps at 0")
    check(hte.wilson(8, 8)[1] == 1.0, "wilson clamps at 1")

    try:
        from sklearn.metrics import roc_auc_score
        scores = [0.3, 0.3, 0.54, 0.66, 0.2, 0.54]
        labels = [1, 0, 1, 1, 0, 0]
        check(close(hte.auroc(scores, labels), float(roc_auc_score(labels, scores)), 1e-12),
              "auroc matches sklearn on tied input")
    except ImportError:                                  # pragma: no cover
        print("  skip  sklearn not installed")


# ----------------------------------------------------------------- 2. invariants

def _agg(private: dict, policy: str, seed: int = 0):
    session = session_for(private, respond(private, policy, seed))
    picks = hte.scored_picks(session)
    stops = hte.termination_rows(session)
    return {a["label"]: a for a in hte.aggregate(picks, stops)}, picks, stops


def test_invariants(private: dict) -> None:
    print("\nexact invariants")
    oracle, picks, stops = _agg(private, "oracle")
    by = lambda t, k: next(v for lbl, v in t.items() if k in lbl)          # noqa: E731

    check(close(by(oracle, "κ_sel")["value"], 1.0), "oracle: kappa_sel = 1")
    check(close(by(oracle, "attainment")["value"], 1.0), "oracle: attainment = 1")
    check(close(by(oracle, "standardized regret")["value"], 0.0), "oracle: regret = 0")
    check(close(by(oracle, "within-noise")["value"], 1.0), "oracle: within-noise = 1")
    check(close(by(oracle, "action agreement")["value"], 1.0), "oracle: action agreement = 1")
    check(close(by(oracle, "margin-weighted")["value"], 1.0), "oracle: margin-weighted = 1")
    check(close(by(oracle, "unweighted")["value"], 1.0), "oracle: unweighted stop = 1")

    adv, apicks, astops = _agg(private, "adversarial")
    check(close(by(adv, "attainment")["value"], 0.0), "adversarial: attainment = 0")
    check(close(by(adv, "margin-weighted")["value"], 0.0), "adversarial: margin-weighted = 0")
    check(close(by(adv, "unweighted")["value"], 0.0), "adversarial: unweighted stop = 0")
    check(by(adv, "κ_sel")["value"] <= 1e-12, "adversarial: kappa_sel <= 0")
    # top-1 is 0 only where no pool has a tie at the top; assert conditionally so a tied
    # pool does not make this flake.
    if all(r["e_top"] * r["n"] == 1 for r in apicks):
        check(close(by(adv, "raw agreement")["value"], 0.0), "adversarial: raw agreement = 0")
    check(by(adv, "standardized regret")["value"] >= by(oracle, "standardized regret")["value"],
          "adversarial regret >= oracle regret")

    auc_o, auc_a = by(oracle, "AUROC")["value"], by(adv, "AUROC")["value"]
    if auc_o is not None and auc_a is not None:
        check(close(auc_o, 1.0) and close(auc_a, 0.0), "oracle AUROC = 1, adversarial = 0")

    print("\nmonotonicity oracle > random > adversarial")
    rnd, _, _ = _agg(private, "random", 12345)
    for key in ("attainment", "κ_sel"):
        o, r, a = by(oracle, key)["value"], by(rnd, key)["value"], by(adv, key)["value"]
        check(o >= r >= a - 1e-12, f"{key}: {o:.3f} >= {r:.3f} >= {a:.3f}")


# ----------------------------------------------------------------- 3. baseline self-check

def test_baselines(private: dict, trials: int = 400) -> None:
    """A uniformly random respondent must, on average, score exactly the chance column."""
    print(f"\nbaseline self-check ({trials} random respondents)")
    # (metric field, its chance field) — `is_top` pairs with `e_top`, not `e_is_top`, so
    # the mapping is spelled out rather than derived by prefixing.
    keys = [("is_top", "e_top"), ("attainment", "e_attainment"), ("d_std", "e_d_std"),
            ("within_noise", "e_within_noise"), ("action_hit", "e_action_hit")]
    obs = {k: [] for k, _ in keys}
    exp = {k: [] for k, _ in keys}
    aurocs = []
    for seed in range(trials):
        session = session_for(private, respond(private, "random", seed))
        picks = hte.scored_picks(session)
        stops = hte.termination_rows(session)
        for k, ek in keys:
            vals = [r[k] for r in picks if r.get(k) is not None]
            base = [r[ek] for r in picks if r.get(ek) is not None]
            assert base, f"chance column {ek} is missing from every pick row"
            if vals:
                obs[k].append(hte._mean(vals))
            exp[k].append(hte._mean(base))
        got = hte.estimate({"key": "utility", "base": None, "est": "auroc",
                            "only": "margin"}, stops)[0]
        if got is not None:
            aurocs.append(got)

    for k, _ in keys:
        mo, me = hte._mean(obs[k]), hte._mean(exp[k])
        sd = math.sqrt(hte._mean([(x - mo) ** 2 for x in obs[k]]))
        se = sd / math.sqrt(len(obs[k])) or 1e-12
        z = abs(mo - me) / se
        check(z < 4.0, f"{k}: random {mo:.4f} vs chance {me:.4f}  (z={z:.2f})")

    if aurocs:
        mo = hte._mean(aurocs)
        sd = math.sqrt(hte._mean([(x - mo) ** 2 for x in aurocs]))
        se = (sd / math.sqrt(len(aurocs))) or 1e-12
        check(abs(mo - 0.5) / se < 4.0, f"AUROC: random {mo:.4f} vs chance 0.5000")


# ----------------------------------------------------------------- 4. JS parity

def test_parity(private: dict, survey_path: Path, truth_path: Path) -> None:
    print("\nJS <-> Python parity")
    if not shutil.which("node"):
        print("  skip  node not installed")
        return
    runner = HERE / "run_metrics.mjs"
    for policy, seed in (("oracle", 0), ("adversarial", 0), ("random", 1), ("random", 2),
                         ("lazy", 3)):
        response = respond(private, policy, seed)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(response, fh)
            path = Path(fh.name)
        try:
            proc = subprocess.run(
                ["node", str(runner), str(survey_path), str(truth_path), str(path)],
                capture_output=True, text=True)
            if proc.returncode:
                check(False, f"{policy}/{seed}: node failed — {proc.stderr.strip()[:200]}")
                continue
            js = json.loads(proc.stdout)
        finally:
            path.unlink(missing_ok=True)

        session = session_for(private, response)
        py = hte.aggregate(hte.scored_picks(session), hte.termination_rows(session))
        ok = len(js["summary"]) == len(py)
        worst = 0.0
        for a, b in zip(js["summary"], py):
            for field in ("value", "chance", "lo", "hi"):
                x, y = a[field], b[field]
                if x is None or y is None:
                    ok = ok and x is None and y is None
                else:
                    worst = max(worst, abs(x - y))
                    ok = ok and abs(x - y) <= 1e-9
            ok = ok and a["n"] == b["n"]
        check(ok, f"{policy}/{seed}: all cells agree (worst delta {worst:.2e})")


# ----------------------------------------------------------------- 5. the download

def test_download(private: dict, survey_path: Path, truth_path: Path) -> None:
    """The downloaded file must be readable on its own — provenance, candidates, scores and
    the human's choice, without `runs/` or the private bundle to hand."""
    print("\ndownloaded file is self-describing")
    if not shutil.which("node"):
        print("  skip  node not installed")
        return
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    response = respond(private, "random", 5)

    script = (
        "const M=require(%r);const fs=require('fs');"
        "const s=JSON.parse(fs.readFileSync(%r,'utf8'));"
        "const t=JSON.parse(fs.readFileSync(%r,'utf8'));"
        # `node -e` does not put the script in argv, so the first extra arg is argv[1].
        "const v=JSON.parse(process.argv[1]).verdicts;"
        "const rows=M.rowsFor(v,t,s.decisions);"
        "const d=M.detail(v,t,s.decisions,s.runs||{},(s.meta.metrics||{}).terms||[]);"
        "console.log(JSON.stringify({decisions:d,counts:M.counts(d,rows),"
        "headline:M.headlines(M.aggregate(rows,s.meta.metrics.table))}));"
        % (str(SURVEY / "metrics.js"), str(survey_path), str(truth_path))
    )
    proc = subprocess.run(["node", "-e", script, json.dumps(response)],
                          capture_output=True, text=True)
    if proc.returncode:
        check(False, f"node failed — {proc.stderr.strip()[:200]}")
        return
    got = json.loads(proc.stdout)
    details, counts_, head = got["decisions"], got["counts"], got["headline"]

    check(len(details) == len(private["decisions"]),
          f"one record per decision ({len(details)})")
    check(all(d["source"].get("run") and d["source"].get("run_dir")
              and d["source"].get("depth") is not None for d in details),
          "every record names its run, file and depth")
    check(all(d["source"].get("trajectory") for d in details if d["termination"]),
          "every stop record names its trajectory")

    with_cands = [d for d in details if d["selection"]]
    check(bool(with_cands), f"{len(with_cands)} record(s) carry a candidate pool")
    check(all(len(d["selection"]["candidates"]) == d["selection"]["n_candidates"]
              for d in with_cands), "candidate lists are complete")
    check(all(c["model_score"] is not None and len(c["score_breakdown"]) == 5
              for d in with_cands for c in d["selection"]["candidates"]),
          "every candidate carries a score and a 5-term breakdown")
    check(all(sum(1 for c in d["selection"]["candidates"] if c["model_top"]) >= 1
              for d in with_cands), "every pool marks D2I's top candidate")

    # The human's answer must be recoverable from the record alone, and must agree with the
    # verdict it came from.
    by_id = {d["id"]: d for d in details}
    mismatched = 0
    for v in response["verdicts"]:
        rec = by_id.get(v["decision"])
        if rec is None or not rec["selection"]:
            continue
        picked = [c["slot"] for c in rec["selection"]["candidates"] if c["picked_by_human"]]
        if v["slot"] is None:
            mismatched += 1 if picked else 0
        else:
            mismatched += 0 if picked == [v["slot"]] else 1
    check(mismatched == 0, "the human's pick is flagged on exactly the right candidate")

    answered = sum(1 for v in response["verdicts"] if v["slot"] is not None)
    check(counts_["candidate_selection"]["answered"] == answered,
          f"candidate-selection count matches the verdicts ({answered})")
    stops = sum(1 for v in response["verdicts"] if v["terminate"] is not None)
    check(counts_["termination_selection"]["answered"] == stops,
          f"termination count matches the verdicts ({stops})")
    check(counts_["termination_selection"]["judged"]
          + counts_["termination_selection"]["inferred"]
          == counts_["termination_selection"]["asked"],
          "judged + inferred accounts for every stop question asked")
    check(counts_["candidates_shown"]
          == sum(d["selection"]["n_candidates"] for d in with_cands),
          f"candidates_shown totals the pools ({counts_['candidates_shown']})")

    check(all(k in head for k in ("kappa_sel", "utility_attainment", "standardized_regret",
                                 "margin_weighted_agreement", "auroc", "implied_threshold")),
          "every headline measure is reported under a stable key")


def main() -> None:
    private_path = SURVEY / "private" / "decisions.json"
    survey_path = SURVEY / "data" / "survey.json"
    truth_path = SURVEY / "data" / "truth.json"
    if not private_path.is_file():
        raise SystemExit("no private bundle — run `python survey/build_survey.py` first")
    private = json.loads(private_path.read_text(encoding="utf-8"))

    test_estimators()
    test_invariants(private)
    test_baselines(private)
    if survey_path.is_file() and truth_path.is_file():
        test_parity(private, survey_path, truth_path)
        test_download(private, survey_path, truth_path)

    print(f"\n{'FAILED: ' + '; '.join(FAILURES) if FAILURES else 'all checks passed'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
