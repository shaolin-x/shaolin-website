#!/usr/bin/env python
"""Synthetic respondents, in exactly the shape `index.html`'s download() produces.

No human has taken the survey yet, so every correctness claim about the scoring has to be
established against answers whose right score is known in advance. Four policies:

    oracle       always the top-scoring candidate, always matches the model's stop verdict
    adversarial  always the worst-scoring candidate, always the opposite stop verdict
    random(seed) uniform over slots, fair coin on stopping
    lazy(p)      random, but skips each decision with probability p

`oracle` and `adversarial` pin the metrics to their exact endpoints; `random` is what the
chance baselines must converge to, which is the test that validates a metric and its own
baseline at the same time.

    python survey/tests/synth_responses.py --policy random --seed 4 -o /tmp/r.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
REPO = SURVEY.parent
sys.path.insert(0, str(REPO))

import human_trajectory_eval as hte        # noqa: E402


def respond(private: dict, policy: str, seed: int = 0, skip_p: float = 0.3) -> dict:
    """One response file. `slot` is what the page records; `true_index` is deliberately
    absent, exactly as in a real response — `score_survey.resolve()` recovers it."""
    rng = random.Random(seed)
    verdicts = []
    for d in private["decisions"]:
        cands, order = d["candidates"], d["order"]
        asked_stop = d.get("d2i_terminate") is not None
        model_stop = d.get("d2i_terminate")

        if policy == "lazy" and rng.random() < skip_p:
            verdicts.append({"decision": d["id"], "terminate": None, "slot": None,
                             "choice": "s", "note": "", "ts": ""})
            continue

        if not asked_stop:
            terminate = None
        elif policy == "oracle":
            terminate = bool(model_stop)
        elif policy == "adversarial":
            terminate = not bool(model_stop)
        else:
            terminate = rng.choice([True, False])

        true_index = None
        if cands:
            if policy == "oracle":
                true_index = 0                                   # candidates are score-sorted
            elif policy == "adversarial":
                true_index = len(cands) - 1
            else:
                true_index = rng.randrange(len(cands))
        slot = (order.index(true_index) + 1) if true_index is not None else None
        verdicts.append({
            "decision": d["id"], "terminate": terminate, "slot": slot,
            "choice": ("t" if terminate else "c") if slot is None else str(slot),
            "note": "", "ts": "",
        })
    return {
        "id": f"synthetic-{policy}-{seed}",
        "name": f"synthetic:{policy}",
        "built_at": private.get("stamp"),
        "started_at": "", "submitted_at": None,
        "verdicts": verdicts,
    }


def session_for(private: dict, response: dict) -> dict:
    """The response resolved back into a session `human_trajectory_eval` can score."""
    by_id = {d["id"]: d for d in private["decisions"]}
    verdicts = []
    for v in response["verdicts"]:
        d = by_id[v["decision"]]
        slot = v["slot"]
        verdicts.append({
            "decision": v["decision"], "terminate": v["terminate"], "slot": slot,
            "true_index": d["order"][slot - 1] if slot is not None else None,
            "choice": v.get("choice", ""), "note": "", "ts": "",
        })
    session = dict(private)
    session["verdicts"] = verdicts
    session["judge"] = response["name"]
    return session


def attach_results(response: dict, survey_path: Path, truth_path: Path) -> dict:
    """Add the `results` block the real page attaches, by running metrics.js over the same
    answers. Lets `score_survey.py --check-parity` be exercised without a browser."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(response, fh)
        path = Path(fh.name)
    try:
        proc = subprocess.run(
            ["node", str(HERE / "run_metrics.mjs"), str(survey_path), str(truth_path), str(path)],
            capture_output=True, text=True, check=True)
    finally:
        path.unlink(missing_ok=True)
    js = json.loads(proc.stdout)
    response["results"] = {
        "engine": "metrics.js/" + js["engine"],
        "metrics_version": hte.METRICS_VERSION,
        "summary": js["summary"],
        "terms": js["terms"],
        "criteria": js["criteria"],
        "n_skipped": js["n_skipped"],
    }
    return response


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--private", type=Path, default=SURVEY / "private" / "decisions.json")
    p.add_argument("--policy", default="random",
                   choices=["oracle", "adversarial", "random", "lazy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-p", type=float, default=0.3)
    p.add_argument("--name", help="respondent name (default: synthetic:<policy>)")
    p.add_argument("--with-results", action="store_true",
                   help="also attach the metrics.js results block, as the page does")
    p.add_argument("-o", "--out", type=Path)
    args = p.parse_args()

    private = json.loads(args.private.read_text(encoding="utf-8"))
    response = respond(private, args.policy, args.seed, args.skip_p)
    if args.name:
        response["name"] = args.name
    response["respondent"] = {"name": response["name"]}
    if args.with_results:
        response = attach_results(response, SURVEY / "data" / "survey.json",
                                  SURVEY / "data" / "truth.json")
    text = json.dumps(response, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"{args.policy} response -> {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
