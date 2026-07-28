#!/usr/bin/env python
"""Freeze one human-trajectory evaluation into a static survey the website can serve.

`human_trajectory_web.py` is a local tool: it holds `runs/` open, samples a fresh session
per judge and enforces blinding inside the request handler. None of that survives on a
static host, so this script moves the same work to build time and splits the result in two:

    survey/data/survey.json     public — exactly what `public_decision` would have sent
    survey/data/truth.json      public, optional — the answer key, for the closing screen
    survey/private/decisions.json   NOT deployed — the full decisions, with the shuffle

Blinding is what the split buys. The public bundle carries no score, no status, no
`d2i_terminate` and no shuffle order, so the page cannot leak an answer it was never given;
the truth of *which slot is which candidate* stays in `private/`, which is why a returned
response only has to say "slot 3" for `score_survey.py` to resolve it later.

Every decision is built by importing `human_trajectory_eval` and calling its own
`build_decisions`, so the sampling, the early-stopped mix, the candidate pool and the
shuffle are the module's, not a re-implementation. One seed, one bundle: every respondent
judges the *same* decisions in the same order, which is what makes their answers comparable
and inter-rater agreement computable at all.

    python survey/build_survey.py                       # 20 decisions from every usable run
    python survey/build_survey.py -n 12 --seed 3
    python survey/build_survey.py --no-reveal           # omit truth.json, strict blinding
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))              # human_trajectory_eval.py sits at the repo root

import human_trajectory_eval as hte        # noqa: E402


def public_decision(decision: dict) -> dict:
    """One decision with every tell removed — the port of `human_trajectory_web.public_decision`.

    Candidates go out in *presentation* order carrying only their action and question. What
    is left out is the point: `score`, `status` and `breakdown` per candidate, and
    `d2i_terminate`, `threshold`, `n_selected`, `lambdas` and `order` per decision. `ask_stop`
    says whether the stop question applies at all, never what its answer is.
    """
    return {
        "id": decision["id"],
        "run": decision["run"],
        "node": decision["node_id"][:8],
        "depth": decision["depth"],
        "child_depth": decision["child_depth"],
        "goal": decision["goal"],
        "columns": decision["columns"],
        "path": decision["path"],
        "ask_stop": decision.get("d2i_terminate") is not None,
        "candidates": [
            {"slot": slot,
             "action": decision["candidates"][t]["action"],
             "question": decision["candidates"][t]["question"]}
            for slot, t in enumerate(decision["order"], 1)
        ],
    }


def truth_entry(decision: dict) -> dict:
    """The answer key for one decision — `human_trajectory_web.reveal_payload` without the
    verdict, since at build time nobody has answered yet. The page fills in the two
    verdict-shaped fields (which row was picked, what the human said about stopping) from
    the response it is holding, so the rank it reports is the same number the CLI prints."""
    slot_of = {t: k for k, t in enumerate(decision["order"], 1)}
    cands = decision["candidates"]
    return {
        "terms": [short for _, short in hte.TERMS],
        "lambdas": decision.get("lambdas"),
        "model_terminate": decision.get("d2i_terminate"),
        "terminate_reason": decision.get("terminate_reason") or "",
        "n": len(cands),
        "rows": [
            {
                "rank": 1 + sum(1 for o in cands if o["score"] > c["score"]),
                "slot": slot_of[i],
                "score": c["score"],
                "answered": c["status"] in hte.SELECTED,
                "action": c["action"],
                "question": c["question"],
                "terms": [(c.get("breakdown") or {}).get(key) for key, _ in hte.TERMS],
            }
            for i, c in enumerate(cands)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=REPO / "runs",
                   help="where the runs live (default: <repo>/runs)")
    p.add_argument("-n", type=int, default=20,
                   help="decision points in the survey, one per trajectory (default: 20)")
    p.add_argument("--seed", type=int, default=20260727,
                   help="RNG seed — fixed on purpose so every respondent sees the same "
                        "survey (default: 20260727)")
    p.add_argument("--min-candidates", type=int, default=2,
                   help="skip decision points offering fewer candidates (default: 2)")
    p.add_argument("--drop-unanswerable", action="store_true",
                   help="leave out candidates D2I selected but failed to answer")
    p.add_argument("--no-reveal", dest="reveal", action="store_false",
                   help="do not write truth.json — the closing screen then reports how much "
                        "was answered but not how well, and no answer key is ever published")
    p.add_argument("--title", default="Would you have asked the same question?",
                   help="headline shown on the survey's landing screen")
    p.add_argument("--out", type=Path, default=HERE,
                   help="where data/ and private/ are written (default: survey/)")
    args = p.parse_args()

    if args.min_candidates < 2:
        raise SystemExit("--min-candidates must be at least 2 — a pool of one is not a choice")

    run_dirs = hte.run_dirs(args.runs)
    if not run_dirs:
        raise SystemExit(f"no run dirs (nothing holding a report.json) under {args.runs}")

    decisions, notes = hte.build_decisions(
        run_dirs, args.n, random.Random(args.seed), args.min_candidates, args.drop_unanswerable
    )
    if not decisions:
        raise SystemExit(
            f"no decision points under {args.runs} — a run needs both repository.json and "
            "report.json's pruned_questions to reconstruct one")

    stopped = sum(1 for d in decisions if d.get("d2i_terminate") is True)
    with_cands = sum(1 for d in decisions if d["candidates"])
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    bundle = {
        "meta": {
            "title": args.title,
            "built_at": built_at,
            "seed": args.seed,
            "n_requested": args.n,
            "n_decisions": len(decisions),
            "n_runs": len(sorted({d["run_dir"] for d in decisions})),
            "n_early_stopped": stopped,
            "n_with_candidates": with_cands,
            "min_candidates": args.min_candidates,
            "drop_unanswerable": args.drop_unanswerable,
            "reveal": args.reveal,
        },
        "decisions": [public_decision(d) for d in decisions],
    }

    # The private half: the full decisions, verbatim from build_decisions, plus everything
    # score_survey.py needs to rebuild a session human_trajectory_eval.report() will accept.
    private = {
        "stamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "judge": None,
        "path": str(args.runs),
        "runs": sorted({d["run_dir"] for d in decisions}),
        "n_requested": args.n,
        "min_candidates": args.min_candidates,
        "drop_unanswerable": args.drop_unanswerable,
        # The survey never reveals mid-run, so the picks stay independent samples — this is
        # the flag report() reads to decide whether to caveat the rates.
        "feedback": False,
        "seed": args.seed,
        "notes": notes,
        "decisions": decisions,
        "verdicts": [],
    }

    data_dir, priv_dir = args.out / "data", args.out / "private"
    data_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "survey.json").write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    (priv_dir / "decisions.json").write_text(json.dumps(private, indent=1) + "\n", encoding="utf-8")

    truth_path = data_dir / "truth.json"
    if args.reveal:
        truth = {d["id"]: truth_entry(d) for d in decisions}
        truth_path.write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    elif truth_path.exists():
        truth_path.unlink()                # a --no-reveal build must not leave a stale key

    for t in notes:
        print(t)
    print(f"\n{len(decisions)} decision(s) from {bundle['meta']['n_runs']} run(s)"
          f"  ({stopped} early-stopped, {with_cands} with a candidate pool), seed {args.seed}")
    print(f"  public  {data_dir / 'survey.json'}  "
          f"({(data_dir / 'survey.json').stat().st_size / 1024:.0f} KB)")
    if args.reveal:
        print(f"  key     {truth_path}  ({truth_path.stat().st_size / 1024:.0f} KB)"
              "   — published; a determined respondent can read it early")
    else:
        print("  key     not written (--no-reveal)")
    print(f"  private {priv_dir / 'decisions.json'}   — keep this out of the deploy")


if __name__ == "__main__":
    main()
