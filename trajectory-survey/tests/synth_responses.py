#!/usr/bin/env python3
"""Synthetic respondents, for testing the scoring without asking anyone anything.

Four kinds, each of which pins down a different end of the scale:

  oracle       answers the judge's rounded mean every time — every agreement measure
               must be at its ceiling and MAE at its floor
  adversarial  answers 6 - judge, the mirror image — correlations must go strongly
               negative
  random       uniform 1-5 — every measure must sit at its own chance column, which is
               what validates the chance columns themselves
  noisy        the judge plus a small integer jitter — the shape a real respondent is
               expected to have, used to check nothing blows up in between

    python3 tests/synth_responses.py --kind noisy -o responses/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agreement import round1  # noqa: E402

KINDS = ("oracle", "adversarial", "random", "noisy")


def answer(kind: str, judge: float, rng: random.Random) -> int:
    if kind == "oracle":
        return max(1, min(5, round1(judge)))
    if kind == "adversarial":
        return max(1, min(5, 6 - round1(judge)))
    if kind == "random":
        return rng.randint(1, 5)
    return max(1, min(5, round1(judge) + rng.choice([-1, 0, 0, 0, 1])))


def make(kind: str, survey: dict, truth: dict, seed: int, name: str | None = None) -> dict:
    rng = random.Random(seed)
    started = datetime(2026, 8, 1, 12, 0, 0)
    answers = []
    for k, item in enumerate(survey["items"]):
        scores = {}
        for crit in item["criteria"]:
            block = truth["items"][item["id"]]["scores"].get(crit)
            judge = block["score"] if block else 3.0
            scores[crit] = answer(kind, judge, rng)
        answers.append({
            "item": item["id"],
            "dataset": item["dataset"],
            "scores": scores,
            "skipped": False,
            "note": "",
            "answered_at": (started + timedelta(minutes=2 * (k + 1))).isoformat(),
            "ms": 120000,
        })
    return {
        "id": f"synth-{kind}-{seed}",
        "started_at": started.isoformat(),
        "finished_at": (started + timedelta(minutes=2 * len(answers))).isoformat(),
        "respondent": {"name": name or f"{kind}-{seed}", "email": "", "background": ""},
        "survey": {"built_at": survey["meta"]["built_at"], "seed": survey["meta"]["seed"],
                   "n_items": survey["meta"]["n_items"],
                   "criteria": survey["meta"]["criteria"],
                   "geval": survey["meta"]["geval"]},
        "synthetic": kind,
        "answers": answers,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=KINDS + ("all",), default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="respondents per kind")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write files here (default: print one to stdout)")
    args = p.parse_args()

    survey = json.loads((args.data / "survey.json").read_text(encoding="utf-8"))
    truth = json.loads((args.data / "truth.json").read_text(encoding="utf-8"))
    kinds = KINDS if args.kind == "all" else (args.kind,)

    made = []
    for kind in kinds:
        for i in range(args.n):
            made.append(make(kind, survey, truth, args.seed + i))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for R in made:
            path = args.out / f"{R['id']}.json"
            path.write_text(json.dumps(R, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {path}")
    else:
        print(json.dumps(made[0], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
