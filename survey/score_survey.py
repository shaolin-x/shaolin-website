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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("responses", type=Path, nargs="+",
                   help="response .json files, or directories holding them")
    p.add_argument("--private", type=Path, default=HERE / "private" / "decisions.json",
                   help="the build's private bundle (default: survey/private/decisions.json)")
    p.add_argument("--per-respondent", action="store_true",
                   help="also print each respondent's own report, not just the pooled one")
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
    lines += inter_rater(all_verdicts)
    lines += consensus(all_verdicts, by_id)
    if warnings:
        lines += ["", "warnings", ""] + [f"  {w}" for w in warnings]
    lines += ["", "respondents", ""] + [
        f"  {who:>28}  {len(vs):>3} answer(s)" for who, vs in sorted(all_verdicts.items())]

    print("\n" + "=" * 72)
    print("\n".join(lines))
    if args.out:
        (args.out / "pooled.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out / 'pooled.txt'}")


if __name__ == "__main__":
    main()
