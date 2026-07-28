#!/usr/bin/env python
"""Human trajectory evaluation — would a person pick the same next question as D2I?

D2I makes two choices as it searches: whether a trajectory is still worth extending (the
multi-agent continuation judge, §5.1) and, if so, which question to ask next (the fused
§6.3 score over S_LLM, S_semantic, S_coverage, S_impact, S_trajectory). This script replays
both to a human, in that order.

Each evaluation step shows one *decision point* from a finished run — the trajectory as it
stood at some depth — and asks:

    1. terminate or continue?  Answered on the trajectory alone, before any candidate is
       shown, which is the position the model's judge is in. `terminate` ends the decision.
    2. which question next?    The full pool that competed at that node, shuffled.

At the end it reports agreement on both: against the judge's early-stop verdict, and
against the scorer's ranking.

Early-stopped nodes are deliberately mixed into the sample. A run holds one or two of them
against a dozen ordinary nodes — and, having stopped, they carry no candidates at all, so
they are invisible to the second question — which is why a uniform draw would leave the
first question with "continue" as its answer nearly every time.

A decision point is reconstructed offline from the run's own artifacts:

    candidates(P) = {r in repository.json.records    : r.parent_id == P}   # answered
                  + {q in report.json.pruned_questions : q.parent_id == P} # never answered

Both halves carry the same fused score, so sorting the union by score descending IS D2I's
ranking at that node (it is the same merge `d2i/run.py:_tree_children` renders in
tree.txt). The trajectory *status* at P is the root-to-P path, walked through `parent_id`
— `trajectory_id` is not a path key, since siblings of one parent carry different ones.

The judge sees only the goal, the schema columns, the path so far, and the shuffled
candidate questions. Scores, ranks, which candidates D2I answered, and the run's
`global_summary` are all withheld — any of them would give the answer away.

Sampling: one decision point per trajectory, the trajectory drawn at random and the depth
drawn at random along it, deduplicated across the sample (trajectories share prefixes).
Every root-to-leaf path is eligible, including branches the beam later abandoned.

Depth 0 is out of scope: choosing which *base insight* to open is a different decision
over `base_insights`, which are not questions and have no candidate text.

The session is written to disk after every pick, so it can be interrupted (`q`, or Ctrl-C)
and resumed with `--resume`. `u` undoes the previous pick; every metric is recomputed from
the pick list at report time, so an undo leaves no trace.

Usage (from the repo root):
    python eval/human_trajectory_eval.py runs/20260727-201135_d2i_bench/carsales-easy 5
    python eval/human_trajectory_eval.py runs/20260726-030731_insighteval 20 --seed 7
    python eval/human_trajectory_eval.py RUNS 10 --no-feedback --judge alex
    python eval/human_trajectory_eval.py --resume eval/human_trajectory_eval/<stamp>_<run>/session.json
    python eval/human_trajectory_eval.py --report eval/human_trajectory_eval/<stamp>_<run>/session.json

PATH is either a single run dir (one holding report.json) or any parent of run dirs — the
d2i_bench, `data-N` and `flag-N` layouts are all found by looking for report.json, never
by directory name. This script imports nothing from d2i and reads only the JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))          # so the sibling evaluators import as modules

OUT_DIR = EVAL_DIR / "human_trajectory_eval"
OUT_FILE = EVAL_DIR / "results_human_trajectory_eval.txt"

#: Candidate statuses meaning "D2I chose to answer this one". `unanswerable` belongs here:
#: it marks a candidate the search selected and then failed to execute (d2i/graph.py:317),
#: not one that lost the ranking — on disk they sit above every `not_selected` at their
#: level. `--drop-unanswerable` removes them for a stricter reading.
SELECTED = ("committed", "unanswerable")

#: Anything not alphanumeric in a session directory label.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ----------------------------------------------------------------- loading a run

def run_dirs(root: Path) -> list[Path]:
    """Every run directory at or under `root`, identified by holding a report.json.

    Looking for the artifact rather than for a `data-N`/`flag-N`/`<dataset>` name means the
    d2i_bench, insighteval and insightbench layouts all work unchanged, and a renamed or
    relocated directory still resolves.
    """
    if (root / "report.json").is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    return sorted({p.parent for p in root.rglob("report.json")})


def load_run(run_dir: Path) -> tuple[dict, dict] | str:
    """`(report, repository)` for a run dir, or a string saying why it is unusable.

    Both files are required. `repository.json` is the only artifact carrying the *answered*
    candidates' scores (report.json's trajectory steps have no score), and
    `pruned_questions` is the only record of the losing candidates — without it a node's
    pool would hold winners only and the human could not be wrong.
    """
    try:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"report.json unreadable ({e.__class__.__name__})"
    repo_path = run_dir / "repository.json"
    if not repo_path.is_file():
        return "no repository.json (the run predates that artifact)"
    try:
        repository = json.loads(repo_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"repository.json unreadable ({e.__class__.__name__})"
    if not report.get("pruned_questions"):
        return "report.json has no pruned_questions, so the losing candidates were never recorded"
    return report, repository


def build_index(repository: dict) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """`(records by id, child ids by parent id)` — the run's record tree."""
    records = {r["id"]: r for r in repository.get("records", [])}
    children: dict[str, list[str]] = defaultdict(list)
    for r in records.values():
        if r.get("parent_id"):
            children[r["parent_id"]].append(r["id"])
    return records, children


def path_to(rid: str, records: dict[str, dict]) -> list[dict]:
    """The root-to-`rid` record path, walked up through `parent_id`."""
    out: list[dict] = []
    seen: set[str] = set()
    cur: str | None = rid
    while cur and cur in records and cur not in seen:
        seen.add(cur)
        out.append(records[cur])
        cur = records[cur].get("parent_id")
    return list(reversed(out))


def trajectories(records: dict[str, dict], children: dict[str, list[str]]) -> list[list[str]]:
    """Every root-to-leaf id path through the record tree.

    Includes the branches the beam abandoned, and a depth-0 record with no answered child
    is a one-element path — still a decision point, since its losing candidates are all in
    `pruned_questions`.
    """
    roots = sorted(
        r["id"] for r in records.values()
        if not r.get("parent_id") or r["parent_id"] not in records
    )
    paths: list[list[str]] = []

    def walk(rid: str, prefix: list[str]) -> None:
        here = prefix + [rid]
        kids = sorted(children.get(rid, []))
        if not kids:
            paths.append(here)
            return
        for kid in kids:
            walk(kid, here)

    for root in roots:
        walk(root, [])
    return paths


def _numeric(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


#: The §6.3 terms, in the order d2i/run.py's `_fmt_breakdown` prints them.
TERMS = (("S_llm", "llm"), ("S_semantic", "sem"), ("S_coverage", "cov"),
         ("S_impact", "imp"), ("S_trajectory", "traj"))


def breakdowns_from_trace(run_dir: Path) -> dict[tuple[str, float], dict]:
    """`{(question, score): score_breakdown}` from the run's trace.

    `repository.json` does not persist `score_breakdown` for the candidates that were
    answered (`_record_dict` omits it), so the per-term decomposition of exactly the
    winning candidates would otherwise be missing. The `scorer.score` tool events in
    trace.jsonl carry it at full precision for the whole batch, winners included. Keyed by
    (question, score) rather than by question alone, since the same question text can be
    proposed at two different nodes and the S_trajectory term differs between them.

    Returns `{}` when there is no trace — the breakdown column then reads `—`.
    """
    path = run_dir / "trace.jsonl"
    if not path.is_file():
        return {}
    out: dict[tuple[str, float], dict] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"scorer.score"' not in line:      # cheap prefilter: traces run to MBs
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                for q in event.get("output") or []:
                    if isinstance(q, dict) and q.get("score_breakdown") and _numeric(q.get("score")):
                        out[(str(q.get("text") or ""), float(q["score"]))] = q["score_breakdown"]
    except OSError:
        return {}
    return out


def lambdas(run_dir: Path) -> dict[str, float] | None:
    """The λ weights the run scored with, for the legend under the ranking."""
    try:
        search = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["search"]
    except (OSError, ValueError, KeyError):
        return None
    got = {short: search.get(f"lambda_{long}")
           for long, short in (("llm", "llm"), ("semantic", "sem"), ("coverage", "cov"),
                               ("impact", "imp"), ("trajectory", "traj"))}
    return got if all(_numeric(v) for v in got.values()) else None


def candidate_pool(
    parent_id: str,
    pruned_by_parent: dict[str, list[dict]],
    records: dict[str, dict],
    children: dict[str, list[str]],
    traced: dict[tuple[str, float], dict],
    *,
    drop_unanswerable: bool = False,
) -> tuple[list[dict], int]:
    """`(pool, n_dropped)` — every candidate that competed to be `parent_id`'s child.

    Sorted by `(-score, question)`, the same key `d2i/run.py:_tree_children` sorts the
    merged list by, so the pool order reproduces tree.txt exactly. Only `action`,
    `question`, `score` and `status` are carried: an answered candidate also has a `label`
    and a pruned one never does, so copying anything else would betray which is which.
    """
    def traced_terms(question: str, score: object) -> dict | None:
        return traced.get((question, float(score))) if _numeric(score) else None

    pool: list[dict] = []
    for cid in children.get(parent_id, []):
        r = records[cid]
        pool.append({
            "action": r.get("action") or "base",
            "question": r.get("question") or "",
            "score": r.get("score"),
            "status": "committed",
            # Never persisted on a record — recovered from the scorer's trace event.
            "breakdown": traced_terms(r.get("question") or "", r.get("score")),
        })
    for q in pruned_by_parent.get(parent_id, []):
        status = q.get("status") or "not_selected"
        if drop_unanswerable and status == "unanswerable":
            continue
        pool.append({
            "action": q.get("action") or "?",
            "question": q.get("question") or "",
            "score": q.get("score"),
            "status": status,
            "breakdown": q.get("score_breakdown")
                         or traced_terms(q.get("question") or "", q.get("score")),
        })
    kept = [c for c in pool if _numeric(c["score"])]
    kept.sort(key=lambda c: (-float(c["score"]), c["question"]))
    return kept, len(pool) - len(kept)


_UTILITY_RE = re.compile(r"continuation utility\s*([\d.]+)\s*<\s*([\d.]+)")


def _utility_from_reason(reason: str) -> tuple[float | None, float | None]:
    """`(utility, threshold)` scraped out of a termination reason, or `(None, None)`.

    Run folders written before `report.json` carried a structured `continuation` object
    still print the judge's arithmetic into the reason prose — `"continuation utility 0.30
    < 0.35 — …"`. That recovers the utility for *terminated* nodes on those runs. It cannot
    recover anything for a node the judge let continue: nothing is printed in that case,
    which is exactly the gap the structured field closes.
    """
    m = _UTILITY_RE.search(reason or "")
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:                                   # pragma: no cover - regex guards it
        return None, None


def continuations(report: dict) -> dict[str, dict]:
    """`{head record id: the judge's continuation ruling}` from `report.json`.

    Newer runs hang a `continuation` object off the trajectory step the judge ruled on::

        {"depth": 1, "head_id": "7a321b94ef50493c", "utility": 0.3, "threshold": 0.35,
         "decision": "terminate", "criteria": {...}, "rationale": "..."}

    `head_id` is a record id, so this joins straight onto `records`. It is the only source
    that carries a *continue* verdict with a utility attached — `terminated` only ever
    describes the trajectories that stopped — which is what makes the margin and the ROC
    metrics possible at all.

    Any top-level list whose items carry a `trajectory` list is walked, rather than the two
    section names in use today, so a future `pruned[]` that grows trajectories is picked up
    without another edit. Older runs have the key nowhere and get `{}`.
    """
    out: dict[str, dict] = {}
    for value in report.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("trajectory"), list):
                continue
            for step in item["trajectory"]:
                cont = step.get("continuation") if isinstance(step, dict) else None
                if not isinstance(cont, dict):
                    continue
                head = cont.get("head_id")
                # First ruling wins: a trajectory can be listed twice (it is both an
                # insight and the head of a terminated line) and both copies are the same
                # judge call, so this is de-duplication rather than a real conflict.
                if isinstance(head, str) and head not in out:
                    out[head] = cont
    return out


def termination_verdicts(
    report: dict, records: dict[str, dict], children: dict[str, list[str]],
    conts: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """`{record id: {terminate, reason, source, utility, threshold, criteria}}`.

    Three verdicts are recoverable, and they are *not* equally good evidence — which is why
    `source` travels with every one of them:

    * `"judged"` — `report.json` carries the judge's own ruling for this node, utility and
      criteria included. The only source that yields a judged **continue**, and the only
      one the margin-weighted and ROC metrics may use.
    * `"terminated"` — the node heads a trajectory in `report.json.terminated`, matched
      through `trajectory_id`. Such a node has no children *and* no pruned questions: the
      search stopped before proposing a follow-up, which is why it is invisible to the
      candidate question and needs this one. Utility is scraped from the reason prose.
    * `"inferred"` — at least one child was answered, so the search did go on from here.
      Nothing was actually ruled: the beam moved on. There is no utility, and scoring a
      human against it is a weaker claim than the other two, so callers report it apart.

    Everything else is left out. A node whose branch merely lost the level's ranking was
    never judged at all, so scoring a human "terminate" against it would be scoring against
    nothing.
    """
    out: dict[str, dict] = {}

    def put(rid: str, terminate: bool, reason: str, source: str,
            utility: float | None, threshold: float | None, criteria: dict | None) -> None:
        out[rid] = {
            "terminate": terminate, "reason": reason, "source": source,
            "utility": utility, "threshold": threshold, "criteria": criteria,
        }

    for rid, cont in (conts or {}).items():
        if rid not in records:
            continue
        util, thr = cont.get("utility"), cont.get("threshold")
        decision = str(cont.get("decision") or "").lower()
        terminate = decision == "terminate"
        if not decision and _numeric(util) and _numeric(thr):
            terminate = float(util) < float(thr)          # ruling implied by the arithmetic
        put(rid, terminate, str(cont.get("rationale") or ""), "judged",
            float(util) if _numeric(util) else None,
            float(thr) if _numeric(thr) else None,
            cont.get("criteria") if isinstance(cont.get("criteria"), dict) else None)

    for t in report.get("terminated", []):
        heads = [r for r in records.values() if r.get("trajectory_id") == t.get("id")]
        if heads:
            head = max(heads, key=lambda r: r.get("depth", 0))
            if head["id"] in out:                        # a judged ruling already covers it
                continue
            reason = str(t.get("reason") or "")
            util, thr = _utility_from_reason(reason)
            put(head["id"], True, reason, "terminated", util, thr, None)

    for rid in records:
        if children.get(rid) and rid not in out:
            put(rid, False, "", "inferred", None, None, None)
    return out


def level_thresholds(report: dict, records: dict[str, dict]) -> dict[int, float]:
    """`{child depth: the lowest score D2I still chose to answer at that depth}`.

    Selection is global per level, not per node (d2i/agents/controller/agent.py:200), so a
    whole node can lose every slot. This threshold is what makes "would the human's pick
    have been answered?" well defined at such a node.
    """
    selected: dict[int, list[float]] = defaultdict(list)
    for r in records.values():
        if r.get("depth", 0) > 0 and _numeric(r.get("score")):
            selected[int(r["depth"])].append(float(r["score"]))
    for q in report.get("pruned_questions", []):
        if q.get("status") == "unanswerable" and _numeric(q.get("score")):
            selected[int(q["depth"])].append(float(q["score"]))
    return {d: min(v) for d, v in selected.items() if v}


def fmt_stat(stat: dict | None) -> str:
    """A compact `p=… eff=…` rendering of a step's statistic — a port of d2i/run.py's
    `_fmt_stat`. Called at build time so the session stores a string: an evidence dict can
    hold NaN, which `json.dumps` writes as a bare `NaN` that is not valid JSON."""
    if not stat:
        return "stat —"
    short = {"p_value": "p", "effect_size": "eff", "statistic": "stat"}
    parts = [
        f"{short[k]}={float(stat[k]):.3g}"
        for k in ("p_value", "effect_size", "statistic")
        if isinstance(stat.get(k), (int, float))
    ]
    if not parts:
        parts = [
            f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in list(stat.items())[:2]
        ]
    return " ".join(parts)


def decision_points(
    run_dir: Path, report: dict, repository: dict, min_candidates: int, drop_unanswerable: bool
) -> tuple[dict[str, dict], list[list[str]], list[str]]:
    """`({node id: decision}, trajectories, notes)` for one run.

    A decision is everything needed to render the screen and score the pick, resolved once
    here so the session file is self-contained and `--report` never re-reads a run.
    """
    records, children = build_index(repository)
    pruned_by_parent: dict[str, list[dict]] = defaultdict(list)
    for q in report.get("pruned_questions", []):
        pruned_by_parent[q["parent_id"]].append(q)

    thresholds = level_thresholds(report, records)
    verdicts = termination_verdicts(report, records, children, continuations(report))
    traced = breakdowns_from_trace(run_dir)
    weights = lambdas(run_dir)
    goal = report.get("goal") or repository.get("goal") or ""
    columns = [f"{c} ({role})" for c, role in (report.get("schema", {}).get("kept") or {}).items()]

    notes: list[str] = []
    out: dict[str, dict] = {}
    for rid, rec in records.items():
        verdict = verdicts.get(rid)
        stop = verdict["terminate"] if verdict else None
        reason = verdict["reason"] if verdict else ""
        offered = len(children.get(rid, [])) + len(pruned_by_parent.get(rid, []))
        if not offered and stop is None:                 # a leaf nobody judged: nothing to ask
            continue
        pool, dropped = candidate_pool(
            rid, pruned_by_parent, records, children, traced,
            drop_unanswerable=drop_unanswerable,
        )
        if dropped:
            notes.append(f"node {rid[:8]}: dropped {dropped} candidate(s) with no usable score")
        if len(pool) < min_candidates:
            # An early-stopped node has no candidates by construction — the search halted
            # before proposing any. It still carries a termination verdict, so it is kept
            # and only that question is put to the judge.
            if stop is None:
                notes.append(f"node {rid[:8]}: skipped — only {len(pool)} candidate(s)")
                continue
            pool = []

        depth = int(rec.get("depth", 0))
        path = [
            {
                "depth": int(s.get("depth", 0)),
                "action": s.get("action") or "base",
                "question": s.get("question") or "",
                "label": s.get("label") or "",
                "statistic": fmt_stat((s.get("evidence") or {}).get("statistic")),
            }
            for s in path_to(rid, records)
        ]
        out[rid] = {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "node_id": rid,
            "goal": goal,
            "columns": columns,
            "depth": depth,
            "child_depth": depth + 1,
            "trajectory": [s["id"] for s in path_to(rid, records)],
            "trajectory_id": rec.get("trajectory_id"),
            "path": path,
            "candidates": pool,
            "n_selected": sum(1 for c in pool if c["status"] in SELECTED),
            "threshold": thresholds.get(depth + 1),
            "lambdas": weights,
            # None when no verdict is recoverable — the node is then candidates-only.
            "d2i_terminate": stop,
            "terminate_reason": reason,
            # How good the verdict above is: "judged" carries the judge's own utility,
            # "terminated" scrapes it from prose, "inferred" has none. Only "judged" rows
            # may feed the margin and ROC metrics — see `stop_row`.
            "terminate_source": verdict["source"] if verdict else None,
            "continuation": {
                "utility": verdict["utility"], "threshold": verdict["threshold"],
                "criteria": verdict["criteria"],
            } if verdict and verdict["utility"] is not None else None,
            "stop_margin": (verdict["utility"] - verdict["threshold"])
            if verdict and verdict["utility"] is not None
            and verdict["threshold"] is not None else None,
        }
    return out, trajectories(records, children), notes


# ----------------------------------------------------------------- sampling and blinding

def blind(decision: dict, rng: random.Random) -> dict:
    """Shuffle the presentation order. Slot `k` (1-based) shows `candidates[order[k-1]]`;
    `candidates` itself stays in D2I's ranking. Storing the order means the exact screen
    the judge saw can be reproduced from session.json alone."""
    order = list(range(len(decision["candidates"])))
    rng.shuffle(order)
    decision["order"] = order
    decision["presented_to_true"] = {str(k + 1): i for k, i in enumerate(order)}
    return decision


def build_decisions(
    paths: list[Path], n: int, rng: random.Random, min_candidates: int, drop_unanswerable: bool
) -> tuple[list[dict], list[str]]:
    """`(decisions, notes)` — n decision points, at most one per trajectory.

    Trajectories share prefixes (one node in carsales-easy sits on 4 of the 9 paths), so a
    global `used` set keeps the sample distinct: without it the same pool would be judged
    several times and would silently carry that much extra weight in the rates.
    """
    notes: list[str] = []
    by_run: dict[str, dict[str, dict]] = {}
    pool: list[tuple[str, list[str]]] = []

    for run_dir in paths:
        loaded = load_run(run_dir)
        if isinstance(loaded, str):
            notes.append(f"{run_dir.name}: skipped — {loaded}")
            continue
        report, repository = loaded
        points, trajs, run_notes = decision_points(
            run_dir, report, repository, min_candidates, drop_unanswerable
        )
        notes += [f"{run_dir.name}: {t}" for t in run_notes]
        if not points:
            notes.append(f"{run_dir.name}: skipped — no node offers {min_candidates}+ candidates")
            continue
        by_run[str(run_dir)] = points
        pool += [(str(run_dir), t) for t in trajs]

    rng.shuffle(pool)
    # Mix in the early-stopped lines: a run holds one or two of them against a dozen
    # ordinary nodes, so a uniform draw would almost never show one and the termination
    # question would only ever be asked where the answer is "continue". Their trajectories
    # go first, capped at half the sample so the ordinary nodes are not crowded out.
    stopped_first = sorted(
        pool,
        key=lambda it: not any(
            by_run[it[0]].get(rid, {}).get("d2i_terminate") is True for rid in it[1]
        ),
    )
    cap = max(1, (n + 1) // 2)

    decisions: list[dict] = []
    used: set[tuple[str, str]] = set()
    n_stopped = 0
    for run_key, traj in stopped_first:
        if len(decisions) >= n:
            break
        points = by_run[run_key]
        # The random depth: any node of this trajectory that is a decision point and has
        # not already been drawn via a sibling trajectory sharing the prefix.
        avail = [rid for rid in traj if rid in points and (run_key, rid) not in used]
        if not avail:
            continue
        stopped = [rid for rid in avail if points[rid]["d2i_terminate"] is True]
        if stopped and n_stopped < cap:
            rid = stopped[0]
            n_stopped += 1
        else:
            rid = rng.choice([r for r in avail if r not in stopped] or avail)
        used.add((run_key, rid))
        decisions.append(blind(json.loads(json.dumps(points[rid])), rng))
    if n_stopped:
        notes.append(f"mixed in {n_stopped} early-stopped node(s) — the termination question "
                     "would otherwise almost always have 'continue' as its answer")

    seen_ids: dict[str, int] = {}
    for d in decisions:
        base = f"{d['run']}#{d['node_id'][:8]}"
        seen_ids[base] = seen_ids.get(base, 0) + 1
        d["id"] = base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}"

    if len(decisions) < n:
        notes.append(
            f"sampled {len(decisions)} of the {n} requested — "
            f"{len(pool)} trajectory(ies) across {len(by_run)} run(s), "
            "distinct decision points exhausted"
        )
    return decisions, notes


# ----------------------------------------------------------------- the terminal screen

def _term_width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 160))


def _wrap(text: str, width: int, indent: str, hang: str | None = None) -> list[str]:
    return textwrap.wrap(
        text, width, initial_indent=indent, subsequent_indent=hang if hang is not None else indent
    ) or [indent.rstrip()]


def show_trajectory(decision: dict, index: int, total: int) -> None:
    """Print the trajectory as it stood at this node — the first half of the screen.

    Shown on its own, before the candidates, because the termination question is asked
    first and must be answered on the line's own merits: the model's judge decides whether
    a trajectory is spent from the trajectory and its open gaps, not from a menu of
    follow-ups. Seeing an appealing candidate first would make "continue" the obvious
    answer every time.

    Deliberately never reads `score`, `status`, `n_selected`, `threshold` or
    `d2i_terminate` — nor the run's `global_summary`/`insights`, which narrate the finished
    search and would name the very branches the judge is being asked to rule on.
    """
    width = _term_width()
    print("\n" + "=" * width)
    print(f"decision {index + 1}/{total}   run {decision['run']}   "
          f"node {decision['node_id'][:8]}   depth {decision['depth']} -> {decision['child_depth']}")
    print("-" * width)
    if decision.get("goal"):
        for line in _wrap(f"goal: {decision['goal']}", width, ""):
            print(line)
    if decision.get("columns"):
        for line in _wrap("columns: " + ", ".join(decision["columns"]), width, ""):
            print(line)

    print("\ntrajectory so far")
    for i, step in enumerate(decision["path"]):
        print(f"  [{i}] depth {step['depth']}  action {step['action']}")
        if step["question"]:
            for line in _wrap(f"question: {step['question']}", width, " " * 6, " " * 16):
                print(line)
        if step["label"]:
            for line in _wrap(f"insight : {step['label']}", width, " " * 6, " " * 16):
                print(line)
        print(f"      evidence: {step['statistic']}")

    print("-" * width)


def show_candidates(decision: dict) -> None:
    """The second half of the screen: the shuffled candidate questions."""
    width = _term_width()
    n = len(decision["candidates"])
    print(f"\nwhich question should be asked next?   ({n} candidates)")
    for slot, true_i in enumerate(decision["order"], 1):
        c = decision["candidates"][true_i]
        print(f"  {slot}. [{c['action']}]")
        for line in _wrap(c["question"], width, " " * 5):
            print(line)
    print("-" * width)


STOP_PROMPT = ("is this trajectory worth continuing?  [c] continue  [t] terminate  "
               "[s] skip  [u] undo  [q] save & quit\n> ")

STOP_HELP = """
  c       keep exploring: this line still has somewhere useful to go
  t       terminate: the line is spent — further questions would add little
  s       skip this decision entirely — not scored, not shown again
  u       undo the previous decision and judge it again
  q       save and quit; resume later with --resume <session.json>
  Add a note after the key, e.g. "t the goal is already answered".
"""

HELP = """
  1..n    this question is the best one to ask next, given the trajectory so far
  s       skip this decision — not scored, not shown again
  u       go back to the continue/terminate question for this decision
  q       save and quit; resume later with --resume <session.json>
  Add a note after the key, e.g. "3 the only one that touches the goal's second half".
"""


def _read(prompt: str, allowed: set[str], help_text: str) -> tuple[str, str] | None:
    """One keypress as (key, note); None on EOF (treated as quit)."""
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            print()
            return None
        if not raw:
            continue
        key, _, note = raw.partition(" ")
        key = key.lower()
        if key in allowed:
            return key, note.strip()
        print(help_text if key in {"?", "h", "help"}
              else f"unknown option {raw!r} — press ? for help")


def ask_terminate(decision: dict, index: int, total: int) -> tuple[str, str] | None:
    """Stage one: stop here, or keep going? Asked before the candidates are shown."""
    show_trajectory(decision, index, total)
    return _read(STOP_PROMPT, {"c", "t", "s", "u", "q"}, STOP_HELP)


def ask_candidate(decision: dict) -> tuple[str, str] | None:
    """Stage two: which question next? Only reached when the judge chose to continue."""
    show_candidates(decision)
    n = len(decision["candidates"])
    prompt = (f"which question should be asked next?  [1-{n}] pick  [s] skip  "
              "[u] back  [q] save & quit\n> ")
    return _read(prompt, {str(i) for i in range(1, n + 1)} | {"s", "u", "q"}, HELP)


def reveal(decision: dict, verdict: dict) -> None:
    """What the model did at this node, printed once the whole decision has been answered.

    Both halves land together, after the candidate pick rather than between the two
    prompts: knowing the model continued here implies it answered one of the candidates,
    which would tilt the pick that follows. `--no-feedback` suppresses all of it, since a
    judge who learns the ranking mid-session starts predicting the scorer and the later
    decisions stop being independent samples.
    """
    stop = decision.get("d2i_terminate")
    if stop is not None:
        agree = verdict.get("terminate") is stop
        did = "terminated the line here" if stop else "kept exploring this line"
        print(f"  the model {did}." + ("  ✓ agreement" if agree else "  ✗ you disagreed"))
        if stop and decision.get("terminate_reason"):
            for line in _wrap(f"judge: {decision['terminate_reason']}", _term_width(),
                              " " * 4, " " * 11):
                print(line)

    true_index = verdict.get("true_index")
    if true_index is None:
        return
    cands = decision["candidates"]
    pick = cands[true_index]
    rank = 1 + sum(1 for c in cands if c["score"] > pick["score"])
    top = cands[0]

    line = f"  the model ranked it {rank} of {len(cands)} (score {pick['score']:.4f})"
    print(f"{line} — its top pick. ✓ agreement" if rank == 1
          else f"{line} — {top['score'] - pick['score']:.4f} below its top pick.")

    # `slot` is the position each candidate held on the screen just read, which is how a
    # row here maps back to its action and question — those are not repeated.
    slot_of = {t: k for k, t in enumerate(decision["order"], 1)}
    print("\n  the model's ranking   (✓ = the model answered it, ← your pick)")
    print(f"    {'#':>2}  {'slot':>4}  {'':2}  {'score':>7}   "
          + "".join(f"{short:>7}" for _, short in TERMS))
    for i, c in enumerate(cands):
        r = 1 + sum(1 for o in cands if o["score"] > c["score"])
        # Two independent flags, never one overwriting the other: the interesting row is a
        # pick that the model also answered.
        mark = ("✓" if c["status"] in SELECTED else " ") + ("←" if i == true_index else " ")
        bd = c.get("breakdown") or {}
        terms = "".join(f"{bd[key]:>7.2f}" if _numeric(bd.get(key)) else f"{'—':>7}"
                        for key, _ in TERMS)
        print(f"    {r:>2}  {slot_of[i]:>4}  {mark}  {c['score']:>7.4f}   {terms}")

    w = decision.get("lambdas")
    if w:
        print("    score = " + " + ".join(f"{w[s]:g}·{s}" for _, s in TERMS))


# ----------------------------------------------------------------- session i/o

def save_session(session: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- metrics

# Bumped whenever a metric's definition changes. `survey.json` carries it and the browser
# echoes it back in the response, so a page served against a stale bundle is detectable
# rather than silently reporting numbers computed two different ways.
METRICS_VERSION = "2"

# The metric surface, in report order. `key` is the per-row field aggregated; `base` the
# per-row chance column (exact, never a closed form — see `pick_row`); `est` how the column
# is reduced, which is what makes κ and the weighted mean fit the same table as a plain
# rate; `only` restricts the rows a metric may use.
METRIC_TABLE = [
    {"key": "is_top", "base": "e_top", "label": "κ_sel (chance-corrected agreement)",
     "fmt": "num", "est": "kappa", "group": "selection", "only": None},
    {"key": "is_top", "base": "e_top", "label": "  ├ raw agreement rate",
     "fmt": "pct", "est": "mean", "group": "selection", "only": None},
    {"key": "attainment", "base": "e_attainment", "label": "utility attainment [0,1]",
     "fmt": "num", "est": "mean", "group": "selection", "only": None},
    {"key": "d_std", "base": "e_d_std", "label": "standardized regret (pool SDs)",
     "fmt": "num", "est": "mean", "group": "selection", "only": "d_std"},
    {"key": "within_noise", "base": "e_within_noise", "label": "  └ within-noise rate (d<0.5)",
     "fmt": "pct", "est": "mean", "group": "selection", "only": "d_std"},
    {"key": "action_hit", "base": "e_action_hit", "label": "action agreement",
     "fmt": "pct", "est": "mean", "group": "selection", "only": None},
    {"key": "agree", "base": None, "label": "margin-weighted agreement",
     "fmt": "pct", "est": "wmean", "group": "continuation", "only": "margin"},
    {"key": "agree", "base": None, "label": "  ├ unweighted (w≡1)",
     "fmt": "pct", "est": "mean", "group": "continuation", "only": "margin"},
    {"key": "utility", "base": None, "label": "AUROC (utility vs human label)",
     "fmt": "num", "est": "auroc", "group": "continuation", "only": "margin"},
    {"key": "utility", "base": None, "label": "implied human threshold τ̂",
     "fmt": "num", "est": "youden", "group": "continuation", "only": "margin"},
]

# A pick is "within noise" of the top when it falls under half a pool standard deviation of
# it — Cohen's conventional small-effect line, used here to separate a real disagreement
# from one D2I's own scores cannot resolve.
WITHIN_NOISE_D = 0.5


def _pool_stats(scores: list[float]) -> dict:
    """Everything about a candidate pool that does not depend on which one was picked."""
    n = len(scores)
    top, lo = max(scores), min(scores)
    mean = math.fsum(scores) / n
    var = math.fsum((s - mean) ** 2 for s in scores) / n          # population SD: the pool
    sd = math.sqrt(var)                                           # is the whole population
    ordered = sorted(scores, reverse=True)
    return {
        "n": n, "top": top, "lo": lo, "mean": mean, "sd": sd,
        "span": top - lo,
        "top2_gap": (ordered[0] - ordered[1]) if n > 1 else 0.0,
        "n_top": sum(1 for s in scores if s == top),
    }


def _one_pick(pool: dict, score: float, action: str, top_action: str) -> dict:
    """The three selection metrics for a single candidate. Shared by the real pick and by
    each of the `n` hypothetical picks the exact chance baseline averages over."""
    return {
        # Score equality, not slot identity: an exact tie with the top is agreement, not a
        # distinction D2I ever drew.
        "is_top": 1.0 if score == pool["top"] else 0.0,
        # A flat pool has nothing to attain and nothing to regret; it is scored as full
        # agreement and counted separately so it can be excluded on inspection.
        "attainment": ((score - pool["lo"]) / pool["span"]) if pool["span"] > 0 else 1.0,
        "d_std": ((pool["top"] - score) / pool["sd"]) if pool["sd"] > 0 else None,
        "within_noise": (1.0 if (pool["top"] - score) / pool["sd"] < WITHIN_NOISE_D else 0.0)
        if pool["sd"] > 0 else None,
        "action_hit": 1.0 if action == top_action else 0.0,
    }


def pick_row(decision: dict, true_index: int) -> dict:
    """One scored row for "the human picked `candidates[true_index]` at this decision".

    The single definition of the selection metrics. `build_survey.py` calls it once per slot
    at build time and publishes the result in `truth.json`, so the browser only ever
    averages numbers computed here — there is no second implementation to drift.

    Every chance baseline is the exact mean of the metric over all `n` candidates, i.e.
    literally "what a uniform pick from this pool would have scored". No closed form, so
    ties, flat pools and odd pool sizes are all handled by construction.
    """
    cands = decision["candidates"]
    scores = [float(c["score"]) for c in cands]
    pool = _pool_stats(scores)
    top_action = cands[0]["action"]
    pick = cands[true_index]

    row = _one_pick(pool, float(pick["score"]), pick["action"], top_action)
    each = [_one_pick(pool, s, c["action"], top_action) for s, c in zip(scores, cands)]

    def chance(key: str) -> float | None:
        vals = [e[key] for e in each if e[key] is not None]
        return math.fsum(vals) / len(vals) if vals else None

    row.update({
        "id": decision["id"],
        "run": decision["run"],
        "depth": decision["child_depth"],
        # Two survey items can hang off one trajectory, so this is the cluster the
        # bootstrap resamples — not the decision.
        "cluster": decision.get("trajectory_id") or decision["id"],
        "n": pool["n"],
        "pick_action": pick["action"],
        "top_action": top_action,
        "pick_status": pick["status"],
        "flat_pool": 1.0 if pool["span"] <= 0 else 0.0,
        "pool_sd": pool["sd"],
        "top2_gap": pool["top2_gap"],
        "e_top": chance("is_top"),
        "e_attainment": chance("attainment"),
        "e_d_std": chance("d_std"),
        "e_within_noise": chance("within_noise"),
        "e_action_hit": chance("action_hit"),
        # Component attribution: the same two headline estimators recomputed against each
        # single term's ranking, so the table stays one-dimensional.
        "terms": _term_picks(cands, true_index),
    })
    return row


def _term_picks(cands: list[dict], true_index: int) -> dict[str, dict]:
    """`{term: {is_top, attainment, e_top, e_attainment}}` — the pool re-ranked by one
    score component at a time. A term with no usable breakdown is left out entirely rather
    than defaulted, so it shows as "not estimable" instead of as agreement."""
    out: dict[str, dict] = {}
    for key, short in TERMS:
        vals = [(c.get("breakdown") or {}).get(key) for c in cands]
        if not all(_numeric(v) for v in vals):
            continue
        vals = [float(v) for v in vals]
        pool = _pool_stats(vals)
        each = [_one_pick(pool, v, "", "") for v in vals]
        mine = _one_pick(pool, vals[true_index], "", "")
        out[short] = {
            "is_top": mine["is_top"],
            "attainment": mine["attainment"],
            "e_top": math.fsum(e["is_top"] for e in each) / len(each),
            "e_attainment": math.fsum(e["attainment"] for e in each) / len(each),
        }
    return out


def stop_row(decision: dict, human: bool) -> dict:
    """One scored row for the human's stop/continue call at this decision.

    `source` decides what the row may be used for, and the line that matters is whether the
    judge actually ruled — not which artifact recorded it. `"judged"` and `"terminated"`
    both carry the judge's own utility (structured, and scraped from its printed arithmetic
    respectively), so both feed the margin-weighted, ROC and threshold metrics. `"inferred"`
    never does: nothing was ruled there, so `margin` is `None`, which is what the
    `only: "margin"` filter keys off.

    The catch with a `"terminated"`-only sample is class imbalance rather than validity —
    every such row is a terminate by construction, so AUROC has one class and returns
    `None`. Only `"judged"` rows can supply a *continue* with a utility attached, which is
    why `n_stop_judged` is worth reporting on its own.
    """
    model = bool(decision["d2i_terminate"])
    cont = decision.get("continuation") or {}
    utility, tau = cont.get("utility"), cont.get("threshold")
    source = decision.get("terminate_source")
    usable = source in ("judged", "terminated") and utility is not None and tau is not None
    return {
        "id": decision["id"],
        "run": decision["run"],
        "depth": decision["depth"],
        "cluster": decision.get("trajectory_id") or decision["id"],
        "source": source,
        "model": model,
        "human": bool(human),
        "agree": 1.0 if model == bool(human) else 0.0,
        # `human_continue` is the ROC label; utility is the score that should order it.
        "human_continue": 0.0 if human else 1.0,
        "utility": float(utility) if usable else None,
        "threshold": float(tau) if usable else None,
        "margin": (float(utility) - float(tau)) if usable else None,
        "criteria": cont.get("criteria") if usable else None,
    }


def scored_picks(session: dict) -> list[dict]:
    """One row per judged (non-skipped) decision.

    Recomputed from the pick list every time, like human_eval's elo/tally — so `u` needs no
    rollback and `--report` works on a half-finished session.
    """
    by_id = {d["id"]: d for d in session.get("decisions", [])}
    rows: list[dict] = []
    for v in session.get("verdicts", []):
        d = by_id.get(v["decision"])
        if v.get("true_index") is None or d is None or not d.get("candidates"):
            continue
        rows.append(pick_row(d, v["true_index"]))
    return rows


def termination_rows(session: dict) -> list[dict]:
    """One row per decision where both the model and the human ruled on stopping."""
    by_id = {d["id"]: d for d in session.get("decisions", [])}
    rows = []
    for v in session.get("verdicts", []):
        d = by_id.get(v["decision"])
        if d is None or v.get("terminate") is None or d.get("d2i_terminate") is None:
            continue
        rows.append(stop_row(d, bool(v["terminate"])))
    return rows


def _mean(xs: list[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


# ----------------------------------------------------------------- estimators
#
# Deliberately tiny and dependency-free: `survey/metrics.js` reimplements exactly these
# four, and `survey/tests/` pins the two against each other. Anything more elaborate lives
# in `score_survey.py`, which is free to use scipy.

def kappa(obs: list[float], exp: list[float]) -> float | None:
    """Chance-corrected agreement with a *heterogeneous* per-item baseline.

    `(ā − ē)/(1 − ē)` — Cohen's correction, but the expected rate varies by item because
    the pools differ in size and in how many candidates share the top score. 0 is chance,
    1 is perfect, negative is worse than guessing. `None` when the baseline is already 1
    (every candidate tied at the top), where the correction is undefined rather than 0.
    """
    a, e = _mean(obs), _mean(exp)
    if a is None or e is None or e >= 1.0:
        return None
    return (a - e) / (1.0 - e)


def wmean(vals: list[float], weights: list[float]) -> float | None:
    """Weight-normalised mean; falls back to the plain mean when every weight is 0 (all
    decisions sat exactly on the threshold), which is the sensible limit rather than 0/0."""
    if not vals:
        return None
    total = math.fsum(weights)
    if total <= 0:
        return _mean(vals)
    return math.fsum(v * w for v, w in zip(vals, weights)) / total


def auroc(scores: list[float], labels: list[float]) -> float | None:
    """Tie-corrected AUC — the Mann–Whitney statistic, half credit for ties.

    Utilities are heavily quantised (mass sits on 0.30/0.54/0.66), so the tie term is not a
    rounding detail here: an untied AUC would silently reward or punish the model for
    orderings its own judge never expressed. `None` unless both classes are present.
    """
    pos = [s for s, y in zip(scores, labels) if y > 0]
    neg = [s for s, y in zip(scores, labels) if y <= 0]
    if not pos or not neg:
        return None
    wins = math.fsum(
        1.0 if p > q else 0.5 if p == q else 0.0
        for p in pos for q in neg
    )
    return wins / (len(pos) * len(neg))


def youden(scores: list[float], labels: list[float]) -> tuple[float, float, float] | None:
    """`(τ̂, lo, hi)` — the cut on `scores` maximising balanced agreement, and the interval
    of cuts that tie for it.

    "Continue" is predicted when `score >= cut`. Because the utilities are quantised, a
    single argmax would be an artefact of which candidate cut happened to be tried first;
    the whole optimal interval is returned instead, and `τ̂` is its midpoint.
    """
    pos = [s for s, y in zip(scores, labels) if y > 0]
    neg = [s for s, y in zip(scores, labels) if y <= 0]
    if not pos or not neg:
        return None
    lo, hi = min(scores), max(scores)
    cuts = sorted({lo - 1e-9, hi + 1e-9, *scores})
    best, keep = -2.0, []
    for cut in cuts:
        j = (sum(1 for s in pos if s >= cut) / len(pos)
             - sum(1 for s in neg if s >= cut) / len(neg))
        if j > best + 1e-12:
            best, keep = j, [cut]
        elif abs(j - best) <= 1e-12:
            keep.append(cut)
    return (math.fsum(keep) / len(keep), min(keep), max(keep))


def wilson(k: float, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Wilson score interval for a proportion — the port of `metrics.js:wilson`.

    Preferred to the normal approximation for the plain rates: at the counts one respondent
    produces, Wald intervals routinely run past 0 or 1.
    """
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mulberry32(seed: int):
    """A 32-bit PRNG, reimplemented byte-for-byte in `survey/metrics.js`.

    The browser and this module must produce the *same* confidence interval for the same
    answers, or a respondent's downloaded table would disagree with the analysis run over
    it. Python's `random` and JS's `Math.random` cannot both be pinned, so neither is used:
    this is short enough to hold identical in two languages and is seeded per report.
    """
    state = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return nxt


# Percentile, not BCa. At the cluster counts this instrument produces (often under 20) the
# bias-correction and acceleration terms are themselves estimated from too little data to
# help, and BCa needs an inverse normal CDF that would have to be duplicated in JS —
# two ways for the numbers to drift for no gain in coverage.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260728


def cluster_bootstrap(rows: list[dict], stat, seed: int = BOOTSTRAP_SEED,
                      draws: int = BOOTSTRAP_DRAWS, alpha: float = 0.05
                      ) -> tuple[float | None, float | None]:
    """A 95% percentile CI for `stat(rows)`, resampling **clusters** rather than rows.

    Two survey items can descend from one trajectory — one judge call, two questions — so
    they are not independent observations. Resampling rows directly would treat them as if
    they were and report an interval that is too narrow. Clusters are drawn with
    replacement and their rows taken whole.
    """
    if not rows:
        return None, None
    groups: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("cluster", r.get("id"))].append(r)
    keys = sorted(groups, key=str)
    if len(keys) < 2:
        return None, None
    rnd = mulberry32(seed)
    got: list[float] = []
    for _ in range(draws):
        sample: list[dict] = []
        for _ in range(len(keys)):
            sample.extend(groups[keys[int(rnd() * len(keys)) % len(keys)]])
        value = stat(sample)
        if value is not None:
            got.append(value)
    if len(got) < draws // 10:            # too many resamples were not estimable to trust
        return None, None
    got.sort()

    def pick(q: float) -> float:
        return got[min(len(got) - 1, max(0, int(q * (len(got) - 1) + 0.5)))]

    return pick(alpha / 2), pick(1 - alpha / 2)


def estimate(spec: dict, rows: list[dict]) -> tuple[float | None, float | None, int]:
    """`(value, chance, n)` for one `METRIC_TABLE` entry over `rows`."""
    only = spec.get("only")
    sel = [r for r in rows if (only is None or r.get(only) is not None)
           and r.get(spec["key"]) is not None]
    if not sel:
        return None, None, 0
    vals = [float(r[spec["key"]]) for r in sel]
    base = ([float(r[spec["base"]]) for r in sel]
            if spec.get("base") and all(r.get(spec["base"]) is not None for r in sel)
            else None)
    est = spec.get("est", "mean")
    if est == "mean":
        return _mean(vals), (_mean(base) if base else None), len(sel)
    if est == "kappa":
        return kappa(vals, base or []), 0.0, len(sel)
    if est == "wmean":
        return wmean(vals, [abs(float(r["margin"])) for r in sel]), None, len(sel)
    if est == "auroc":
        return auroc(vals, [float(r["human_continue"]) for r in sel]), 0.5, len(sel)
    if est == "youden":
        # Not identifiable when every utility sits on one side of D2I's own threshold:
        # there is no evidence about where the human would have cut on the other side, so
        # any argmax is an artefact of the range that happened to be sampled.
        taus = [r["threshold"] for r in sel if r.get("threshold") is not None]
        if taus and (max(vals) < min(taus) or min(vals) >= max(taus)):
            return None, None, len(sel)
        got = youden(vals, [float(r["human_continue"]) for r in sel])
        return (got[0] if got else None), None, len(sel)
    raise ValueError(f"unknown estimator {est!r}")


def aggregate(picks: list[dict], stops: list[dict] | None = None,
              seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """The full metric table as data: `{key, label, group, fmt, value, chance, lo, hi, n}`.

    This is what the page renders, what the download carries and what `--check-parity`
    compares, so it is deliberately free of formatting.
    """
    out = []
    for spec in METRIC_TABLE:
        rows = picks if spec["group"] == "selection" else (stops or [])
        value, chance_, n = estimate(spec, rows)
        lo = hi = None
        if value is not None:
            lo, hi = cluster_bootstrap(rows, lambda rs: estimate(spec, rs)[0], seed=seed)
        out.append({
            "key": spec["key"], "label": spec["label"], "group": spec["group"],
            "fmt": spec["fmt"], "est": spec.get("est", "mean"),
            "value": value, "chance": chance_, "lo": lo, "hi": hi, "n": n,
        })
    return out


def summarise(picks: list[dict], stops: list[dict] | None = None
              ) -> list[tuple[str, str, str, int]]:
    """`(label, human, chance, n)` for every metric, formatted for the table."""
    out: list[tuple[str, str, str, int]] = []
    for spec in METRIC_TABLE:
        rows = picks if spec["group"] == "selection" else (stops or [])
        value, chance_, n = estimate(spec, rows)

        def cell(x: float | None) -> str:
            if x is None:
                return f"{'—':>9}"
            return f"{100 * x:>8.1f}%" if spec["fmt"] == "pct" else f"{x:>9.3f}"

        out.append((spec["label"], cell(value), cell(chance_), n))
    return out


def by_key(rows: list[dict], key: str) -> dict[object, dict[str, float | int]]:
    """Per-depth / per-action / per-run breakdown: n, agreement rate, mean attainment."""
    groups: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {
        k: {"n": len(v), "is_top": _mean([r["is_top"] for r in v]) or 0.0,
            "attainment": _mean([r["attainment"] for r in v]) or 0.0}
        for k, v in groups.items()
    }


def confusion(rows: list[dict], row_key: str, col_key: str,
              labels: list[str] | None = None) -> tuple[list[str], list[list[int]]]:
    """`(labels, matrix)` with `matrix[i][j]` = rows whose `row_key` is `labels[i]` and
    `col_key` is `labels[j]`. Used for both the 7×7 action matrix and the 2×2 stop one."""
    if labels is None:
        labels = sorted({str(r[row_key]) for r in rows} | {str(r[col_key]) for r in rows})
    index = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for r in rows:
        i, j = index.get(str(r[row_key])), index.get(str(r[col_key]))
        if i is not None and j is not None:
            m[i][j] += 1
    return labels, m


def term_attribution(picks: list[dict]) -> list[dict]:
    """Component attribution: κ_sel and attainment recomputed per score term.

    Only picks whose pool had a usable breakdown for that term contribute, so `n` varies by
    row and is reported."""
    out = []
    for _, short in TERMS:
        sel = [p["terms"][short] for p in picks if short in (p.get("terms") or {})]
        if not sel:
            out.append({"term": short, "n": 0, "kappa": None, "attainment": None})
            continue
        out.append({
            "term": short,
            "n": len(sel),
            "kappa": kappa([s["is_top"] for s in sel], [s["e_top"] for s in sel]),
            "attainment": _mean([s["attainment"] for s in sel]),
            "e_attainment": _mean([s["e_attainment"] for s in sel]),
        })
    return out


def criterion_attribution(stops: list[dict]) -> list[dict]:
    """Continuation attribution: AUROC of each judge criterion against the human label,
    for comparison with the AUROC of the aggregate utility."""
    sel = [s for s in stops if s.get("criteria")]
    names: list[str] = []
    for s in sel:
        for k in s["criteria"]:
            if k not in names:
                names.append(k)
    labels = [float(s["human_continue"]) for s in sel]
    out = []
    for name in names:
        vals = [s["criteria"].get(name) for s in sel]
        if not all(_numeric(v) for v in vals):
            out.append({"criterion": name, "n": 0, "auroc": None})
            continue
        out.append({"criterion": name, "n": len(sel),
                    "auroc": auroc([float(v) for v in vals], labels)})
    return out


def report(session: dict) -> list[str]:
    """The result tables for a session, as lines (printed and written to --out)."""
    decisions = session.get("decisions", [])
    verdicts = session.get("verdicts", [])
    rows = scored_picks(session)
    skipped = sum(1 for v in verdicts if v.get("true_index") is None)
    runs = sorted({d["run"] for d in decisions})

    stops = termination_rows(session)
    n_stopped = sum(1 for d in decisions if d.get("d2i_terminate") is True)

    lines = [
        "D2I vs human — search-decision agreement",
        "",
        f"  session : {session['stamp']}"
        + (f"  (judge: {session['judge']})" if session.get("judge") else ""),
        f"  path    : {session['path']}",
        f"  runs    : {len(runs)}  ({', '.join(runs) if runs else '—'})",
        f"  sampled : {len(decisions)} decision point(s) of the {session['n_requested']} "
        f"requested ({n_stopped} early-stopped), seed {session['seed']}",
        f"  judged  : {len(rows)} pick(s), {len(stops)} stop/continue call(s), of "
        f"{len(decisions)} decision(s) "
        f"({skipped} without a pick, {len(decisions) - len(verdicts)} unjudged)",
    ]

    # -- the metric table --------------------------------------------------
    # "ruled" = the judge actually decided here and left a utility behind; "inferred" = the
    # search merely carried on. Only the first may carry the continuous metrics.
    judged_stops = [r for r in stops if r["margin"] is not None]
    inferred = [r for r in stops if r["margin"] is None]
    table = summarise(rows, stops)
    lines += ["", "alignment", "",
              f"{'metric':>34}  {'human':>9}  {'chance':>9}  {'n':>4}", "-" * 62]
    group = None
    for spec, (label, human, chance_, n) in zip(METRIC_TABLE, table):
        if spec["group"] != group:
            group = spec["group"]
            lines.append(f"  {'-- selection --' if group == 'selection' else '-- continuation --'}")
        lines.append(f"{label:>34}  {human}  {chance_}  {n:>4}")
    lines += [
        "-" * 62,
        "(κ_sel is chance-corrected: 0 is a uniform pick from the same pool, 1 is D2I's own",
        " top choice every time. attainment is the share of the pool's score range captured;",
        " standardized regret is the shortfall in pool SDs, so <0.5 is inside D2I's own noise.",
        " margin-weighted agreement discounts stop calls the judge itself was unsure about.)",
    ]

    # -- stop/continue detail ----------------------------------------------
    if stops:
        lines += ["", "terminate or continue", ""]
        if inferred:
            lines.append(f"  {len(judged_stops)} of {len(stops)} stop call(s) carry the judge's own"
                         " utility; the rest are inferred from the search having continued,")
            lines.append("  and are reported apart because no verdict was actually made there.")
        with_crit = sum(1 for r in judged_stops if r.get("criteria"))
        utils = [r["utility"] for r in judged_stops if r["utility"] is not None]
        taus = [r["threshold"] for r in judged_stops if r["threshold"] is not None]
        if utils and taus and (max(utils) < min(taus) or min(utils) >= max(taus)):
            lines.append("  every utility falls on one side of the threshold, so τ̂ is not"
                         " identifiable against it and AUROC is a restricted-range test:")
            lines.append("  only a run whose report.json carries a structured `continuation`"
                         " records a utility for the trajectories it let continue.")
        if with_crit < len(judged_stops):
            lines.append(f"  {with_crit} of {len(judged_stops)} carry the judge's criteria"
                         " breakdown (structured `continuation` only).")
        for name, sub in (("ruled", judged_stops), ("inferred", inferred)):
            if not sub:
                continue
            lines.append(f"  {name}: agreement "
                         f"{100 * (_mean([r['agree'] for r in sub]) or 0):.1f}% over {len(sub)}")
            for want, label in ((True, "where it terminated"), (False, "where it continued")):
                part = [r for r in sub if r["model"] is want]
                if part:
                    lines.append(f"      {label:<22}"
                                 f"{100 * (_mean([r['agree'] for r in part]) or 0):>6.1f}%"
                                 f"  n={len(part)}")
        lines += ["", f"  {'confusion (judged)':<20}{'human continue':>16}  {'human terminate':>16}"]
        for want, name in ((False, "model continue"), (True, "model terminate")):
            row = [r for r in judged_stops if r["model"] is want]
            lines.append(f"  {name:<20}{sum(1 for r in row if not r['human']):>16}  "
                         f"{sum(1 for r in row if r['human']):>16}")
        # Same identifiability guard as `estimate`: with every utility on one side of the
        # threshold there is no evidence about where the human would have cut.
        identifiable = utils and taus and min(utils) < max(taus) <= max(utils)
        cut = youden(utils, [r["human_continue"] for r in judged_stops
                             if r["utility"] is not None]) if identifiable else None
        if cut:
            tau = min(taus)
            lines.append(f"  implied human threshold {cut[0]:.3f} (optimal cuts {cut[1]:.2f}"
                         f"–{cut[2]:.2f}) against D2I's {tau:.2f} — humans would prune "
                         f"{'less' if cut[0] < tau else 'more'} aggressively.")

    # -- attribution --------------------------------------------------------
    if rows:
        lines += ["", "component attribution (which score term the human tracks)", "",
                  f"{'term':>14}  {'κ_sel':>8}  {'attain':>8}  {'chance':>8}  {'n':>4}", "-" * 50]
        for a in term_attribution(rows):
            k = f"{a['kappa']:>8.3f}" if a["kappa"] is not None else f"{'—':>8}"
            at = f"{a['attainment']:>8.3f}" if a["attainment"] is not None else f"{'—':>8}"
            ch = f"{a.get('e_attainment'):>8.3f}" if a.get("e_attainment") is not None else f"{'—':>8}"
            lines.append(f"{a['term']:>14}  {k}  {at}  {ch}  {a['n']:>4}")
        w = (session.get("decisions") or [{}])[0].get("lambdas")
        if w:
            lines.append("  D2I weights: " + "  ".join(f"{s}={w[s]:g}" for _, s in TERMS
                                                       if s in w))

        labels, matrix = confusion(rows, "top_action", "pick_action")
        if len(labels) > 1:
            lines += ["", "action confusion (rows: D2I's top, cols: the human's pick)", "",
                      "  " + " " * 14 + "".join(f"{l[:6]:>7}" for l in labels)]
            for i, lab in enumerate(labels):
                lines.append(f"  {lab:>14}" + "".join(f"{v:>7}" for v in matrix[i]))

    crit = criterion_attribution(judged_stops)
    if any(c["auroc"] is not None for c in crit):
        lines += ["", "criterion attribution (AUROC of each judge criterion vs the human)", "",
                  f"{'criterion':>20}  {'AUROC':>8}  {'n':>4}", "-" * 38]
        for c in crit:
            a = f"{c['auroc']:>8.3f}" if c["auroc"] is not None else f"{'—':>8}"
            lines.append(f"{c['criterion']:>20}  {a}  {c['n']:>4}")

    ended = sum(1 for v in verdicts if v.get("terminate") and v.get("true_index") is None)
    if ended:
        lines.append(f"({ended} decision(s) ended at 'terminate', so no next question was picked"
                     " there — the two tables are over different subsets.)")

    flat = sum(1 for r in rows if r["flat_pool"])
    if flat:
        lines.append(f"({flat} pool(s) had every candidate on the same score — attainment is 1 and"
                     " regret undefined there, so they carry no evidence either way.)")
    if session.get("feedback"):
        lines.append("(D2I's rank was revealed after each pick, so later picks are not independent"
                     " samples — rerun with --no-feedback for a clean rate.)")

    for key, title, head in (
        ("depth", "by depth (of the candidate being chosen)", "depth"),
        ("top_action", "by action of D2I's top candidate", "action"),
        ("run", "by run", "run"),
    ):
        groups = by_key(rows, key)
        if len(groups) < 2:
            continue
        lines += ["", title, "",
                  f"{head:>14}  {'n':>4}  {'agree':>7}  {'attain':>10}", "-" * 40]
        for k in sorted(groups, key=lambda x: (isinstance(x, str), x)):
            g = groups[k]
            lines.append(f"{str(k):>14}  {g['n']:>4}  {100 * g['is_top']:>6.1f}%  "
                         f"{g['attainment']:>10.3f}")

    notes = [v for v in verdicts if v.get("note")]
    if notes:
        lines += ["", "notes", ""]
        for v in notes:
            what = (f"picked {v['slot']}" if v.get("slot") is not None
                    else "terminate" if v.get("terminate") else "skipped")
            lines.append(f"  {v['decision']:>24}  [{what}]  {v['note']}")

    if session.get("notes"):
        lines += ["", "coverage", ""] + [f"  {t}" for t in session["notes"]]
    return lines


# ----------------------------------------------------------------- the judging loop

def judge(session: dict, session_path: Path) -> bool:
    """Run the interactive loop over the unjudged decisions. True if all were judged."""
    decisions = session["decisions"]
    verdicts = session["verdicts"]
    judged = {v["decision"] for v in verdicts}

    print(f"\n{len(decisions) - len(judged)} decision(s) to judge  ({len(judged)} already done)")
    print(f"session: {session_path}")
    print("press ? at the prompt for the key list")

    i = 0
    while i < len(decisions):
        decision = decisions[i]
        if decision["id"] in judged:
            i += 1
            continue

        try:
            outcome = judge_one(decision, i, len(decisions))
        except KeyboardInterrupt:
            print("\ninterrupted — session saved")
            return False
        if outcome is None:
            return False
        kind, verdict = outcome

        if kind == "quit":
            print("saved — resume with:")
            print(f"  python eval/human_trajectory_eval.py --resume {session_path}")
            return False

        if kind == "undo":
            if not verdicts:
                print("nothing to undo")
                continue
            undone = verdicts.pop()
            judged.discard(undone["decision"])
            i = next(j for j, d in enumerate(decisions) if d["id"] == undone["decision"])
            save_session(session, session_path)
            print(f"undid {undone['decision']}")
            continue

        verdicts.append(verdict)
        judged.add(decision["id"])
        save_session(session, session_path)
        if session.get("feedback"):
            reveal(decision, verdict)
        i += 1

    return True


def judge_one(decision: dict, index: int, total: int) -> tuple[str, dict] | None:
    """One decision, both stages. `("verdict", record)` / `("undo", {})` / `("quit", {})`,
    or None on EOF.

    The two stages produce ONE verdict record, so undo stays a single pop and the pair is
    never half-saved. Terminating ends the decision: honouring the judge's call is the
    point of asking, and a next question they just said not to ask is not a real choice.
    `u` at the candidate prompt rewinds to the termination question for this same decision,
    since nothing has been written yet.
    """
    stop_asked = decision.get("d2i_terminate") is not None
    while True:
        terminate = None
        note = ""
        if stop_asked:
            answer = ask_terminate(decision, index, total)
            if answer is None:
                return None
            key, note = answer
            if key in {"q", "u"}:
                return ("quit" if key == "q" else "undo"), {}
            if key == "s":
                print("  → skipped")
                return "verdict", make_verdict(decision, None, None, key, note)
            terminate = key == "t"
            print("  → " + ("terminate" if terminate else "continue"))
            if terminate or not decision["candidates"]:
                return "verdict", make_verdict(decision, terminate, None, key, note)
        elif not decision["candidates"]:                 # nothing left to ask about
            return "verdict", make_verdict(decision, None, None, "s", "")
        else:
            show_trajectory(decision, index, total)

        answer = ask_candidate(decision)
        if answer is None:
            return None
        key, note2 = answer
        if key == "q":
            return "quit", {}
        if key == "u":
            if stop_asked:                               # back to stage one, nothing saved
                continue
            return "undo", {}
        note = note2 or note
        if key == "s":
            print("  → skipped")
            return "verdict", make_verdict(decision, terminate, None, key, note)
        slot = int(key)
        true_index = decision["order"][slot - 1]
        print(f"  → picked {slot}. [{decision['candidates'][true_index]['action']}]")
        return "verdict", make_verdict(decision, terminate, slot, key, note)


def make_verdict(decision: dict, terminate: bool | None, slot: int | None,
             key: str, note: str) -> dict:
    return {
        "decision": decision["id"],
        "terminate": terminate,                   # the human's stop/continue call, if asked
        "slot": slot,                             # the shuffled position shown
        "true_index": None if slot is None else decision["order"][slot - 1],
        "choice": key,                            # the raw key, for auditing
        "note": note,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Would a human pick the same next question as D2I, "
                    "given the trajectory so far?")
    p.add_argument("path", type=Path, nargs="?",
                   help="a run dir (one holding report.json) or any parent of run dirs, "
                        "e.g. runs/20260727-201135_d2i_bench/carsales-easy")
    p.add_argument("n", type=int, nargs="?", default=10,
                   help="trajectories to sample, one decision point each (default: 10)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the trajectory/depth sampling and the candidate "
                        "shuffle (default: a random seed, recorded in the session)")
    p.add_argument("--judge", default=None,
                   help="name of the person judging, recorded in the session")
    p.add_argument("--min-candidates", type=int, default=2,
                   help="skip decision points offering fewer candidates (default: 2)")
    p.add_argument("--drop-unanswerable", action="store_true",
                   help="leave out candidates D2I selected but failed to answer (they are "
                        "included by default: they competed and won on score)")
    p.add_argument("--no-feedback", dest="feedback", action="store_false",
                   help="do not reveal D2I's rank after each pick, keeping the picks "
                        "independent (the rank is revealed by default)")
    p.add_argument("--session-dir", type=Path, default=None,
                   help=f"where to write session.json (default: a timestamped dir under {OUT_DIR})")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue a session.json written by an earlier invocation")
    p.add_argument("--report", type=Path, default=None,
                   help="print the results for a session.json and exit (no judging)")
    p.add_argument("-o", "--out", type=Path, default=OUT_FILE,
                   help=f"also write the results to this file (default: {OUT_FILE}); "
                        "pass --out - to skip writing a file")
    args = p.parse_args()

    # -- report-only: read a finished (or partial) session and print its tables.
    if args.report:
        session = json.loads(args.report.read_text(encoding="utf-8"))
        session_path = args.report
    # -- resume: everything (decisions, order, seed) comes from the saved session.
    elif args.resume:
        session_path = args.resume
        if not session_path.is_file():
            raise SystemExit(f"no session file at {session_path}")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        judge(session, session_path)
    # -- fresh session.
    else:
        if not args.path:
            p.error("give a run dir and a count, e.g. "
                    "`human_trajectory_eval.py runs/<stamp>_d2i_bench/carsales-easy 5` "
                    "(or use --resume/--report)")
        if args.n < 1:
            raise SystemExit("the number of trajectories must be at least 1")
        if args.min_candidates < 2:
            raise SystemExit("--min-candidates must be at least 2 — a pool of one is not a choice")

        paths = run_dirs(args.path)
        if not paths:
            raise SystemExit(f"no run dirs (nothing holding a report.json) under {args.path}")

        seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
        rng = random.Random(seed)
        decisions, notes = build_decisions(
            paths, args.n, rng, args.min_candidates, args.drop_unanswerable
        )
        for t in notes:
            print(t)
        if not decisions:
            raise SystemExit(
                f"no decision points under {args.path} — a run needs both repository.json "
                "and report.json's pruned_questions to reconstruct one")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = {
            "stamp": stamp,
            "judge": args.judge,
            "path": str(args.path),
            "runs": sorted({d["run_dir"] for d in decisions}),
            "n_requested": args.n,
            "min_candidates": args.min_candidates,
            "drop_unanswerable": args.drop_unanswerable,
            "feedback": args.feedback,
            "seed": seed,
            "notes": notes,
            "decisions": decisions,
            "verdicts": [],
        }
        label = _UNSAFE.sub("-", args.path.name) or "runs"
        session_path = (args.session_dir or (OUT_DIR / f"{stamp}_{label}")) / "session.json"
        save_session(session, session_path)
        judge(session, session_path)

    lines = report(session)
    print("\n" + "\n".join(lines))

    (session_path.parent / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {session_path.parent / 'results.txt'}")
    if str(args.out) != "-":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
