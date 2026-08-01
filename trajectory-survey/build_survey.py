#!/usr/bin/env python3
"""Freeze one sampled human-vs-judge rating study into a static bundle.

The G-Eval judge (`eval/geval_trajectory.py` in the D2I repo) scores every trajectory of
every run on six criteria, 1-5, three repeats each. This builds the human half of that
same measurement: a sample of those trajectories, shown to a person with **exactly the
content the judge was given** -- the analytical goal, the dataset profile, and the
trajectory's statements, evidence and system metadata, rendered by the judge's own
`_format_trajectory` -- and scored against the judge's own rubric text.

Nothing here re-implements the judge. The module that produced the runs is imported and
called, and the build refuses to proceed unless its prompt hash matches the hash recorded
in the scoring run's `meta.json`. A rubric edit therefore fails the build instead of
silently producing a survey that measures a different instrument.

    python3 trajectory-survey/build_survey.py                    # every dataset, 1 per system
    python3 trajectory-survey/build_survey.py -d 2 -n 2          # 2 datasets, 2 per system per dataset
    python3 trajectory-survey/build_survey.py --seed 7 --no-reveal

Sampling. Datasets are drawn first and are shared by all three systems -- every system is
judged on the same datasets, so a system difference is never a dataset difference. Within
a dataset each system's trajectories are then drawn independently: which trajectory
indices come from D2I has nothing to do with which come from QUIS.

Blinding. `data/survey.json` names no system and carries no score. The system behind each
item is in `data/truth.json`, which is published for the closing screen (as in the earlier
survey, `--no-reveal` withholds it entirely) and in `private/build.json`, which is not.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent

DEFAULT_GEVAL_RUN = SITE / "runs" / "geval_runs" / "20260731-133448_systems"
DEFAULT_MODULE = Path("/Users/shaolinx/Desktop/D2I_repo/eval/geval_trajectory.py")
DEFAULT_RUNS_ROOT = SITE / "runs"

# The directory name each system's runs live under in the site's runs/ copy.
ARM_DIR = {"agentpoirot": "agentpoirot", "d2i": "d2i", "quis": "quis"}

TITLE = "How good is this analysis? — D2I trajectory rating"

# Shown in place of a criterion the judge is not asked for either (see the judge module's
# ASSIGNED_TRUSTWORTHINESS). Deliberately a paraphrase: the judge's own wording names the
# system, and naming it on the screen would unblind the item. The substance is the same —
# the criterion is fixed by policy because the system persists no claim-level evidence, so
# scoring it would measure an output format rather than the claims.
ASSIGNED_NOTE = (
    "Not required for this trajectory. This system records no claim-level evidence — every "
    "step above says so — because its statements are direct renderings of statistics "
    "computed over the stated subspace, and hold of the data by construction. There is "
    "nothing here for an evidence-grounded criterion to weigh, so it is fixed by policy "
    "rather than judged, and the model is not asked for it either. Answer it if you have a "
    "view; you can move on without it.")


# --------------------------------------------------------------------------- module

def load_geval(module_path: Path):
    """Import the judge module itself, so the survey cannot drift from it.

    It imports a sibling `sweeps`, so its directory goes on the path first.
    """
    if not module_path.is_file():
        raise SystemExit(
            f"build_survey: no judge module at {module_path}\n"
            "  pass --geval-module /path/to/D2I_repo/eval/geval_trajectory.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("geval_trajectory", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["geval_trajectory"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- rubric

def _unwrap(text: str) -> str:
    """Join the rubric's hard-wrapped lines back into paragraphs, so the page can set
    its own measure. Blank lines stay as paragraph breaks."""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        out.append(" ".join(line.strip() for line in para.splitlines() if line.strip()))
    return "\n\n".join(out)


def parse_preamble(text: str) -> dict:
    """The rubric's shared opening: a line of introduction and four numbered steps. The
    numbering is kept as structure rather than run together into a paragraph, because the
    page renders it as a list."""
    intro, *steps = re.split(r"\n(?=\d\. )", text.strip())
    return {"intro": _unwrap(intro),
            "steps": [_unwrap(re.sub(r"^\d\. ", "", s)) for s in steps]}


def parse_rubric(rubric: str) -> tuple[dict, dict]:
    """Split the judge's RUBRIC into its shared preamble and one block per criterion.

    Each block keeps the three things the judge was shown, separately, because the page
    shows them in three different places: the lettered evaluation steps, the question the
    criterion asks, and the five level descriptions the score is chosen from.
    """
    parts = re.split(r"\n--- (.+?) ---\n", rubric)
    preamble = parse_preamble(parts[0])
    blocks = {}
    for title, body in zip(parts[1::2], parts[2::2]):
        key = title.strip().lower().replace(" ", "_")

        steps: list[str] = []
        m = re.search(r"Evaluation steps:\n(.*?)(?=\n\n)", body, flags=re.S)
        if m:
            for chunk in re.split(r"\n\s{2}[a-z]\. ", "\n" + m.group(1))[1:]:
                steps.append(" ".join(x.strip() for x in chunk.strip().splitlines()))
            body = body[m.end():]

        # The scale is the FIRST unbroken run of "n = ..." lines. Information Gain then
        # carries a second, shorter scale below a blank line, for single-step
        # trajectories; parsing greedily would let that one overwrite levels 1-3 of the
        # scale actually being used, which is a silent and very wrong rubric.
        levels, question, notes = {}, [], []
        current, closed = None, False
        for line in body.strip().splitlines():
            lvl = re.match(r"^([1-5]) = (.*)$", line.strip())
            if closed:
                notes.append(line.strip())
            elif lvl:
                current = int(lvl.group(1))
                levels[current] = lvl.group(2).strip()
            elif line.startswith("    ") and current:          # a wrapped level line
                levels[current] += " " + line.strip()
            elif current:                                      # blank line ends the scale
                closed = True
                notes.append(line.strip())
            else:
                question.append(line.strip())
        if len(levels) != 5:
            raise SystemExit(f"build_survey: could not parse 5 levels for {title!r}")
        blocks[key] = {
            "title": title.strip(),
            "steps": steps,
            "question": _unwrap("\n".join(question)),
            "levels": {str(k): v for k, v in sorted(levels.items())},
            "notes": _unwrap("\n".join(notes)),
        }
    return preamble, blocks


# --------------------------------------------------------------------------- runs

def d2i_full_paths(run_dir: Path, trajectories: list[list[dict]], g) -> list[list[dict]]:
    """Repair D2I trajectories that the judge's loader truncates.

    `load_d2i_trajectories` groups `repository.json`'s records by `trajectory_id`. But D2I
    mints a NEW trajectory_id when the beam forks, and the ancestors keep the parent
    branch's id -- so grouping returns the tail SEGMENT of a path, not the path. Across
    this repository's D2I runs that truncates 95 of 129 reported trajectories, 44 of them
    all the way down to a single node sitting at depth 3, with no base insight in sight.

    That matters more here than almost anywhere: Resolution and Information Gain are
    defined against the base insight and against what came before, so a lone depth-3 node
    is being rated -- by a person or by the judge -- on progression it was never shown.

    The real path is recoverable exactly, because every record carries its own `id` and its
    `parent_id`: walk from the deepest node of the reported trajectory back to the root.
    Node dicts are rebuilt in the loader's own shape, with evidence from the judge's own
    `d2i_node_evidence`, so nothing else about the rendering changes.
    """
    repo_path = run_dir / "repository.json"
    if not repo_path.is_file():
        return trajectories
    records = json.loads(repo_path.read_text(encoding="utf-8"))["records"]
    by_id = {r["id"]: r for r in records}
    by_traj: dict[str, list[dict]] = {}
    for r in records:
        by_traj.setdefault(r["trajectory_id"], []).append(r)

    reported = [t for t in g.reported_d2i_trajectory_ids(run_dir) if t in by_traj]
    order = reported or list(by_traj)
    if len(order) != len(trajectories):
        return trajectories               # not the shape we know how to repair

    out = []
    for tid, loaded in zip(order, trajectories):
        tail = max(by_traj[tid], key=lambda r: r["depth"])
        path, seen, cur = [], set(), tail
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            path.append(cur)
            cur = by_id.get(cur.get("parent_id"))
        path.reverse()
        if len(path) <= len(loaded):
            out.append(loaded)            # nothing to add: leave the loader's own answer
            continue
        out.append([{"description": r["label"], "depth": r["depth"],
                     "evidence": g.d2i_node_evidence(r)} for r in path])
    return out


def run_dir_for(rec: dict, runs_root: Path, geval_root: Path) -> Path:
    """Where this scored run's artifacts are.

    The site's `runs/` copy is preferred -- it is the one that travels with this
    repository -- and the original path recorded in the scoring output is the fallback.
    Both were checked to load byte-identical trajectories.
    """
    arm = ARM_DIR.get(rec["arm"], rec["arm"])
    local = runs_root / arm / f"{rec['dataset']}-{rec['level']}"
    if local.is_dir():
        return local
    original = geval_root / rec["run_dir"]
    if original.is_dir():
        return original
    raise SystemExit(f"build_survey: no run directory for {rec['arm']}/{rec['dataset']} "
                     f"(tried {local} and {original})")


def load_runs(runs_root: Path, level: str, quis_nodes: str, g,
              d2i_as_judged: bool = False) -> list[dict]:
    """One record per run directory, with no judge involved.

    This is the collection-only mode: the trajectories are loaded exactly as the judge's
    loaders load them — same `detect_kind`, same `load_run_trajectories`, same QUIS
    top-k cut from the run's own config — but nothing is paired with a score, so a
    freshly rewritten `runs/` can be put in front of people the same day it lands.

    Run discovery is `sweeps.discover`, the judge's own, so "which arm, which dataset,
    what config did it run under" is answered the same way here as there.
    """
    refs = g.sweeps.discover(runs_root, level=level)
    out = []
    for ref in refs:
        try:
            kind = g.detect_kind(ref.path)
        except ValueError as e:
            print(f"  skip {ref.name}: {e}")
            continue
        goal, trajectories, _ = g.load_run_trajectories(
            ref.path, kind, ref.config, quis_nodes=quis_nodes)
        if kind == "d2i" and not d2i_as_judged:
            fixed = d2i_full_paths(ref.path, trajectories, g)
            repaired = sum(1 for a, b in zip(trajectories, fixed) if len(b) > len(a))
            if repaired:
                print(f"  {ref.name}: rebuilt {repaired} truncated trajectory/ies from "
                      "parent_id (see d2i_full_paths)")
            trajectories = fixed
        if not trajectories:
            print(f"  skip {ref.name}: no trajectories load from it")
            continue
        if not goal:
            print(f"  skip {ref.name}: no goal recorded")
            continue
        out.append({
            "arm": ref.arm, "system": kind, "dataset": ref.dataset, "level": ref.level,
            "run_dir": ref.path, "goal": goal, "trajectories": trajectories,
            "indices": list(range(len(trajectories))),
            "scored": None,
            # The same policy the judge applies, read from the judge: a criterion it fixes
            # rather than asks for is one a human should not be held to either.
            "assigned": (("trustworthiness",)
                         if kind in g.ASSIGNED_TRUSTWORTHINESS else ()),
        })
    if not out:
        raise SystemExit(f"build_survey: no usable runs under {runs_root}")
    return out


def load_scored_runs(geval_run: Path, runs_root: Path, g) -> list[dict]:
    """One record per scored run: its judge scores, and the trajectories the judge saw.

    The alignment between a `trajectory_index` in the scores and a trajectory loaded here
    is the whole basis of the study, so it is checked rather than trusted: every
    trajectory's step count and depth must match what the scoring run recorded for that
    index.
    """
    runs_dir = geval_run / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"build_survey: no runs/ under {geval_run}")

    geval_root = Path(g.ROOT)
    out = []
    for path in sorted(runs_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        run_dir = run_dir_for(rec, runs_root, geval_root)
        kind = g.detect_kind(run_dir)
        if kind != rec["system"]:
            raise SystemExit(f"build_survey: {run_dir} looks like {kind}, "
                             f"but {path.name} was scored as {rec['system']}")
        goal, trajectories, _ = g.load_run_trajectories(
            run_dir, kind, rec["run_config"], quis_nodes=rec.get("quis_nodes", "reported"))
        goal = rec["goal"] or goal

        for scored in rec["trajectories"]:
            i = scored["trajectory_index"]
            if i >= len(trajectories):
                raise SystemExit(f"build_survey: {path.name} scored trajectory {i}, "
                                 f"but only {len(trajectories)} load from {run_dir}")
            traj = trajectories[i]
            depth = max((n["depth"] for n in traj), default=None)
            if (len(traj), depth) != (scored["n_steps"], scored["max_depth"]):
                raise SystemExit(
                    f"build_survey: trajectory {i} of {path.name} does not match what was "
                    f"scored ({len(traj)} steps/depth {depth} here, "
                    f"{scored['n_steps']}/{scored['max_depth']} recorded) — the run "
                    "directory has changed since it was judged")

        out.append({
            "arm": rec["arm"], "system": rec["system"], "dataset": rec["dataset"],
            "level": rec["level"], "run_dir": run_dir, "goal": goal,
            "trajectories": trajectories,
            # Only the trajectories the judge actually scored are offered, so every item
            # has something to be compared against.
            "indices": [t["trajectory_index"] for t in rec["trajectories"]],
            "scored": {t["trajectory_index"]: t for t in rec["trajectories"]},
            "assigned": tuple(rec.get("assigned_criteria") or ()),
        })
    if not out:
        raise SystemExit(f"build_survey: no scored runs under {runs_dir}")
    return out


def steps_of(traj: list[dict]) -> list[dict]:
    """The judge's own rendering, split back into fields the page can style.

    `_format_trajectory` is the authority on order, on the `[base]`/`[+n]` tags and on
    what an empty evidence string prints as; this walks the same sorted nodes so the three
    parts can sit in three elements instead of one preformatted block.
    """
    out = []
    for i, node in enumerate(sorted(traj, key=lambda n: n["depth"])):
        metadata = node.get("metadata") or {}
        out.append({
            "tag": "[base]" if i == 0 else f"[  +{i}]",
            "statement": node["description"],
            "evidence": (node.get("evidence") or "").strip() or "(none recorded)",
            "metadata": ", ".join(f"{k}={v}" for k, v in metadata.items()) if metadata else "",
            "depth": node["depth"],
        })
    return out


def item_id(run: dict, index: int, seed: int) -> str:
    """An opaque, stable id. Opaque because a readable one ('quis-3') would unblind the
    item it names, and stable so the same seed rebuilds the same bundle."""
    key = f"{seed}|{run['arm']}|{run['dataset']}|{run['level']}|{index}"
    return hashlib.blake2b(key.encode(), digest_size=4).hexdigest()


# --------------------------------------------------------------------------- sampling

def sample(runs: list[dict], n_datasets: int | None, per_system: int,
           rng: random.Random) -> list[tuple[dict, int]]:
    """(run record, trajectory index) pairs, grouped by dataset in a shuffled order.

    Datasets first, shared by every system: a comparison across systems is only a
    comparison of systems if they were all judged on the same data. Trajectories within a
    dataset are then drawn per system and independently -- there is no reason for the
    third trajectory of one system to be paired with the third of another.

    Items are presented grouped by dataset because each dataset carries a ~6 000-character
    profile the respondent has to read; interleaving datasets would make them re-read it
    on every screen, and that fatigue is worse than any order effect grouping introduces.
    """
    by_dataset: dict[str, list[dict]] = {}
    for r in runs:
        by_dataset.setdefault(r["dataset"], []).append(r)

    systems = {r["system"] for r in runs}
    complete = sorted(ds for ds, rs in by_dataset.items()
                      if {r["system"] for r in rs} == systems)
    if not complete:
        raise SystemExit("build_survey: no dataset has a run from every system")
    dropped = sorted(set(by_dataset) - set(complete))
    if dropped:
        print(f"  note: {len(dropped)} dataset(s) are not present for all "
              f"{len(systems)} systems and are not drawn from: {', '.join(dropped)}")

    chosen = sorted(rng.sample(complete, min(n_datasets, len(complete)))) if n_datasets else complete
    if n_datasets and n_datasets > len(complete):
        print(f"  note: asked for {n_datasets} datasets, only {len(complete)} have a run "
              "from every system")

    order = list(chosen)
    rng.shuffle(order)

    picked: list[tuple[dict, int]] = []
    for ds in order:
        block: list[tuple[dict, int]] = []
        for run in sorted(by_dataset[ds], key=lambda r: r["arm"]):
            available = list(run["indices"])
            k = min(per_system, len(available))
            if k < per_system:
                print(f"  note: {run['arm']}/{ds} has only {len(available)} "
                      f"trajectories, asked for {per_system}")
            for i in sorted(rng.sample(available, k)):
                block.append((run, i))
        rng.shuffle(block)                 # no system is systematically judged first
        picked.extend(block)
    return picked


# --------------------------------------------------------------------------- bundle

SCORE_KEYS = {"score", "scores", "sd", "min", "max", "reason", "reasons", "assigned"}


def assert_no_scores(public: dict, criteria: tuple[str, ...]) -> None:
    """The public half must never answer the questions it asks.

    A judge score in the served bundle would make every rating collected against it
    worthless, and the failure would be silent, so it is checked rather than assumed.

    Note what this no longer checks. The system behind each item is carried in the public
    bundle by default (see `--blind-systems`), so it is not searched for here.
    """
    for item in public["items"]:
        for crit in criteria:
            block = item.get(crit)
            if isinstance(block, dict) and SCORE_KEYS & set(block):
                raise SystemExit(f"build_survey: item {item['id']} carries judge scores")
    if any(k in public for k in ("scores", "truth", "key")):
        raise SystemExit("build_survey: the public bundle carries an answer key")


def assert_blind(public: dict, criteria: tuple[str, ...]) -> None:
    """`--blind-systems`: the served bundle must additionally not say which system wrote
    each item, so a respondent cannot look one up while rating it."""
    assert_no_scores(public, criteria)
    text = json.dumps(public)
    for word in ("agent_poirot", "agentpoirot", "\"quis\"", "QUIS"):
        if word in text:
            raise SystemExit(f"build_survey: --blind-systems, but the public bundle "
                             f"mentions {word!r} — that would unblind the items")


def identify(args) -> int:
    """Say which trajectory each rated item was, from the response file alone.

    Every answer carries the sha256 of the judge prompt its trajectory rendered to. This
    re-renders every trajectory under `--runs-root` and matches on that hash, so a returned
    response can be resolved back to `<system>/<dataset> #<index>` with nothing but
    `runs/` — no private bundle, no answer key, no trust in an id that only names
    coordinates. It is also how you find out that a trajectory has since been rewritten:
    it simply will not match.
    """
    g = load_geval(args.geval_module)
    R = json.loads(args.identify.read_text(encoding="utf-8"))
    wanted: dict[str, list[str]] = {}
    for a in R.get("answers", []):
        if a.get("content"):
            wanted.setdefault(a["content"], []).append(a["item"])
    if not wanted:
        raise SystemExit(
            f"build_survey: {args.identify} carries no content fingerprints — it was "
            "collected\n  by a build older than those. Resolve it with private/build.json "
            "instead, matching\n  on the item ids.")

    print(f"{len(wanted)} distinct trajectory/ies rated in {args.identify.name}; "
          f"searching {args.runs_root}")
    runs = load_runs(args.runs_root, args.level, args.quis_nodes, g,
                     d2i_as_judged=args.d2i_as_judged)
    profiles: dict[str, str] = {}
    found: dict[str, tuple] = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in profiles:
            try:
                profiles[ds] = g.load_profile_text(
                    ds, g.load_dataset_profile(ds, args.profile))
            except (ValueError, KeyError):
                continue                       # no profile: cannot rebuild its prompts
        for index, traj in enumerate(run["trajectories"]):
            prompt = g._build_user_prompt(run["goal"], profiles[ds], traj, run["assigned"])
            h = hashlib.sha256(prompt.encode()).hexdigest()
            if h in wanted and h not in found:
                found[h] = (run["system"], ds, index, run["run_dir"], run, traj)

    for h, ids in sorted(wanted.items(), key=lambda kv: kv[1]):
        where = found.get(h)
        label = ", ".join(sorted(set(ids)))
        if where:
            system, ds, index, path = where[:4]
            print(f"  {label}  ->  {system}/{ds} trajectory #{index}\n"
                  f"            {path}")
        else:
            print(f"  {label}  ->  NOT FOUND under {args.runs_root} — that trajectory has "
                  f"been rewritten\n            or lives elsewhere ({h[:16]}…)")
    n = len(found)
    print(f"\n{n}/{len(wanted)} resolved")

    if args.annotate:
        # An UNBLINDED copy of the response: every answer gains the system, dataset and
        # trajectory index behind it, resolved by content hash rather than taken on trust.
        # It is written here, offline, and never during collection — the respondent's own
        # download stays blind, because a file that names the system is a file that can be
        # opened in another tab and then rated against.
        out = json.loads(json.dumps(R))
        for a in out.get("answers", []):
            where = found.get(a.get("content"))
            if not where:
                a["resolved"] = None
                continue
            system, ds, index, path, run, traj = where
            a["resolved"] = {
                "system": system, "dataset": ds, "trajectory_index": index,
                "run_dir": str(path),
                "goal": run["goal"],
                # The trajectory itself, re-rendered from the run by the judge's own
                # loaders. A submitted response carries no step text (it has to fit in one
                # spreadsheet cell), so this is where a Sheets export gets it back.
                "steps": steps_of(traj),
            }
        out["resolved_against"] = {"runs_root": str(args.runs_root),
                                   "level": args.level,
                                   "quis_nodes": args.quis_nodes,
                                   "matched": n, "of": len(wanted)}
        args.annotate.parent.mkdir(parents=True, exist_ok=True)
        args.annotate.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.annotate} — the same response with each answer's system, "
              "dataset and\n  trajectory index attached")
    return 0 if n == len(wanted) else 1


def attach_scores(args) -> int:
    """Write the answer key for a bundle that was built without one.

    A collection-only build has no `truth.json`, and rebuilding to get one would move
    `built_at` and every item id — which is exactly what `score_survey.py` refuses, and
    rightly: those would be answers to different questions. So the key is produced for the
    EXISTING items instead, from the private bundle they were recorded in.

    The check that makes this safe is the prompt receipt. Each item stored the sha256 of
    the exact judge prompt its trajectory rendered to at build time; here that prompt is
    rebuilt from the run and the scoring run's config and must hash the same. If a run
    directory has been rewritten since people rated it, this fails instead of quietly
    pairing a human rating with a different trajectory's score.
    """
    g = load_geval(args.geval_module)
    priv_path = HERE / "private" / "build.json"
    if not priv_path.is_file():
        raise SystemExit(f"build_survey: no {priv_path} — the bundle it describes is the "
                         "only thing that can be scored after the fact")
    priv = json.loads(priv_path.read_text(encoding="utf-8"))
    meta = priv["meta"]

    meta_path = args.geval_run / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"build_survey: no meta.json under {args.geval_run}")
    gmeta = json.loads(meta_path.read_text(encoding="utf-8"))
    if gmeta.get("prompt_hash") != g._prompt_hash() and not args.allow_hash_mismatch:
        raise SystemExit(
            f"build_survey: {meta_path} was scored with prompt {gmeta.get('prompt_hash')}, "
            f"the module here builds {g._prompt_hash()} — different instruments")

    runs = {(r["arm"], r["dataset"], r["level"]): r
            for r in load_scored_runs(args.geval_run, args.runs_root, g)}
    criteria = tuple(gmeta.get("criteria") or g.CRITERIA)

    profiles: dict[str, str] = {}
    truth, missing, moved = {}, [], []
    for item in priv["items"]:
        key = (item["arm"], item["dataset"], meta.get("level", "level_2"))
        run = runs.get(key) or runs.get((item["arm"], item["dataset"], "level_2"))
        index = item["trajectory_index"]
        if not run or index not in (run["scored"] or {}):
            missing.append(f"{item['arm']}/{item['dataset']} #{index}")
            continue
        ds = item["dataset"]
        if ds not in profiles:
            profiles[ds] = g.load_profile_text(ds, g.load_dataset_profile(ds, args.profile))
        traj = run["trajectories"][index]
        prompt = g._build_user_prompt(run["goal"], profiles[ds], traj, run["assigned"])
        if hashlib.sha256(prompt.encode()).hexdigest() != item.get("prompt_sha256"):
            moved.append(f"{item['arm']}/{ds} #{index}")
            continue
        scored = run["scored"][index]
        truth[item["id"]] = {
            "system": run["system"], "arm": run["arm"], "dataset": ds,
            "level": run["level"], "trajectory_index": index,
            "assigned_criteria": list(run["assigned"]),
            "n_repeats": scored.get("n_repeats"),
            "scores": {c: dict(scored[c]) for c in criteria if c in scored},
        }

    if moved:
        raise SystemExit(
            "build_survey: these trajectories are not what was shown when the ratings were "
            "collected —\n  " + "\n  ".join(moved) +
            "\n  The run directories have changed since the bundle was built, so their "
            "scores cannot be\n  attached to those ratings. Score them against the runs "
            "people actually saw.")
    if not truth:
        raise SystemExit("build_survey: none of the bundle's items are in that scoring run")
    if missing:
        print(f"  note: {len(missing)} item(s) are not in that scoring run and get no key: "
              + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else ""))

    key_meta = {"built_at": meta["built_at"], "seed": meta["seed"],
                "criteria": list(criteria),
                "geval": {
                    "tag": gmeta.get("tag"), "written_at": gmeta.get("written_at"),
                    "judge_model": gmeta.get("judge_model"),
                    "temperature": gmeta.get("temperature"),
                    "rubric_version": gmeta.get("rubric_version"),
                    "prompt_hash": gmeta.get("prompt_hash"),
                    "repeats": gmeta.get("repeats"),
                    "run": str(args.geval_run.relative_to(SITE)
                               if args.geval_run.is_relative_to(SITE) else args.geval_run),
                },
                "attached_after_collection": True}
    out = HERE / "data" / "truth.json"
    out.write_text(json.dumps({"meta": key_meta, "items": truth}, indent=1) + "\n",
                   encoding="utf-8")
    priv["truth"] = truth
    priv_path.write_text(json.dumps(priv, indent=1) + "\n", encoding="utf-8")
    print(f"attached scores for {len(truth)}/{len(priv['items'])} items of the bundle built "
          f"{meta['built_at']}\n  every prompt hash matched — the ratings were given on "
          f"exactly these trajectories\nwrote {out}\n      {priv_path}")
    print("\n  score the responses with:\n"
          "    python3 trajectory-survey/score_survey.py trajectory-survey/responses/")
    return 0


def build(args) -> int:
    g = load_geval(args.geval_module)
    gmeta: dict = {}

    if args.no_judge:
        # Collection-only: no scoring run is consulted at all, so `runs/` can be replaced
        # and put in front of people before anything has judged it. Nothing is published
        # for the closing screen to score against, which is why --no-judge forces
        # --no-reveal below.
        print(f"no-judge build: trajectories from {args.runs_root}, no scores attached")
        runs = load_runs(args.runs_root, args.level, args.quis_nodes, g,
                     d2i_as_judged=args.d2i_as_judged)
        print(f"loaded {len(runs)} runs, "
              f"{sum(len(r['trajectories']) for r in runs)} trajectories")
    else:
        meta_path = args.geval_run / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"build_survey: no meta.json under {args.geval_run}\n"
                             "  build with --no-judge to collect ratings without one")
        gmeta = json.loads(meta_path.read_text(encoding="utf-8"))

        if gmeta.get("prompt_hash") != g._prompt_hash():
            msg = (f"build_survey: the judge module's prompt hash is {g._prompt_hash()}, but "
                   f"{meta_path} was scored with {gmeta.get('prompt_hash')} "
                   f"(rubric {gmeta.get('rubric_version')} vs {g.RUBRIC_VERSION}).\n"
                   "  The human half would be measuring a different instrument. Point "
                   "--geval-module at the module that produced these scores.")
            if not args.allow_hash_mismatch:
                raise SystemExit(msg)
            print("WARNING " + msg)

        print(f"judge: {gmeta['judge_model']}, rubric {gmeta['rubric_version']} "
              f"({gmeta['prompt_hash']}), {gmeta['repeats']} repeats")
        runs = load_scored_runs(args.geval_run, args.runs_root, g)
        print(f"loaded {len(runs)} scored runs, "
              f"{sum(len(r['indices']) for r in runs)} judged trajectories")

    criteria = tuple(gmeta.get("criteria") or g.CRITERIA)
    preamble, rubric = parse_rubric(g.RUBRIC)
    missing = [c for c in criteria if c not in rubric]
    if missing:
        raise SystemExit(f"build_survey: no rubric section for {missing}")

    rng = random.Random(args.seed)
    picked = sample(runs, args.datasets, args.per_system, rng)

    profiles: dict[str, str] = {}
    goals: dict[str, str] = {}
    items, truth, private = [], {}, []

    for run, index in picked:
        ds = run["dataset"]
        if ds not in profiles:
            profile = g.load_dataset_profile(ds, args.profile)
            profiles[ds] = g.load_profile_text(ds, profile)
            goals[ds] = run["goal"]
        if goals[ds] != run["goal"]:
            raise SystemExit(f"build_survey: {ds} has two different goals across systems; "
                             "showing them would unblind the items")

        traj = run["trajectories"][index]
        scored = (run["scored"] or {}).get(index)
        iid = item_id(run, index, args.seed)

        # Exactly the user message the judge was sent for this trajectory, hashed. It is
        # the receipt that the human saw the same content, not a reconstruction of it —
        # and in a no-judge build, the receipt of what a judge WOULD be sent for it.
        assigned = tuple(run["assigned"])
        prompt = g._build_user_prompt(run["goal"], profiles[ds], traj, assigned)
        content = hashlib.sha256(prompt.encode()).hexdigest()

        # A criterion the judge is not asked for is still SHOWN — dropping it would make
        # the item identifiable by its shorter scale — but it is marked as not required and
        # says why, so nobody is left scoring evidence that was never recorded and nobody
        # is blocked by a scale that cannot be answered. --match-assigned drops it instead.
        asked = [c for c in criteria
                 if not (args.match_assigned and c in assigned)]
        optional = [] if args.match_assigned else [c for c in criteria if c in assigned]
        items.append({
            "id": iid,
            # The sha256 of the judge prompt this item renders to — a CONTENT name for the
            # trajectory, next to `id`, which only names its coordinates (seed, arm,
            # dataset, index). Those coordinates survive a rewrite of runs/ unchanged, so
            # the id alone cannot tell "d2i/salesfact #1 as it was in July" from "…as it is
            # now"; this can. It is a hash, so it discloses nothing about which system
            # produced the item — and it is what makes a returned response identifiable
            # against runs/ on its own, without the private bundle.
            "content": content,
            "dataset": ds,
            # Which system produced this trajectory, so a returned response says so per
            # trajectory without needing runs/ or the private bundle. It is in the served
            # bundle, therefore readable by a respondent who opens the network tab —
            # --blind-systems leaves it out and keeps the old behaviour.
            **({} if args.blind_systems else {"system": run["system"], "arm": run["arm"]}),
            "n_steps": scored["n_steps"] if scored else len(traj),
            "max_depth": (scored["max_depth"] if scored
                          else max((n["depth"] for n in traj), default=None)),
            "steps": steps_of(traj),
            "criteria": asked,
            **({"optional": optional,
                "notes": {c: ASSIGNED_NOTE for c in optional}} if optional else {}),
        })
        truth[iid] = {
            "system": run["system"],
            "arm": run["arm"],
            "dataset": ds,
            "level": run["level"],
            "trajectory_index": index,
            "content": content,
            "run_dir": str(run["run_dir"]),
            "assigned_criteria": list(assigned),
            "n_repeats": scored.get("n_repeats") if scored else None,
            "scores": ({c: dict(scored[c]) for c in criteria if c in scored}
                       if scored else {}),
        }
        private.append({
            "id": iid, "run_dir": str(run["run_dir"]), "arm": run["arm"],
            "system": run["system"], "dataset": ds, "trajectory_index": index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt),
        })

    # The ids key everything a respondent sends back, so a collision would silently merge
    # two trajectories' ratings. blake2b-32 over 18 items collides with probability ~4e-8 —
    # small enough to be worth one line, not small enough to leave unchecked.
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"build_survey: item id collision on {dupes} — rebuild with a "
                         "different --seed, or widen item_id()'s digest")

    built_at = datetime.now().replace(microsecond=0).isoformat()
    public = {
        "meta": {
            "title": TITLE,
            "built_at": built_at,
            "seed": args.seed,
            "n_items": len(items),
            "datasets": sorted({i["dataset"] for i in items}),
            "per_system": args.per_system,
            "criteria": list(criteria),
            "scale": {"min": 1, "max": 5},
            "reveal": bool(args.reveal),
            "match_assigned": bool(args.match_assigned),
            "blind_systems": bool(args.blind_systems),
            "judged": not args.no_judge,
            # Provenance of the model half, so a response can be traced to the scoring run
            # it will be compared against without this repository. In a no-judge build
            # there is no scoring run yet, so only the instrument is recorded — the rubric
            # these ratings were given against, which is what a later scoring run has to
            # match to be comparable with them.
            "geval": {
                "tag": gmeta.get("tag"),
                "written_at": gmeta.get("written_at"),
                "judge_model": gmeta.get("judge_model"),
                "temperature": gmeta.get("temperature"),
                "rubric_version": gmeta.get("rubric_version") or g.RUBRIC_VERSION,
                "prompt_hash": gmeta.get("prompt_hash") or g._prompt_hash(),
                "repeats": gmeta.get("repeats"),
                "run": (None if args.no_judge else
                        str(args.geval_run.relative_to(SITE)
                            if args.geval_run.is_relative_to(SITE) else args.geval_run)),
            },
        },
        "instructions": {
            "system_prompt": g.SYSTEM_PROMPT,
            "preamble": preamble,
            "criteria": {c: rubric[c] for c in criteria},
        },
        "goals": goals,
        "profiles": profiles,
        "items": items,
    }
    (assert_blind if args.blind_systems else assert_no_scores)(public, criteria)

    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "survey.json").write_text(json.dumps(public, indent=1) + "\n", encoding="utf-8")

    key = {"meta": {"built_at": built_at, "seed": args.seed,
                    "criteria": list(criteria),
                    "geval": public["meta"]["geval"]},
           "items": truth}
    if args.reveal:
        (data / "truth.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    else:
        (data / "truth.json").unlink(missing_ok=True)

    priv = HERE / "private"
    priv.mkdir(exist_ok=True)
    (priv / "build.json").write_text(json.dumps(
        {"meta": public["meta"], "items": private, "truth": truth}, indent=1) + "\n",
        encoding="utf-8")

    by_ds: dict[str, int] = {}
    for i in items:
        by_ds[i["dataset"]] = by_ds.get(i["dataset"], 0) + 1
    print(f"\n{len(items)} items, seed {args.seed}, built {built_at}")
    for ds, n in sorted(by_ds.items()):
        systems = sorted(truth[i["id"]]["system"] for i in items if i["dataset"] == ds)
        print(f"  {ds:24s} {n:2d}  ({', '.join(systems)})")
    n_assigned = sum(1 for t in truth.values() for c in t["assigned_criteria"])
    if args.no_judge:
        print("\n  No judge scores are attached: this build only COLLECTS ratings. The page\n"
              "  closes with a thank-you and no agreement summary, and the responses stay\n"
              "  scorable later — score them against a scoring run of these same runs with\n"
              "  score_survey.py once one exists.")
    elif n_assigned and not args.match_assigned:
        print(f"\n  {n_assigned} criterion cell(s) are assigned by policy rather than judged "
              "(QUIS trustworthiness).\n  They are still asked of the human — the instrument "
              "stays identical across items — but\n  they are excluded from agreement and "
              "reported separately. --match-assigned instead\n  drops them from the item, "
              "which is stricter about matching the judge's prompt and\n  makes those items "
              "identifiable as QUIS.")
    print(f"\nwrote {data / 'survey.json'}"
          + (f"\n      {data / 'truth.json'}" if args.reveal else
             "\n      (no truth.json — built with --no-reveal)")
          + f"\n      {priv / 'build.json'}  (private, gitignored)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--datasets", type=int, default=None, metavar="N",
                   help="how many datasets to draw (default: every dataset scored for "
                        "every system). The same datasets are used for all systems.")
    p.add_argument("-n", "--per-system", type=int, default=1, metavar="N",
                   help="trajectories per system per dataset, drawn independently "
                        "(default 1). Total items = datasets x systems x N.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-judge", action="store_true",
                   help="build from runs/ alone, with no scoring run: sample trajectories "
                        "and collect ratings, attaching no model scores. Implies "
                        "--no-reveal, so the page closes with a thank-you and shows no "
                        "agreement summary. Use this when runs/ has moved on from whatever "
                        "was last judged.")
    p.add_argument("--attach-scores", action="store_true",
                   help="do not build anything: write the answer key for the bundle that "
                        "is already deployed, from --geval-run, keeping its built_at and "
                        "item ids so responses collected under it still score. Each item's "
                        "judge prompt must hash to what it hashed to at build time.")
    p.add_argument("--identify", type=Path, default=None, metavar="RESPONSE.json",
                   help="do not build anything: say which trajectory each rated item in "
                        "that response was, by matching its content fingerprint against "
                        "everything under --runs-root")
    p.add_argument("--annotate", type=Path, default=None, metavar="OUT.json",
                   help="with --identify: write an unblinded copy of the response, each "
                        "answer carrying the system, dataset, trajectory index and the "
                        "trajectory's own text")
    p.add_argument("--level", default="level_2",
                   help="which level's runs to discover in --no-judge mode (default level_2)")
    p.add_argument("--quis-nodes", default="full-path",
                   help="how QUIS trajectories are rendered in --no-judge mode; must match "
                        "what a later scoring run uses (default full-path, as the last one "
                        "used)")
    p.add_argument("--geval-run", type=Path, default=DEFAULT_GEVAL_RUN,
                   help="the scoring run to compare against (a directory holding "
                        "meta.json and runs/)")
    p.add_argument("--geval-module", type=Path, default=DEFAULT_MODULE,
                   help="the judge that produced it; its prompt hash must match")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                   help="where the system run directories live (default: the site's runs/)")
    p.add_argument("--profile", type=Path, default=None,
                   help="bench_profile.json (default: the judge module's own)")
    p.add_argument("--no-reveal", dest="reveal", action="store_false",
                   help="publish no answer key: the page then closes with a thank-you "
                        "and nothing about the judge is readable from the site")
    p.add_argument("--d2i-as-judged", action="store_true",
                   help="do not repair D2I trajectories that the judge's loader truncates "
                        "(see d2i_full_paths). Shows exactly what the last scoring run saw "
                        "— a lone depth-3 node with no base insight, for 95 of 129 of "
                        "them — which is what you want only if you are reproducing that "
                        "run rather than collecting fresh ratings.")
    p.add_argument("--blind-systems", action="store_true",
                   help="leave the system name out of the served bundle, so a respondent "
                        "cannot see which agent produced a trajectory while rating it. "
                        "Their download then does not say either; resolve it afterwards "
                        "with --identify --annotate.")
    p.add_argument("--match-assigned", action="store_true",
                   help="do not ask a human for a criterion the judge was not asked for "
                        "either (QUIS trustworthiness). Stricter prompt fidelity, but it "
                        "makes those items identifiable as QUIS.")
    p.add_argument("--allow-hash-mismatch", action="store_true",
                   help="build even if the judge module's rubric is not the one that "
                        "produced the scores. For inspection only.")
    p.set_defaults(reveal=True)
    args = p.parse_args()

    # There is nothing to reveal without a judge, and a page that fetched a truth.json left
    # over from an earlier build would score these ratings against the wrong trajectories.
    if args.no_judge:
        args.reveal = False

    if args.profile is None:
        g_root = Path(str(args.geval_module.resolve().parents[1]))
        args.profile = g_root / "data" / "d2i_bench" / "bench_profile.json"
    if args.identify:
        return identify(args)
    return attach_scores(args) if args.attach_scores else build(args)


if __name__ == "__main__":
    raise SystemExit(main())
