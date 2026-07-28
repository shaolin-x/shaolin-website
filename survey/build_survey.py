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

What counts as a decision point, how a candidate pool is assembled and how it is shuffled
all come from `human_trajectory_eval`, called directly, so the survey cannot drift from the
CLI on any of it. Only the draw itself is local (`sample_decisions`): trajectories are taken
uniformly, without the CLI's deliberate over-sampling of early-stopped nodes, because this
survey puts the candidate question at every node that has one regardless of the stop answer.
See that function for why the correction is not needed here.

One seed, one bundle: every respondent judges the *same* decisions in the same order, which
is what makes their answers comparable and inter-rater agreement computable at all.

    python survey/build_survey.py                       # the default pool, 20 decisions
    python survey/build_survey.py -n 12 --seed 3
    python survey/build_survey.py --runs runs           # sample from every run instead
    python survey/build_survey.py --no-reveal           # omit truth.json, strict blinding
"""

from __future__ import annotations

import argparse
import json
import math
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


def data_sample(report: dict) -> dict | None:
    """The run's sampled rows, trimmed to what the page draws.

    `report.json`'s `data_sample` is the same handful of rows the agent itself was shown, so
    putting it above the question costs no blinding: it says what the data *looks like*, not
    what anyone concluded from it. Only the shape and the rows are kept — `seed` and `sampled`
    describe how the sample was drawn, which the respondent has no use for.
    """
    s = report.get("data_sample") or {}
    rows = s.get("rows") or []
    if not rows:
        return None
    return {
        "n_rows": s.get("n_rows"),
        "n_columns": s.get("n_columns"),
        "columns": s.get("columns") or list(rows[0].keys()),
        "rows": rows,
    }


def run_provenance(run_dir: Path, report: dict) -> dict:
    """Where a run came from, for the response file to carry.

    A returned response says "on decision X I picked slot 3". That is enough to score
    against the private bundle, but it is not enough to *read* — six months later nobody can
    tell which dataset, which run or which depth a decision belonged to without going back
    to `runs/`. This travels with the download so the file explains itself.

    None of it is a tell: the goal and the schema are already on the screen the respondent
    answered, and the model and beam settings say how the trajectories were produced, not
    which candidate won.
    """
    meta: dict = {"run_dir": run_dir.name, "parent": run_dir.parent.name,
                  "path": str(run_dir), "goal": report.get("goal") or ""}
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {}
    for key in ("provider", "model", "beam_width", "max_depth", "depth_reached",
                "questions_answered", "n_insights", "started_at", "duration_s"):
        if run.get(key) is not None:
            meta[key] = run[key]
    sample = report.get("data_sample") or {}
    if sample.get("n_rows") is not None:
        meta["dataset"] = {"n_rows": sample.get("n_rows"), "n_columns": sample.get("n_columns")}
    schema = report.get("schema") or {}
    if schema.get("kept"):
        meta["columns"] = [f"{c} ({role})" for c, role in schema["kept"].items()]
    return meta


def sample_decisions(
    paths: list[Path], n: int, rng: random.Random, min_candidates: int, drop_unanswerable: bool,
    share: tuple[float, float] = (0.30, 0.70), strict: bool = False,
) -> tuple[list[dict], list[str], dict[str, dict], dict[str, dict]]:
    """`(decisions, notes, samples, sources)` — n decision points, one per trajectory,
    early-stopped nodes held to `share` of the sample, plus each contributing run's data
    sample and its provenance, both keyed by run name.

    `hte.build_decisions` front-loads early-stopped nodes and caps them at half the sample.
    That correction exists because the CLI ends a decision the moment the judge says
    "terminate", so a terminated node — which has no candidates — contributes nothing to the
    second question. This survey asks the candidate question at every node that has one
    whatever the stop answer was, so the cap is not what is needed; what is needed is simply
    that neither kind of node dominates, which is what `share` bounds.

    Within the band the *natural* rate is kept: the target is the pool's own proportion of
    early-stopped nodes, clamped into `[lo, hi]`. A pool that is already balanced is left
    alone, and only a lopsided one is corrected, by the least amount that satisfies the band.

    The band is a target, not a precondition. When the pool holds too few early-stopped nodes
    to reach `lo` the draw takes every one it can find, keeps the requested `n`, and says in a
    note that the share came out under the band — so adding runs to the pool later raises the
    share on its own, with no change here. `strict=True` instead shortens the sample until the
    band genuinely holds, for when the share has to be a guarantee rather than an aim.

    Everything else is hte's, called directly — which nodes count as decision points, how a
    candidate pool is assembled, and the blinding shuffle.
    """
    notes: list[str] = []
    by_run: dict[str, dict[str, dict]] = {}
    samples: dict[str, dict] = {}           # run *name* — the only run id a public decision carries
    sources: dict[str, dict] = {}           # same key: where that run came from
    pool: list[tuple[str, list[str]]] = []

    for run_dir in paths:
        loaded = hte.load_run(run_dir)
        if isinstance(loaded, str):
            notes.append(f"{run_dir.name}: skipped — {loaded}")
            continue
        report, repository = loaded
        points, trajs, run_notes = hte.decision_points(
            run_dir, report, repository, min_candidates, drop_unanswerable
        )
        notes += [f"{run_dir.name}: {t}" for t in run_notes]
        if not points:
            notes.append(f"{run_dir.name}: skipped — no node offers {min_candidates}+ candidates")
            continue
        by_run[str(run_dir)] = points
        sources[run_dir.name] = run_provenance(run_dir, report)
        sample = data_sample(report)
        if sample:
            samples[run_dir.name] = sample
        else:
            notes.append(f"{run_dir.name}: report.json carries no data_sample — "
                         "its decisions show the columns but no rows")
        pool += [(str(run_dir), t) for t in trajs]

    if not pool:
        return [], notes, samples, sources

    rng.shuffle(pool)                       # the trajectory, drawn at random

    def stopped(run_key: str, rid: str) -> bool:
        return by_run[run_key][rid].get("d2i_terminate") is True

    def nodes(i: int, want_stopped: bool, used: set) -> list[str]:
        """The still-unused decision points of trajectory `i`, of the kind asked for."""
        run_key, traj = pool[i]
        pts = by_run[run_key]
        return [r for r in traj
                if r in pts and (run_key, r) not in used and stopped(run_key, r) is want_stopped]

    # What the pool could yield: distinct early-stopped nodes, and how many trajectories can
    # reach one — a trajectory contributes at most one decision point, so both bound the draw.
    all_stop = {(k, r) for k in by_run for r in by_run[k] if stopped(k, r)}
    reachable = len({rid for i in range(len(pool)) for rid in nodes(i, True, set())})
    avail_stop = min(len(all_stop), reachable,
                     sum(1 for i in range(len(pool)) if nodes(i, True, set())))
    total_pts = sum(len(v) for v in by_run.values())
    natural = len(all_stop) / total_pts if total_pts else 0.0

    lo, hi = share
    floor_of = lambda k: math.ceil(lo * k - 1e-9)
    ceil_of = lambda k: math.floor(hi * k + 1e-9)

    requested = n
    if strict:
        # Shorten until the band's floor is reachable. The band is only meaningful as a
        # promise about the sample that is actually produced.
        while n > 1 and floor_of(n) > avail_stop:
            n -= 1
        if n < requested:
            notes.append(
                f"shortened the sample to {n} from {requested}: the pool holds only "
                f"{avail_stop} early-stopped decision point(s), and {requested} would need at "
                f"least {floor_of(requested)} to stay within "
                f"{lo:.0%}/{1 - lo:.0%}–{hi:.0%}/{1 - hi:.0%}"
            )

    want_stop = min(max(round(natural * n), floor_of(n)), ceil_of(n), avail_stop)

    decisions: list[dict] = []
    used: set[tuple[str, str]] = set()
    used_traj: set[int] = set()

    def take(i: int, want_stopped: bool) -> bool:
        run_key, _ = pool[i]
        avail = nodes(i, want_stopped, used)
        if not avail:
            return False
        rid = rng.choice(avail)             # the depth, drawn at random along the trajectory
        used.add((run_key, rid))
        used_traj.add(i)
        decisions.append(hte.blind(json.loads(json.dumps(by_run[run_key][rid])), rng))
        return True

    # Early-stopped first, to the target, then everything else fills the sample. Trajectories
    # share prefixes, so `used` keeps the nodes distinct and `used_traj` keeps to hte's rule
    # of at most one decision point per trajectory.
    n_stop = 0
    for i in range(len(pool)):
        if n_stop >= want_stop:
            break
        if i not in used_traj and take(i, True):
            n_stop += 1
    for i in range(len(pool)):
        if len(decisions) >= n:
            break
        if i not in used_traj:
            take(i, False)

    # One decision point per trajectory caps the sample at the number of trajectories, which
    # is below `n` on a small pool. Since the requested size is to be held, keep going over
    # trajectories already drawn from — the *nodes* are still distinct, so the candidate pools
    # are too, and nothing is judged twice. Only the "at most one per trajectory" spacing is
    # given up, and only once nothing else is left.
    if len(decisions) < n:
        before = len(decisions)
        for want_stopped in (True, False):
            for i in range(len(pool)):
                while len(decisions) < n and take(i, want_stopped):
                    pass
                if len(decisions) >= n:
                    break
        if len(decisions) > before:
            notes.append(
                f"took {len(decisions) - before} extra decision point(s) from trajectories "
                f"already drawn from, to hold the sample at {len(decisions)}: only "
                f"{len(pool)} trajectory(ies) exist, and the nodes are still all distinct"
            )

    # Present a run's decisions together. The draw above is unchanged — which trajectories,
    # which depths and how many early-stopped ones are all still random — this only fixes the
    # order they are shown in, so a respondent reads one dataset's goal and columns, judges
    # every node sampled from it, and only then moves to the next. Jumping between datasets
    # costs a re-read of the schema at every screen, and that fatigue is a worse bias than any
    # order effect grouping introduces.
    #
    # Both levels are still shuffled: the run order, and the decisions inside each run. So the
    # early-stopped nodes are not bunched at the front of a run, and no dataset is
    # systematically judged first across rebuilds.
    grouped: dict[str, list[dict]] = {}
    for d in decisions:
        grouped.setdefault(d["run"], []).append(d)
    order = list(grouped)
    rng.shuffle(order)
    decisions = []
    for run_name in order:
        block = grouped[run_name]
        rng.shuffle(block)
        decisions += block

    seen: dict[str, int] = {}
    for d in decisions:
        base = f"{d['run']}#{d['node_id'][:8]}"
        seen[base] = seen.get(base, 0) + 1
        d["id"] = base if seen[base] == 1 else f"{base}-{seen[base]}"

    blocks = []
    for run_name in order:
        n_stop = sum(1 for d in grouped[run_name] if d.get("d2i_terminate") is True)
        blocks.append(f"{run_name} ({len(grouped[run_name])}"
                      + (f", {n_stop} early-stopped)" if n_stop else ")"))
    if len(order) > 1:
        notes.append("shown in run order: " + " -> ".join(blocks))

    got = sum(1 for d in decisions if d.get("d2i_terminate") is True)
    if decisions:
        pct = got / len(decisions)
        notes.append(
            f"early-stopped share: {got}/{len(decisions)} ({pct:.0%}) — "
            f"target band {lo:.0%}–{hi:.0%}, the pool's own rate is {natural:.0%}"
            + ("" if lo - 1e-9 <= pct <= hi + 1e-9 else "  ** OUTSIDE THE BAND **")
        )
    if len(decisions) < n:
        notes.append(
            f"sampled {len(decisions)} of the {n} sought — "
            f"{len(pool)} trajectory(ies) across {len(by_run)} run(s), "
            "distinct decision points exhausted"
        )
    return decisions, notes, samples, sources


def truth_entry(decision: dict) -> dict:
    """The answer key for one decision — `human_trajectory_web.reveal_payload` without the
    verdict, since at build time nobody has answered yet. The page fills in the two
    verdict-shaped fields (which row was picked, what the human said about stopping) from
    the response it is holding, so the rank it reports is the same number the CLI prints."""
    slot_of = {t: k for k, t in enumerate(decision["order"], 1)}
    cands = decision["candidates"]
    cont = decision.get("continuation") or {}
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
        # Every metric this decision can contribute to, precomputed once per slot. The page
        # looks a row up by the slot the respondent clicked and averages; it never evaluates
        # a formula, so there is no second implementation of the metrics to drift from this
        # one. Publishing it leaks nothing: each row is a function of the scores already in
        # `rows` above.
        "picks": {str(slot_of[i]): hte.pick_row(decision, i) for i in range(len(cands))},
        # The stop half, minus the human's answer — `metrics.js` fills in `human`/`agree`.
        "stop": {
            "model": decision.get("d2i_terminate"),
            "source": decision.get("terminate_source"),
            "utility": cont.get("utility"),
            "threshold": cont.get("threshold"),
            "margin": decision.get("stop_margin"),
            "criteria": cont.get("criteria"),
            "cluster": decision.get("trajectory_id") or decision["id"],
            "run": decision["run"],
            "depth": decision["depth"],
        },
    }


# Exactly the keys `public_decision` may emit. Asserted at build time rather than only in a
# test, because a test can be skipped and the build cannot: anything else here is a leak.
PUBLIC_KEYS = {"id", "run", "node", "depth", "child_depth", "goal", "columns", "path",
               "ask_stop", "candidates", "data_sample"}

# Substrings that must never appear anywhere in the public bundle. `score` catches the
# candidate scores and every `score_breakdown`; `utility` the judge's ruling.
FORBIDDEN = ("score", "utility", "n_selected", "threshold", "d2i_terminate", "status",
             "order", "terminate_source", "breakdown", "criteria", "lambdas")


def assert_blind(public: dict) -> None:
    """Fail the build if the public bundle carries anything that gives an answer away.

    `terminate_source` is the subtle one: `"inferred"` implies the node had children, which
    implies the search continued, which *is* the stop answer. It is withheld even though it
    looks like harmless provenance.
    """
    for d in public["decisions"]:
        extra = set(d) - PUBLIC_KEYS
        if extra:
            raise SystemExit(f"build aborted: {d['id']} would publish {sorted(extra)}")
        for c in d["candidates"]:
            if set(c) != {"slot", "action", "question"}:
                raise SystemExit(f"build aborted: {d['id']} candidate keys {sorted(c)}")
    blob = json.dumps(public["decisions"])
    for token in FORBIDDEN:
        if f'"{token}"' in blob:
            raise SystemExit(f"build aborted: the public bundle contains a {token!r} key")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=REPO / "runs" / "20260728-014100_d2i_bench",
                   help="the sampling pool: a run dir, or any folder of them. Every run under "
                        "it that has both repository.json and pruned_questions is sampled from "
                        "(default: runs/20260728-014100_d2i_bench)")
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
    p.add_argument("--stop-share", type=float, nargs=2, metavar=("LO", "HI"), default=(0.30, 0.70),
                   help="bounds on the early-stopped share of the sample "
                        "(default: 0.30 0.70, i.e. between 30/70 and 70/30)")
    p.add_argument("--strict-share", dest="strict", action="store_true",
                   help="shorten the sample until the --stop-share band genuinely holds. By "
                        "default the band is aimed for but not enforced: every early-stopped "
                        "node available is taken, -n is kept, and a shortfall is reported")
    p.add_argument("--no-feedback", dest="feedback", action="store_false",
                   help="hold the model's answer back until the whole survey is done. The "
                        "picks are then independent samples, which is the cleaner measurement; "
                        "by default each decision is revealed as soon as it is answered")
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

    lo, hi = sorted(args.stop_share)
    if not 0.0 <= lo <= hi <= 1.0:
        raise SystemExit("--stop-share bounds must satisfy 0 <= LO <= HI <= 1")
    decisions, notes, samples, sources = sample_decisions(
        run_dirs, args.n, random.Random(args.seed), args.min_candidates,
        args.drop_unanswerable, (lo, hi), args.strict,
    )
    if not decisions:
        raise SystemExit(
            f"no decision points under {args.runs} — a run needs both repository.json and "
            "report.json's pruned_questions to reconstruct one")

    stopped = sum(1 for d in decisions if d.get("d2i_terminate") is True)
    with_cands = sum(1 for d in decisions if d["candidates"])
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Per-decision feedback needs an answer key in the browser, so it cannot outlive
    # --no-reveal. Silently showing nothing would be worse than saying so.
    feedback = args.feedback and args.reveal
    if args.feedback and not args.reveal:
        print("--no-reveal publishes no answer key, so per-decision feedback is off too")

    bundle = {
        "meta": {
            "title": args.title,
            "built_at": built_at,
            "seed": args.seed,
            "n_requested": args.n,
            "n_decisions": len(decisions),
            "n_runs": len(sorted({d["run_dir"] for d in decisions})),
            "n_early_stopped": stopped,
            "stop_share": round(stopped / len(decisions), 4) if decisions else 0,
            "stop_share_band": [lo, hi],
            "n_with_candidates": with_cands,
            "min_candidates": args.min_candidates,
            "drop_unanswerable": args.drop_unanswerable,
            "reveal": args.reveal,
            "feedback": feedback,
            # How many stop calls the judge actually ruled on. The completion page says so
            # out loud: the rest are inferred from the search having carried on, which is a
            # weaker claim and must not be pooled into the headline.
            "n_stop_judged": sum(1 for d in decisions if d.get("terminate_source") == "judged"),
            "n_stop_inferred": sum(1 for d in decisions
                                   if d.get("terminate_source") in ("inferred", "terminated")),
            # The metric surface, shipped as data so the page renders whatever this build
            # defined. `version` lets a page served against a stale bundle be detected.
            "metrics": {"version": hte.METRICS_VERSION,
                        "table": [dict(m) for m in hte.METRIC_TABLE],
                        "terms": [short for _, short in hte.TERMS],
                        "within_noise_d": hte.WITHIN_NOISE_D},
        },
        # Keyed by run name, not repeated per decision: several decisions come from the same
        # run, and the page looks the sample up by `decision.run`.
        "samples": {d["run"]: samples[d["run"]] for d in decisions if d["run"] in samples},
        # Where each run came from. Copied into the download so a returned response can be
        # read without `runs/` to hand.
        "runs": {d["run"]: sources[d["run"]] for d in decisions if d["run"] in sources},
        "decisions": [public_decision(d) for d in decisions],
    }
    assert_blind(bundle)

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
        # report() reads this to decide whether to caveat the rates. With feedback on, a
        # respondent has seen the model's ranking before their later picks, so those picks
        # are not independent samples and the caveat has to be printed — recording it as
        # False here would quietly overstate the result.
        "feedback": feedback,
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
          f"  ({stopped} early-stopped = {stopped / max(1, len(decisions)):.0%}, "
          f"{with_cands} with a candidate pool), seed {args.seed}")
    print(f"  pool    {args.runs}  ({len(run_dirs)} run dir(s) found)")
    print(f"  public  {data_dir / 'survey.json'}  "
          f"({(data_dir / 'survey.json').stat().st_size / 1024:.0f} KB)")
    if args.reveal:
        print(f"  key     {truth_path}  ({truth_path.stat().st_size / 1024:.0f} KB)"
              + ("   — read by the page from the start, to reveal each answer"
                 if feedback else
                 "   — published; a determined respondent can read it early"))
    else:
        print("  key     not written (--no-reveal)")
    print(f"  feedback {'after every decision — report() will caveat the rates as dependent'
                       if feedback else 'held back until the end — picks stay independent'}")
    print(f"  private {priv_dir / 'decisions.json'}   — keep this out of the deploy")


if __name__ == "__main__":
    main()
