"""G-Eval-style rubric-based scorer for insight trajectories (paper §6.1.4).

Given the analytical goal, a dataset profile, and an ordered insight trajectory
(each step's statement AND its supporting evidence, matching the paper's
I = (s, e) -- section 3.1), GPT-4o independently scores four criteria on a 1-5
scale in a single Structured Outputs call, then the exploration quality score is
computed deterministically in Python (never trusted from the model) as the
macro-average of the four:

    exploration_quality_score = (goal_relevance + information_gain
                                  + interestingness + trustworthiness) / 4

This is NOT a reproduction of the original G-Eval paper's token-logprob soft
scoring (Liu et al., 2023), nor of insight-bench's own compute_g_eval (which
compares two texts pairwise using that logprob technique, since it scores
against a labeled ground-truth insight). There is no ground truth here --
D2I-Bench ships goals only -- so report this in the paper as a "G-Eval-style
rubric-based evaluator using GPT-4o", not as G-Eval itself.

Trustworthiness is scoped to EVIDENCE-GROUNDED trustworthiness: whether a
conclusion is supported by the evidence recorded alongside it in the trajectory
and consistent with the dataset profile -- NOT independent re-verification
against the raw dataset. That's what section 6.6's validity/execution-accuracy
experiments already cover; keeping this scorer free of the raw dataset avoids
the two measuring the same thing twice.

Input
-----
  goal            : str  -- the analytical goal (data/d2i_bench/goals.json)
  dataset_profile : dict -- the dataset's bench_profile.json entry
  trajectory      : list[{"description": str, "depth": int, "evidence": str}]
                    -- one insight trajectory, ordered by depth; "evidence" is
                    whatever supporting computation the source system recorded
                    for that step (see the load_* adapters below -- richness
                    varies by system, see each adapter's docstring)

Output
------
  {
    "goal_relevance":   {"score": int (1-5), "reason": str},
    "information_gain": {"score": int (1-5), "reason": str},
    "interestingness":  {"score": int (1-5), "reason": str},
    "trustworthiness":  {"score": int (1-5), "reason": str},
    "exploration_quality_score": float,   # macro-average of the four scores
  }

CLI
---
  python evaluation/geval_trajectory.py --system d2i --run-dir runs/<stamp>_d2i_bench/<dataset>-<level>
  python evaluation/geval_trajectory.py --system quis --run-dir baselines/quis/<stamp>/<dataset>
  python evaluation/geval_trajectory.py --system agent_poirot --run-dir baselines/agent_poirot/<stamp>/<dataset>
Scores every trajectory in the run and writes <run-dir>/geval_trajectory.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-4o-2024-08-06"
RUBRIC_VERSION = "v2"  # bump whenever SYSTEM_PROMPT or RUBRIC changes meaningfully
TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Prompt -- fixed evaluation steps and rubric, not regenerated per call, so
# scoring stays reproducible across systems, datasets, and repeated runs.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert evaluator of automated exploratory data analysis.

Your task is to evaluate an ordered insight trajectory produced by a data-analysis
agent. The trajectory should progressively contribute to the user's analytical goal
through relevant, non-redundant, interesting, and evidence-supported observations.

Evaluate only the provided trajectory. Do not introduce external facts, perform
unsupported calculations, or assume that a claim is correct merely because it sounds
plausible.

Use the dataset profile only to understand the dataset schema, column semantics,
and dataset-level composition. Use the evidence attached to each trajectory
step -- not the raw dataset, which you do not have -- to assess whether its
conclusions are supported.

Each trajectory step may also carry system-generated metadata (e.g. an internal
confidence, ranking, or pattern score). Metadata reflects what the source system
itself believes about the step -- it is not evidence that the factual claim is
correct, and must never be used to support Trustworthiness.

Evaluate the trajectory independently on four criteria:
1. Goal Relevance
2. Information Gain
3. Interestingness
4. Trustworthiness

Assign an integer score from 1 to 5 for each criterion according to the supplied
rubrics. Assess the trajectory as a whole, while considering the ordered progression
between successive observations.

Important evaluation rules:
- Do not reward trajectory length by itself.
- Repetition, paraphrasing, and increasingly narrow slices without new understanding
  do not constitute information gain.
- A later observation may provide information gain by introducing new evidence,
  refining, explaining, localizing, contextualizing, or validating an earlier finding.
- A trajectory may be trustworthy but uninteresting, or interesting but insufficiently
  supported. Score each criterion independently.
- Do not penalize a valid trajectory merely because other possible analyses were not
  explored. Evaluate the quality of this trajectory, not the completeness of the
  entire analysis.
- Claims that exceed the supplied evidence, confuse association with causation, or
  use unjustified language must reduce the Trustworthiness score.
- Trustworthiness has two parts: whether the claim is CONSISTENT with the evidence
  attached to it, and whether that evidence is SUFFICIENT to verify the claim. Do
  not infer that a claim is false solely because its evidence is missing or thin --
  but missing or non-diagnostic evidence caps how high Trustworthiness can score,
  because the claim cannot be verified from what the trajectory provides.
- Return only the requested structured output.
""".strip()

RUBRIC = """
Follow these evaluation steps:

1. Understand the analytical goal and identify what a useful analysis should address.
2. Review the dataset profile to understand schema and value domains. Do not treat
   profile statistics as evidence for claims that require a separate computation.
3. Read the trajectory in order. For each observation, identify its claim, the
   evidence supporting it, how it relates to the preceding observations, and how it
   contributes to the analytical goal.
4. Evaluate Goal Relevance.
5. Evaluate Information Gain by comparing every step with all preceding steps.
6. Evaluate Interestingness.
7. Evaluate Trustworthiness using the evidence attached to each step.
8. Assign an integer score from 1 to 5 for each criterion using the rubrics below.

Goal Relevance: how well the trajectory contributes to answering the original
analysis objective.
1 = Largely unrelated to the goal or dominated by tangential exploration.
2 = Only a small portion meaningfully contributes to the goal.
3 = Generally relevant, with some peripheral, weakly connected, or drifting
    observations.
4 = Nearly all observations clearly contribute, with only minor drift.
5 = Consistently and directly focused; the trajectory substantially advances the goal.

Information Gain: how much each successive observation adds beyond what the
trajectory already established.
1 = Redundant, paraphrased, or no meaningful progression between steps.
2 = Limited new information; most later steps are repetitive or trivial refinements.
3 = Several steps add useful new information, but progression is uneven or partly
    redundant.
4 = Most successive observations meaningfully refine, explain, validate, localize, or
    extend earlier findings, with little redundancy.
5 = Strong cumulative progression: nearly every step contributes substantial new
    evidence or understanding beyond what was already established.

A single-step trajectory has no successive observation to compare against, so score
it on whether that one observation itself provides meaningful analytical
understanding beyond the dataset profile -- not on progression:
1 = the observation is trivial or merely restates profile information.
2 = it provides limited new understanding.
3 = it provides a meaningful standalone finding.
Scores of 4-5 require demonstrated progression across multiple observations and are
not available to single-step trajectories.

Interestingness: how non-trivial, surprising, or practically meaningful the
observations are.
1 = Trivial, obvious, or analytically uninformative.
2 = Minor useful facts but little that would attract an analyst's attention.
3 = Some non-trivial or practically useful findings, mixed with routine observations.
4 = Clearly meaningful, non-obvious, or decision-relevant findings.
5 = Highly insightful, surprising, or practically important findings that
    substantially deepen understanding of the analytical problem.

Trustworthiness: whether each claim is consistent with its recorded evidence AND
whether that evidence is sufficient (diagnostic, not merely system metadata) to
verify the claim. A claim is not "supported" just because a number or score sits
next to it -- the evidence must actually establish the claim.
1 = Main claims contradict the evidence, or are almost entirely unsupported.
2 = Multiple important claims lack diagnostic evidence, rely only on system
    metadata (e.g. an internal score), or overstate what the evidence establishes.
3 = Claims are plausible and broadly consistent with their evidence, but the
    evidence is incomplete or insufficient to fully verify some material claims.
4 = Most material claims are supported by sufficiently diagnostic evidence; only
    minor details are unverifiable.
5 = Every material claim is supported by sufficiently detailed, diagnostic
    evidence; no material overclaim or unverifiable claim is present.
""".strip()

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        crit: {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        }
        for crit in ("goal_relevance", "information_gain", "interestingness", "trustworthiness")
    },
    "required": ["goal_relevance", "information_gain", "interestingness", "trustworthiness"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_profile(dataset_profile: dict) -> str:
    """Compact text rendering of a bench_profile.json entry -- a full column dump
    would blow up the prompt for wide datasets, so only name/role per column, plus
    the dataset-level composition/cardinality stats bench_profile.json actually
    carries. There are no per-column value domains or basic statistics (min/max/
    examples) in this profile -- don't claim the judge can see those."""
    lines = [f"  - {c['name']} ({c.get('role', '?')})" for c in dataset_profile.get("columns", [])]
    header = (
        f"domain={dataset_profile.get('domain', '?')}, sector={dataset_profile.get('sector', '?')}\n"
        f"{dataset_profile.get('rows', '?')} rows, {len(dataset_profile.get('columns', []))} columns "
        f"(numeric={dataset_profile.get('n_numeric', '?')}, "
        f"categorical={dataset_profile.get('n_categorical', '?')}, "
        f"temporal={dataset_profile.get('n_temporal', '?')}, "
        f"text={dataset_profile.get('n_text', '?')}, "
        f"id={dataset_profile.get('n_id', '?')}, "
        f"boolean={dataset_profile.get('n_boolean', '?')})\n"
        f"missing_frac={dataset_profile.get('missing_frac', '?')}, "
        f"max_cat_cardinality={dataset_profile.get('max_cat_cardinality', '?')}, "
        f"dirty_numeric={dataset_profile.get('dirty_numeric', '?')}, "
        f"date_span={dataset_profile.get('date_span', '?')}"
    )
    return header + "\nColumns:\n" + "\n".join(lines)


def _format_trajectory(trajectory: list[dict]) -> str:
    lines = []
    for i, node in enumerate(sorted(trajectory, key=lambda n: n["depth"])):
        tag = "[base]" if i == 0 else f"[  +{i}]"
        lines.append(f"{tag} Statement: {node['description']}")
        evidence = (node.get("evidence") or "").strip()
        lines.append(f"       Evidence: {evidence or '(none recorded)'}")
        metadata = node.get("metadata") or {}
        if metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            lines.append(f"       System metadata (NOT evidence): {meta_str}")
    return "\n".join(lines)


def _build_user_prompt(goal: str, dataset_profile: dict, trajectory: list[dict]) -> str:
    return f"""\
Evaluate the following insight trajectory.

<analytical_goal>
{goal}
</analytical_goal>

<dataset_profile>
{_format_profile(dataset_profile)}
</dataset_profile>

<trajectory>
{_format_trajectory(trajectory)}
</trajectory>

{RUBRIC}
"""


# ---------------------------------------------------------------------------
# The G-Eval call -- one Structured Outputs call scores all four criteria
# together, so all four judgments come from the same single reading of the
# trajectory (cheaper and more internally consistent than four separate calls
# re-reading it under four different framings).
# ---------------------------------------------------------------------------

def score_trajectory(
    goal: str,
    dataset_profile: dict,
    trajectory: list[dict],
    model: str = MODEL,
    client: Optional[OpenAI] = None,
) -> dict:
    client = client or OpenAI()
    response = client.responses.create(
        model=model,
        temperature=TEMPERATURE,
        store=False,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(goal, dataset_profile, trajectory)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "trajectory_geval",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )
    result = json.loads(response.output_text)
    # Deterministic macro-average -- never trust the model's own arithmetic.
    result["exploration_quality_score"] = sum(
        result[k]["score"] for k in
        ("goal_relevance", "information_gain", "interestingness", "trustworthiness")
    ) / 4
    return result


# ---------------------------------------------------------------------------
# Per-system loaders: pull (goal, [trajectories]) out of each system's own run
# output into the generic [{"description", "depth", "evidence"}, ...] shape.
# Evidence richness genuinely differs by system -- each docstring says what's
# actually available, rather than fabricating a uniform depth that isn't there.
# ---------------------------------------------------------------------------

def load_d2i_trajectories(run_dir: Path) -> tuple[str, list[list[dict]]]:
    """D2I's repository.json: a flat list of records carrying trajectory_id/depth/
    parent_id/label/insight.analysis/evidence.statistic. Evidence = the agent's own
    analysis paragraph plus the computed statistic/p-value/effect-size dict -- the
    richest of the three systems, matching the paper's I=(s,e) most directly."""
    repo = json.loads((run_dir / "repository.json").read_text())
    goal = repo["goal"]
    by_traj: dict[str, list[dict]] = {}
    for rec in repo["records"]:
        analysis = (rec.get("insight") or {}).get("analysis", "")
        statistic = (rec.get("evidence") or {}).get("statistic", {})
        evidence = analysis
        if statistic:
            evidence = f"{analysis}\nComputed statistic: {statistic}"
        by_traj.setdefault(rec["trajectory_id"], []).append(
            {"description": rec["label"], "depth": rec["depth"], "evidence": evidence}
        )
    trajectories = [sorted(nodes, key=lambda n: n["depth"]) for nodes in by_traj.values()]
    return goal, trajectories


def load_quis_trajectories(run_dir: Path) -> tuple[str, list[list[dict]]]:
    """QUIS's report.json: {"trajectories": [{"nodes": [{"description","score","pattern"}]}]}.
    QUIS's own output carries no claim-level execution evidence (no statistic dict,
    no underlying values) -- only the ISGen score and pattern name, which reflect the
    system's own confidence/ranking, not a computation supporting the claim. Those go
    into metadata (explicitly excluded from Trustworthiness), not evidence -- passing
    them as evidence would let a self-reported score stand in for actual support."""
    report = json.loads((run_dir / "report.json").read_text())
    goal = report.get("goal", "")
    trajectories = []
    for traj in report.get("trajectories", []):
        nodes = []
        for i, n in enumerate(traj["nodes"]):
            nodes.append({
                "description": n["description"],
                "depth": i,
                "evidence": "No claim-level execution evidence was retained by the system.",
                "metadata": {"pattern": n.get("pattern", "?"), "system_score": n.get("score", "?")},
            })
        trajectories.append(nodes)
    return goal, trajectories


def load_agent_poirot_trajectories(run_dir: Path) -> tuple[str, list[list[dict]]]:
    """AgentPoirot's report.json trajectories carry question/insight/figure; the
    justification behind each insight lives in the sibling insights_history.json
    (same run dir), and the chart's underlying values in question_N/stat.json next
    to the figure. Both are pulled in as evidence when present."""
    report = json.loads((run_dir / "report.json").read_text())
    goal = report.get("goal", "")

    justification_by_question = {}
    history_path = run_dir / "insights_history.json"
    if history_path.is_file():
        for item in json.loads(history_path.read_text()):
            justification_by_question[item.get("question", "")] = item.get("justification", "")

    trajectories = []
    for traj in report.get("trajectories", []):
        nodes = []
        for i, n in enumerate(traj["nodes"]):
            parts = []
            justification = justification_by_question.get(n.get("question", ""))
            if justification:
                parts.append(justification)
            figure = n.get("figure")
            if figure:
                stat_path = Path(figure).with_name("stat.json")
                if stat_path.is_file():
                    stat = json.loads(stat_path.read_text())
                    parts.append(f"Chart data ({stat.get('name', '')}): {stat.get('value')}")
            nodes.append({
                "description": n["insight"], "depth": i,
                "evidence": " ".join(parts) if parts else "",
            })
        trajectories.append(nodes)
    return goal, trajectories


LOADERS = {
    "d2i": load_d2i_trajectories,
    "quis": load_quis_trajectories,
    "agent_poirot": load_agent_poirot_trajectories,
}


def _prompt_hash() -> str:
    """Hash of the fixed system prompt + rubric, so results written with a changed
    prompt/rubric are never silently mixed with results from an earlier version."""
    return hashlib.sha256((SYSTEM_PROMPT + RUBRIC).encode("utf-8")).hexdigest()[:12]


def load_dataset_profile(dataset_name: str, profile_path: Path) -> dict:
    profile = json.loads(profile_path.read_text())
    for d in profile["datasets"]:
        if d["name"] == dataset_name:
            return d
    raise ValueError(f"no profile entry for dataset {dataset_name!r} in {profile_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=sorted(LOADERS), required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="one dataset's run dir")
    p.add_argument("--dataset", type=str, default=None,
                    help="dataset name for the profile lookup (default: run-dir's basename up to the first '-')")
    p.add_argument("--profile", type=Path, default=ROOT / "data" / "d2i_bench" / "bench_profile.json")
    p.add_argument("--model", type=str, default=MODEL)
    p.add_argument("--goal", type=str, default=None,
                    help="override the goal read from the run's own output -- needed for runs "
                         "predating goal persistence (older QUIS runs never wrote it)")
    p.add_argument("--out", type=Path, default=None, help="default: <run-dir>/geval_trajectory.json")
    args = p.parse_args()

    dataset_name = args.dataset or args.run_dir.name.rsplit("-", 1)[0]
    goal, trajectories = LOADERS[args.system](args.run_dir)
    goal = args.goal or goal
    if not goal:
        raise SystemExit(
            f"no goal found in {args.run_dir} for system {args.system!r}; pass --goal explicitly"
        )
    dataset_profile = load_dataset_profile(dataset_name, args.profile)

    client = OpenAI()
    results = []
    for i, traj in enumerate(trajectories):
        print(f"scoring trajectory {i + 1}/{len(trajectories)} ({len(traj)} nodes)...", flush=True)
        scores = score_trajectory(goal, dataset_profile, traj, model=args.model, client=client)
        results.append({"trajectory_index": i, "depth": len(traj), **scores})

    keys = ("goal_relevance", "information_gain", "interestingness", "trustworthiness")
    summary = {k: sum(r[k]["score"] for r in results) / len(results) for k in keys} if results else {}
    if results:
        summary["exploration_quality_score"] = sum(r["exploration_quality_score"] for r in results) / len(results)

    out = {
        "system": args.system,
        "dataset": dataset_name,
        "goal": goal,
        "model": args.model,
        "temperature": TEMPERATURE,
        "rubric_version": RUBRIC_VERSION,
        "prompt_hash": _prompt_hash(),
        "n_trajectories": len(results),
        "summary": summary,
        "trajectories": results,
    }
    out_path = args.out or (args.run_dir / "geval_trajectory.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
