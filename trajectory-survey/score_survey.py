#!/usr/bin/env python3
"""Turn returned rating responses into the study's report.

    python3 trajectory-survey/score_survey.py responses/
    python3 trajectory-survey/score_survey.py responses/ --per-respondent --export -o results/
    python3 trajectory-survey/score_survey.py responses/ --check-parity

What it prints, in the order it prints it:

  agreement          per criterion and pooled, with the judge's agreement with ITSELF
                     across its repeats in the last column — the ceiling every other
                     number should be read against
  by system          each system's mean score from the human and from the judge, and
                     whether the two RANK the systems the same way. This is the finding a
                     paper needs: a judge that disagrees with people cell by cell but
                     orders the systems identically still supports the same conclusion
  by dataset         the same agreement, split by dataset
  human vs human     with two or more respondents, how often THEY agree with each other —
                     the second ceiling, and usually the more honest one
  splits             the trajectories respondents disagreed about most

The estimators live in `agreement.py`, which is the twin of the browser's `agreement.js`;
nothing here reimplements one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agreement as A  # noqa: E402


# --------------------------------------------------------------------------- loading

def load_responses(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                R = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"  skip {f.name}: not JSON ({e})")
                continue
            if not isinstance(R, dict) or not isinstance(R.get("answers"), list):
                print(f"  skip {f.name}: no answers")
                continue
            R["_file"] = str(f)
            out.append(R)
    return out


def cells_of(R: dict, truth: dict, mismatched: list | None = None) -> list[dict]:
    """One cell per (trajectory, criterion) this respondent rated.

    An answer names its trajectory twice: by `item`, which is the build's id for a slot,
    and by `content`, the hash of the judge prompt that was on the screen. The id survives
    a rewrite of `runs/` unchanged and the content hash does not, so where both are present
    the second is checked — a rating is only scored against a judge score for the same
    trajectory, not merely for the same coordinates.
    """
    out = []
    for a in R["answers"]:
        if a.get("skipped"):
            continue
        t = truth["items"].get(a["item"])
        if not t:
            continue
        if a.get("content") and t.get("content") and a["content"] != t["content"]:
            if mismatched is not None:
                mismatched.append((R.get("respondent", {}).get("name") or R.get("id", "?"),
                                   a["item"], t.get("dataset", "?")))
            continue
        for crit, value in (a.get("scores") or {}).items():
            block = t["scores"].get(crit)
            if not block:
                continue
            out.append({
                "respondent": R.get("respondent", {}).get("name") or R.get("id", "?"),
                "item": a["item"], "dataset": t["dataset"], "system": t["system"],
                "criterion": crit, "human": value, "judge": block["score"],
                "judge_scores": block.get("scores") or [],
                "assigned": bool(block.get("assigned")),
            })
    return out


# --------------------------------------------------------------------------- extras

def by_group(cells: list[dict], key: str, order: list[str], reps: int) -> list[tuple[str, dict]]:
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(c[key], []).append(c)
    return [(g, A.compute(groups[g], order=order, reps=reps)) for g in sorted(groups)]


def system_table(cells: list[dict], criteria: list[str]) -> dict:
    """Each system's mean score from each side, and whether they order the systems alike.

    Cell-by-cell agreement and system ORDERING are different questions, and a judge can
    fail the first while passing the second. Only the second is what a benchmark table
    actually rests on, so it is computed separately rather than inferred from a
    correlation.
    """
    usable = [c for c in cells if not c["assigned"]]
    systems = sorted({c["system"] for c in usable})
    out = {"systems": systems, "criteria": {}, "overall": {}}
    for crit in criteria + ["__all__"]:
        rows = usable if crit == "__all__" else [c for c in usable if c["criterion"] == crit]
        if not rows:
            continue
        human = {s: A.mean([c["human"] for c in rows if c["system"] == s]) for s in systems}
        judge = {s: A.mean([c["judge"] for c in rows if c["system"] == s]) for s in systems}
        human = {s: v for s, v in human.items() if not math.isnan(v)}
        judge = {s: v for s, v in judge.items() if not math.isnan(v)}
        shared = sorted(set(human) & set(judge))
        rank_h = sorted(shared, key=lambda s: -human[s])
        rank_m = sorted(shared, key=lambda s: -judge[s])
        # Two systems on the same mean have no order between them, so `sorted` invents one
        # from the system name and "same order?" would answer a question the data does not
        # ask. Reported as a tie instead of as a disagreement.
        tied = (len({round(human[s], 10) for s in shared}) < len(shared)
                or len({round(judge[s], 10) for s in shared}) < len(shared))
        block = {
            "human": human, "judge": judge,
            "human_order": rank_h, "judge_order": rank_m, "tied": tied,
            "same_order": None if tied else rank_h == rank_m,
            "spearman": A.spearman([human[s] for s in shared], [judge[s] for s in shared]),
            "n": len(rows),
        }
        if crit == "__all__":
            out["overall"] = block
        else:
            out["criteria"][crit] = block
    return out


def human_vs_human(per_respondent: list[tuple[str, list[dict]]], reps: int) -> list[dict]:
    """Every pair of respondents, over the cells they both rated. This is the ceiling that
    matters most: agreement with a judge should be read against how often two people agree
    with each other, not against 100%."""
    out = []
    for i in range(len(per_respondent)):
        for j in range(i + 1, len(per_respondent)):
            (na, ca), (nb, cb) = per_respondent[i], per_respondent[j]
            ib = {(c["item"], c["criterion"]): c for c in cb if not c["assigned"]}
            paired = [{"item": c["item"], "criterion": c["criterion"], "human": c["human"],
                       "judge": ib[(c["item"], c["criterion"])]["human"],
                       "judge_scores": [], "assigned": False}
                      for c in ca if not c["assigned"] and (c["item"], c["criterion"]) in ib]
            if not paired:
                continue
            out.append({"a": na, "b": nb, "n": len(paired), **A.stats(paired)})
    return out


def splits(cells: list[dict]) -> list[dict]:
    """Where respondents disagreed with each other, worst first — the trajectories worth
    reading before trusting any single number about them."""
    groups: dict[tuple, list[dict]] = {}
    for c in cells:
        groups.setdefault((c["item"], c["criterion"]), []).append(c)
    out = []
    for (item, crit), rows in groups.items():
        if len(rows) < 2:
            continue
        vals = [r["human"] for r in rows]
        spread = max(vals) - min(vals)
        if spread < 2:
            continue
        out.append({"item": item, "criterion": crit, "system": rows[0]["system"],
                    "dataset": rows[0]["dataset"], "spread": spread, "human": vals,
                    "judge": rows[0]["judge"]})
    return sorted(out, key=lambda r: -r["spread"])


# --------------------------------------------------------------------------- printing

def hr(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


def print_systems(tab: dict, criteria: list[str]) -> None:
    if not tab.get("overall"):
        return
    systems = tab["systems"]
    head = ["criterion"] + [f"{s} (you)" for s in systems] + [f"{s} (judge)" for s in systems] \
        + ["same order?", "rho"]
    rows = []
    for crit in criteria + ["__all__"]:
        b = tab["overall"] if crit == "__all__" else tab["criteria"].get(crit)
        if not b:
            continue
        rows.append([A.label(crit) if crit != "__all__" else "ALL CRITERIA"]
                    + [A.fmt(b["human"].get(s), 2) for s in systems]
                    + [A.fmt(b["judge"].get(s), 2) for s in systems]
                    + ["tie" if b["same_order"] is None
                       else ("yes" if b["same_order"] else "NO"),
                       A.fmt(b["spearman"], 2)])
    w = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "  ".join(v.ljust(w[i]) if i == 0 else v.rjust(w[i]) for i, v in enumerate(r))
    print(line(head))
    print("  ".join("-" * n for n in w))
    for r in rows:
        print(line(r))
    b = tab["overall"]
    print(f"\nhuman ranking: {' > '.join(b['human_order'])}")
    print(f"judge ranking: {' > '.join(b['judge_order'])}"
          + ("   — two systems tie, so there is no order to compare" if b["same_order"] is None
             else "   — the same order" if b["same_order"] else "   — A DIFFERENT ORDER"))


# --------------------------------------------------------------------------- export

def export(out_dir: Path, pooled: dict, cells: list[dict], systems: dict,
           criteria: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.csv").write_text(A.to_csv(pooled) + "\n", encoding="utf-8")

    with (out_dir / "cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["respondent", "item", "dataset", "system", "criterion", "human",
                    "judge_mean", "judge_repeats", "assigned"])
        for c in cells:
            w.writerow([c["respondent"], c["item"], c["dataset"], c["system"], c["criterion"],
                        c["human"], c["judge"],
                        " ".join(str(x) for x in c["judge_scores"]), int(c["assigned"])])

    with (out_dir / "systems.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "system", "human_mean", "judge_mean"])
        for crit in criteria + ["__all__"]:
            b = systems["overall"] if crit == "__all__" else systems["criteria"].get(crit)
            if not b:
                continue
            for s in systems["systems"]:
                w.writerow([crit, s, b["human"].get(s, ""), b["judge"].get(s, "")])

    # booktabs, pdfLaTeX-safe: the terminal's Greek and ± become math mode.
    rows = pooled["criteria"] + [pooled["pooled"]]
    tex = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
           "Criterion & $n$ & Human & Judge & Exact & $\\pm 1$ & MAE & $r$ & QWK \\\\",
           "\\midrule"]
    for r in rows:
        if r is pooled["pooled"]:
            tex.append("\\midrule")
        tex.append(" & ".join([
            A.label(r["criterion"]), str(r["n"]), A.fmt(r["human_mean"], 2),
            A.fmt(r["judge_mean"], 2), A.pct(r["exact"]).replace("%", "\\%"),
            A.pct(r["within1"]).replace("%", "\\%"), A.fmt(r["mae"], 2),
            A.fmt(r["pearson"], 2), A.fmt(r["qwk"], 2)]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "metrics.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir}/metrics.csv, cells.csv, systems.csv, metrics.tex")


# --------------------------------------------------------------------------- parity

def check_parity(R: dict, mine: dict) -> None:
    """The browser's own table against a recomputation here. Near-tautological by
    construction, so a failure means a page was served against a stale truth.json rather
    than a formula bug."""
    theirs = (R.get("results") or {}).get("summary")
    if not theirs:
        print("    no browser-computed table in this response — nothing to compare")
        return
    mine_rows = {r["criterion"]: r for r in [mine["pooled"]] + mine["criteria"]}
    bad = []
    for row in theirs:
        m = mine_rows.get(row["criterion"])
        if not m:
            bad.append(f"{row['criterion']}: not recomputed here")
            continue
        for k in ("n", "human_mean", "judge_mean", "exact", "within1", "mae", "pearson",
                  "qwk", "alpha"):
            a, b = row.get(k), m.get(k)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 1e-9:
                bad.append(f"{row['criterion']}.{k}: browser {a} vs here {b}")
    print("    parity: " + ("ok" if not bad else f"{len(bad)} difference(s)"))
    for b in bad[:8]:
        print("      " + b)


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("responses", nargs="+", type=Path,
                   help="response .json files, or directories of them")
    p.add_argument("--truth", type=Path, default=HERE / "data" / "truth.json",
                   help="the answer key written by build_survey.py")
    p.add_argument("--per-respondent", action="store_true",
                   help="print each respondent's own table as well as the pooled one")
    p.add_argument("--check-parity", action="store_true",
                   help="diff each response's browser-computed table against a "
                        "recomputation here")
    p.add_argument("--export", action="store_true", help="write csv/tex alongside the report")
    p.add_argument("-o", "--out", type=Path, default=HERE / "results")
    p.add_argument("--bootstrap", type=int, default=A.BOOTSTRAP,
                   help="resamples for the 95%% intervals (0 to skip)")
    args = p.parse_args()

    if not args.truth.is_file():
        raise SystemExit(f"score_survey: no answer key at {args.truth}\n"
                         "  build with reveal on, or point --truth at the private bundle")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if "items" not in truth and "truth" in truth:          # private/build.json
        truth = {"meta": truth["meta"], "items": truth["truth"]}
    criteria = list(truth["meta"]["criteria"])

    responses = load_responses(args.responses)
    if not responses:
        raise SystemExit("score_survey: no usable responses")

    built = truth["meta"]["built_at"]
    keep = []
    for R in responses:
        theirs = (R.get("survey") or {}).get("built_at")
        if theirs and theirs != built:
            print(f"  skip {Path(R['_file']).name}: built {theirs}, key is {built} — a "
                  "rebuild moved the items, so these are answers to other questions")
            continue
        keep.append(R)
    responses = keep
    if not responses:
        raise SystemExit("score_survey: every response came from a different build")

    mismatched: list = []
    per = [(R.get("respondent", {}).get("name") or R.get("id", "?"),
            cells_of(R, truth, mismatched)) for R in responses]
    if mismatched:
        print(f"\n** {len(mismatched)} rating(s) dropped: the trajectory rated is not the "
              "one the key scores **")
        for who, item, ds in mismatched[:8]:
            print(f"   {who}: item {item} ({ds})")
        print("   The run directory was rewritten between the rating and the scoring. "
              "Score those\n   against a scoring run of the trajectories people actually "
              "saw — build_survey.py\n   --attach-scores checks the same hashes and will "
              "say which.\n")
    pooled_cells = [c for _, cs in per for c in cs]
    if not pooled_cells:
        raise SystemExit("score_survey: no ratings in these responses")

    # Skips are counted and printed, never inferred from a missing row: "this respondent
    # could not judge 16 of 18" is a finding about the instrument, and a report that showed
    # only the 2 they did rate would read as a much smaller study rather than a refused one.
    skips = [(R.get("respondent", {}).get("name") or R.get("id", "?"),
              sum(1 for a in R["answers"] if a.get("skipped")),
              len(R["answers"])) for R in responses]
    n_skipped = sum(s for _, s, _ in skips)
    n_answers = sum(t for _, _, t in skips)

    print(f"{len(responses)} response(s), {len(pooled_cells)} ratings, "
          f"{len({c['item'] for c in pooled_cells})} distinct trajectories")
    if n_skipped:
        print(f"{n_skipped} of {n_answers} trajectory screens were marked "
              f"'cannot judge' and carry no ratings:")
        for name, s, t in skips:
            if s:
                print(f"    {name}: {s}/{t}"
                      + ("   ** most of this response **" if s > t / 2 else ""))
    print(f"judge: {truth['meta']['geval']['judge_model']}, rubric "
          f"{truth['meta']['geval']['rubric_version']}, "
          f"{truth['meta']['geval']['repeats']} repeats, build {built}")

    if args.per_respondent:
        for (name, cs), R in zip(per, responses):
            if not cs:
                continue
            rep = A.compute(cs, order=criteria, reps=args.bootstrap)
            print(hr(f"{name}"))
            print(A.to_text(rep))
            if args.check_parity:
                check_parity(R, rep)

    pooled = A.compute(pooled_cells, order=criteria, reps=args.bootstrap)
    print(hr("agreement — all respondents pooled"))
    print(A.to_text(pooled))
    if len(responses) > 1:
        print("\n(pooled over respondents: the same trajectory appears once per person, so "
              "these are\nnot independent observations of it — read the per-respondent "
              "tables beside this one.)")

    print(hr("by system"))
    systems = system_table(pooled_cells, criteria)
    print_systems(systems, criteria)

    print(hr("by dataset"))
    for name, rep in by_group(pooled_cells, "dataset", criteria, 0):
        p_ = rep["pooled"]
        print(f"  {name:24s} n={p_['n']:3d}  exact {A.pct(p_['exact']):>6s}  "
              f"±1 {A.pct(p_['within1']):>6s}  MAE {A.fmt(p_['mae'], 2)}  "
              f"r {A.fmt(p_['pearson'], 2)}")

    print(hr("agreement by system"))
    for name, rep in by_group(pooled_cells, "system", criteria, 0):
        p_ = rep["pooled"]
        print(f"  {name:24s} n={p_['n']:3d}  you {A.fmt(p_['human_mean'], 2)}  "
              f"judge {A.fmt(p_['judge_mean'], 2)}  exact {A.pct(p_['exact']):>6s}  "
              f"MAE {A.fmt(p_['mae'], 2)}")

    pairs = human_vs_human(per, args.bootstrap)
    if pairs:
        print(hr("human vs human — the ceiling to read the judge against"))
        for r in pairs:
            print(f"  {r['a']} vs {r['b']}: n={r['n']}, exact {A.pct(r['exact'])}, "
                  f"±1 {A.pct(r['within1'])}, MAE {A.fmt(r['mae'], 2)}, "
                  f"r {A.fmt(r['pearson'], 2)}")
        print(f"\n  the judge agrees with itself {A.pct(pooled['pooled']['judge_self']['exact'])} "
              f"of the time (MAE {A.fmt(pooled['pooled']['judge_self']['mae'], 2)})")
    elif len(responses) > 1:
        print(hr("human vs human"))
        print("  no two respondents rated the same (trajectory, criterion)")

    sp = splits(pooled_cells)
    if sp:
        print(hr("where respondents split"))
        for r in sp[:12]:
            print(f"  {r['dataset']:22s} {r['system']:12s} {A.label(r['criterion']):17s} "
                  f"humans {r['human']}  judge {A.fmt(r['judge'], 2)}")

    if pooled["n_excluded_assigned"]:
        print(hr("excluded"))
        print(f"  {pooled['n_excluded_assigned']} rating(s) sit on criteria the judge was "
              "never asked about\n  (QUIS trustworthiness is assigned 5 by policy). They "
              "are in cells.csv, and out of\n  every agreement number above.")
        vals = [c["human"] for c in pooled["assigned"]]
        if vals:
            print(f"  humans gave those cells a mean of {A.fmt(A.mean(vals), 2)} "
                  f"against the assigned 5.00")

    if args.export:
        export(args.out, pooled, pooled_cells, systems, criteria)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
