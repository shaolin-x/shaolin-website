# D2I human evaluation — the web survey  *(CLOSED)*

> **This study is closed.** `index.html` is now a notice pointing at
> [`../trajectory-survey/`](../trajectory-survey/), which measures human/model agreement on
> whole trajectories instead. The survey itself is preserved unchanged as
> `index_legacy.html` — unlinked, `noindex`, and with an empty `ENDPOINT`, so it collects
> nothing. Everything below describes that archived study; restore it by moving
> `index_legacy.html` back over `index.html`.

The static counterpart to `human_trajectory_web.py`. That tool is local: it holds `runs/`
open, samples a fresh session per judge, and enforces blinding inside the request handler.
A GitHub Pages site can do none of that, so the same work moves to build time and the
result is split into a public half and a private one.

```
survey/
  build_survey.py          freezes one sampled evaluation into the two halves
  index.html               the survey itself — static, no build step, no dependencies
  metrics.js               the scoring the closing screen and the download use — deployed
  score_survey.py          turns returned responses into the evaluation's own report
  data/
    survey.json            PUBLIC. blinded decisions — deployed
    truth.json             PUBLIC. the answer key + the precomputed scores — deployed
  private/
    decisions.json         the full decisions with the shuffle — gitignored, never deployed
  responses/               where you drop the files people send back — gitignored
  tests/
    synth_responses.py     oracle / adversarial / random respondents
    test_metrics.py        the scoring's correctness suite
    run_metrics.mjs        runs metrics.js from node, for the parity check
```

## Running it

```bash
python3 survey/build_survey.py            # rebuild after runs/ changes
python3 -m http.server 8000               # from the site root
open http://localhost:8000/survey/
```

Opening `index.html` straight off disk will not work — the browser blocks `fetch` on
`file://`. The page says so if you try.

Deploying is committing `survey/index.html`, `survey/metrics.js` and `survey/data/`.
**All three** — `index.html` without `metrics.js` still runs the survey but shows no results
at the end, and the failure is silent. `runs/` and `survey/private/` are gitignored and are
not needed at serve time.

## The scoring

Nine metrics, in two families. Selection asks whether the human chose the candidate D2I
scored highest, and — separately — *how far off* they were when they did not; continuation
asks the same two questions of the stop/continue call, against the judge's own continuation
utility and its 0.35 threshold.

| | metric | what it answers |
|---|---|---|
| **Selection** | `κ_sel` | agreement with D2I's top pick, corrected for chance *per pool*, so pools of different sizes and tie structures are comparable. 0 is a uniform random pick, 1 is D2I's choice every time |
| | utility attainment `A` | the share of the pool's achievable score range the human captured |
| | standardized regret `d` | the shortfall in units of the pool's own score SD. `d < 0.5` means the pick sat inside D2I's own noise — the same rank-2 pick is a near-miss in a tight pool and a real divergence in a spread one |
| **Continuation** | margin-weighted agreement | stop/continue agreement weighted by `\|u − τ\|`, so disagreeing about a 0.30-against-0.35 call counts for less than disagreeing about a 0.66 one. The unweighted rate is printed beside it — the pair is the finding |
| | AUROC | whether the utility *orders* the human's calls at all, independent of where the threshold sits. Tie-corrected, because the utilities are heavily quantised |
| | implied threshold `τ̂` | the cut that would have maximised agreement, against D2I's 0.35. Reported as the interval of cuts that tie, and suppressed entirely when every utility falls on one side of `τ` — there is then no evidence about the other side |

Plus four diagnostics: component attribution (κ_sel and attainment recomputed against each
of the five score terms alone), criterion attribution (AUROC of each of the judge's five
criteria), and the action and stop confusion matrices.

Every figure carries the chance rate it must beat and a 95% interval from a bootstrap that
resamples **whole trajectories**, not single decisions — two questions can descend from one
trajectory and one judge call, so treating them as independent would make the intervals too
narrow.

**Where it lives.** The metric *definitions* are only in `human_trajectory_eval.py`.
`build_survey.py` calls `pick_row` once per candidate slot at build time and publishes the
result in `truth.json`, so the page scores a pick by looking its slot up and never evaluates
a scoring formula. `metrics.js` reimplements only the four estimators that need the
respondent's own answers (κ, weighted mean, AUROC, Youden) plus the bootstrap — down to a
shared PRNG, so a CI computed in the browser and one computed offline agree to the last bit.
`survey/tests/test_metrics.py` asserts that.

Publishing the precomputed rows leaks nothing: each one is a function of candidate scores
`truth.json` already carried. `build_survey.py` refuses to write a bundle whose public half
contains a score, a utility, a threshold or the shuffle order.

### Judged vs inferred stop calls

`report.json` records the judge's ruling — utility, threshold and the five criteria — only
for trajectory heads it actually ruled on. Any other node with children is marked
`d2i_terminate = false` by *inference*: the search went on, so it evidently did not stop.
Those two are not equal evidence, so `terminate_source` travels with every row and the
continuous metrics (margin-weighted agreement, AUROC, `τ̂`) use only rows carrying a real
utility. Agreement is reported for both, apart, and never pooled.

Older run folders predate the structured `continuation` field; their terminated
trajectories still print `continuation utility 0.30 < 0.35` into the reason prose, which is
scraped as a fallback. That recovers the *terminate* side only — a run has to carry the
structured field for a **continue** to have a utility attached, and without both sides `τ̂`
is not identifiable and says so.

### What the downloaded file contains

The `.json` a respondent downloads is self-describing: it can be read months later without
`runs/`, without the private bundle, and without this repository.

```
results
  counts                 decisions, distinct trajectories, candidates shown, and the
                         candidate-selection / termination-selection counts split into
                         asked / answered / skipped and judged / inferred
  headline               each measure under a stable key: value, chance, 95% CI, n,
                         and whether it was estimable at that n
  summary                the same measures in table order, as raw floats
  term_attribution       kappa_sel and attainment per score term
  criterion_attribution  AUROC per judge criterion
  action_confusion       7x7, D2I's top action against the human's pick
  stop_confusion         2x2, over the calls the judge actually ruled on
  runs                   per run: directory, path, goal, columns, model, beam settings,
                         dataset shape — the provenance every decision points into
  lambdas                the weights D2I fused the five terms with
  decisions[]            one record per decision:
      source             run, run_dir, run_group, path, model, node, depth,
                         candidate_depth, trajectory
      trajectory_so_far  every step from the base insight down, with its statistic
      termination        model verdict, human verdict, agreement, verdict_source,
                         continuation_utility, threshold, margin, the five criteria,
                         and the judge's rationale
      selection          n_candidates, human_slot, model_top_slot, and candidates[]:
                         slot, action, question, model_rank, model_score,
                         score_breakdown (all five terms), model_top,
                         answered_by_model, picked_by_human
                         plus `metrics`, the scored row this decision contributed
  rows / stops           the flat per-decision metric rows the summary was computed from
  table_text             the rendered table
```

`goal` and `columns` are not repeated inside each decision — they are per *run* and live
once in `results.runs[source.run]`.

The **submitted** copy carries everything except `decisions` and `table_text`. Apps Script
writes the response body into one spreadsheet cell and Sheets truncates a cell at 50 000
characters with no error; dropping the per-decision detail keeps a 20-decision response
around 30 KB, and that detail is reconstructible from `survey.json` + `truth.json` + the
verdicts in any case. Use the download, or the emailed attachment, when you want the full
record.

### Checking it

```bash
python3 survey/tests/test_metrics.py      # estimators, invariants, baselines, JS<->Python
```

Four groups: hand-worked estimator values (and a cross-check of the tie-corrected AUROC
against `sklearn`); exact invariants, where an oracle respondent must score κ_sel 1 and
attainment 1 and an adversarial one attainment 0; a baseline self-check that runs hundreds
of uniformly random respondents and requires each metric to converge on its own chance
column — which validates a metric and its baseline at once, with no ground truth; and the
parity check, which runs `metrics.js` under node over the same answers and diffs every cell
at 1e-9.

## The Submit button

Out of the box the closing screen offers only a download, because `ENDPOINT` at the top of
`index.html` is empty. Fill it in and a **Submit my answers** button appears, posting each
response to a Google Apps Script that appends it to a Sheet and emails it to
`shaolinx@usc.edu` with the `.json` attached.

Set it up once — the steps are also at the top of [`apps_script/Code.gs`](apps_script/Code.gs):

1. `sheets.new`, name it something like *D2I survey responses*.
2. **Extensions → Apps Script**; replace the stub with `apps_script/Code.gs`; Save.
3. **Deploy → New deployment → Web app**, *Execute as* **Me**, *Who has access* **Anyone**.
   It must be "Anyone", not "Anyone with a Google account" — respondents are not signed in.
4. Authorise (it asks for Sheets and Gmail: it writes rows and sends mail).
5. Open the `/exec` URL in a browser. It should answer `ready — 0 response(s) received so far`.
6. Paste that `/exec` URL into `const ENDPOINT` in `index.html`, then commit and push.

Two sheets fill up: **responses**, one row per person including the whole response as
`raw_json`, and **answers**, one row per answer — the shape to pivot on.

After editing `Code.gs`, **Deploy → Manage deployments → edit → New version**. Saving alone
changes nothing at the existing URL, which is the usual reason a fix appears not to work.

Nothing about this weakens the blinding: a response still only names slots, so the Sheet is
no more revealing than the download was.

### Why the submit path looks the way it does

The POST goes out as `text/plain`, not `application/json`. That keeps it a CORS *simple
request*; `application/json` would trigger an `OPTIONS` preflight, which an Apps Script web
app cannot answer, and the submission would fail before it was sent.

Apps Script also answers from a redirect whose CORS headers are not guaranteed, so the
browser sometimes refuses to let the page *read* a reply that did arrive. Reporting that as
a failure would be wrong, so the page retries once with `mode: "no-cors"` and reports "sent,
but we could not confirm it". That retry is why every response carries a `id` and why
`doPost` ignores an id it has already stored — otherwise the occasional response would be
counted twice.

If a submission genuinely fails, the page says so and falls back to the download plus the
mailto, so no one is left with nothing.

### Quotas and consent

Gmail sending caps at 100 emails/day on a consumer account, 1500/day on Workspace. Sheet
writes are effectively unlimited at survey scale. If you expect a burst past the mail quota,
set `NOTIFY = ''` in `Code.gs` — the Sheet keeps working and stops being email-bound.

The intro screen tells respondents their answers stay in the browser until they choose to
send, and marks the "about you" fields optional. That wording is now load-bearing: with a
live endpoint, pressing Submit does transmit a name and email if they entered them. If this
feeds a publication, check whether your USC IRB determination covers it before recruiting.

## Scoring what comes back

Save the `.json` attachments into `survey/responses/`, then:

```bash
python3 survey/score_survey.py survey/responses/ --per-respondent -o survey/results/
python3 survey/score_survey.py survey/responses/ -o survey/results/ --export --check-parity
```

The `raw_json` column of the responses sheet holds the same thing, if you would rather
export from there than from the mailbox. That sheet also carries the headline metrics as
their own columns, so it is readable without parsing anything.

The tables are `human_trajectory_eval.report()` — the same function the CLI prints, not a
second implementation of the metrics — plus a pooled report over all respondents,
pairwise human-vs-human agreement, and a list of the decisions people split on.

Two things only the pooled path can do, because they need more than one respondent or more
choice sets than one person supplies:

* **human-implied weights** — a conditional logit over the five score terms, fitted to the
  observed picks, rescaled to D2I's own λ total and printed beside it. It answers "which
  term would the scorer have to weight more to agree with people more often". Note that
  `S_trajectory` is constant within a pool, so it cancels in the softmax and always fits to
  0 — it cannot discriminate between siblings at a node, which is itself worth knowing.
* **the human–human ceiling** — with two or more respondents, how often *they* agree with
  each other. Agreement with D2I should be read against that, not against 100%.

`--export` also writes `metrics.csv`, `picks.csv`, `stops.csv` and a booktabs
`metrics.tex` (pdfLaTeX-safe: the Greek and box-drawing characters in the terminal labels
become math mode). `--check-parity` compares each response's browser-computed table against
a recomputation here — it should be near-tautological, so a failure means a page was served
against a stale `truth.json` rather than a formula bug.

## How blinding survives a static host

A response says only *"on decision X I picked slot 3"*. Slots are a shuffle, and the map
from slot back to candidate is in `private/decisions.json`, which is never published. So
`survey.json` can be read in full by anyone and still gives away nothing: it carries no
score, no candidate status, no `d2i_terminate`, and no shuffle order. Scoring happens
offline, in `score_survey.py`, against the private half.

The one deliberate exception is `truth.json`. Showing respondents what the agent did — after
each decision, and again at the close — needs the answer key in the browser, and with feedback
on the page fetches it at load. It is a public URL either way: **someone determined could read
it ahead of answering.** If that matters more than the feedback, build with

```bash
python3 survey/build_survey.py --no-reveal
```

and no key is published at all. The page then closes with a plain thank-you, and
`score_survey.py` is unaffected — it never reads `truth.json`.

## What a decision looks like

Every question opens with the dataset itself: the goal, the typed columns, and **the rows the
agent was shown** — `report.json`'s `data_sample`, carried into `survey.json` under a
top-level `samples` map keyed by run name and drawn above the trajectory. It is the data, not
anything anyone concluded from it, so it costs no blinding; without it the goal and the
candidate questions have to be read as abstractions. A run whose `report.json` has no
`data_sample` still builds — the build prints a note and that run's decisions show the column
pills alone.

Each decision then asks up to two questions, and reveals the answer to each one before moving on:

1. **Stop or continue?** — judged on the trajectory alone.
2. → **what the search actually did**, and why, if it stopped.
3. **Which question next?** — asked at *every* node that has a candidate pool, whichever way
   you answered the first question.
4. → **the full ranking**: every candidate's score, the per-term breakdown, where your pick
   landed, and which ones the search went on to answer.

Both questions still produce **one** verdict, written when the decision completes, so a
response has at most one row per decision and `report()` reads it unchanged.

Step 3 is a deliberate departure from `human_trajectory_eval`, which ends a decision the
moment the judge says "terminate". Asking anyway roughly doubles the picks a sample yields —
a terminated node's pool is still worth judging, as "which would you ask *if you had to*" —
and it removes the CLI's caveat that the two tables cover different subsets. Revealing the
stop answer before the pick is the cost: the CLI holds both halves back precisely so that
learning the search continued here cannot tilt the pick that follows.

## Feedback after each decision

Revealing as you go is the default. `report()`'s existing footnote about dependence covers
it.

**This costs you statistical independence, and the tooling says so.** A respondent who has
seen the ranking eight times starts predicting the scorer, so their later picks are no
longer independent samples. `build_survey.py` records `feedback: true` in the private
bundle, and `human_trajectory_eval.report()` therefore prints:

> (D2I's rank was revealed after each pick, so later picks are not independent samples —
> rerun with --no-feedback for a clean rate.)

That is the CLI's own caveat, not something added here. For a headline agreement number in
a paper, build the clean version:

```bash
python3 survey/build_survey.py --no-feedback     # reveal only at the very end
python3 survey/build_survey.py --no-reveal       # never reveal, no key published
```

The trade is real in both directions: feedback makes the survey far more engaging to sit
through — respondents get something back at every step — and engagement is what gets a
survey finished. A reasonable split is `--no-feedback` for the run you cite, feedback on for
the public-facing version.

One consequence worth knowing: with feedback on, `truth.json` is fetched when the page
loads rather than at the close, so the answer key is in the tab from the start. It was
already a public URL either way; what changes is only how early it is there.

## The sampling pool and the early-stopped share

`--runs` is the pool, defaulting to `runs/20260728-014100_d2i_bench`; every run under it with
both `repository.json` and a non-empty `pruned_questions` is sampled from. Point it anywhere:
`--runs runs` uses the whole tree.

Trajectories are drawn at random, one decision point each, at a random depth. Early-stopped
nodes are then held to `--stop-share LO HI`, default `0.30 0.70`. Within that band the pool's
*natural* rate is kept, so a balanced pool is left alone and only a lopsided one is corrected,
by the least amount that satisfies the band.

**The band is a target, not a precondition.** If the pool holds too few early-stopped nodes,
the draw takes every one it can find, keeps `-n`, and says so:

```
early-stopped share: 2/20 (10%) — target band 30%–70%, the pool's own rate is 10%  ** OUTSIDE THE BAND **
```

Adding runs raises the share on its own, with no change here — the draw always reaches for the
band first. `--strict-share` instead shortens the sample until the band genuinely holds, for
when the share has to be a guarantee.

Early-stopped nodes are scarce: they are the ones the multi-agent judge halted, and a run
yields one or two against a dozen ordinary nodes. As of writing the default pool has **2**
(20%), so 30% is not reachable there; adding runs of the same shape raises it.

Older runs are not usable as a pool any more: their `report.json` predates `data_sample`, so
their questions would be shown without the rows. Build from runs of the current shape.

Decisions are **presented grouped by run**: every decision sampled from `carsales-easy` is
shown together, then every one from the next dataset. The draw is unchanged — which
trajectories, which depths, and how many early-stopped nodes are all still random — only the
order they appear in is fixed, so a respondent reads one dataset's goal, columns and sample
rows, judges everything drawn from it, and only then moves on. Jumping between datasets costs
a re-read of the schema on every screen, and that fatigue is a worse bias than any order
effect grouping introduces. Both the run order and the order within each run are still
shuffled, so no dataset is systematically judged first and the early-stopped nodes are not
bunched at the front of a block.

Sample size is held even when the pool is small. One decision point per trajectory caps the
draw at the number of trajectories (6 in the default pool), so once those run out the draw
continues through trajectories already used — the *nodes* stay distinct, so no candidate pool
is judged twice, and only the one-per-trajectory spacing is given up. It says so when it does.

## What each build is

One seed, one bundle: every respondent judges the **same** decisions in the same order.
That is what makes their answers comparable and inter-rater agreement computable. Rebuilding
changes `built_at`, which invalidates any answers saved in a respondent's browser (the ids
and the shuffle have moved) and makes `score_survey.py` refuse responses from an older
seed — so rebuild *before* you start recruiting, not during.

Sampling, the candidate pools, the early-stopped mix and the shuffle are all
`human_trajectory_eval.build_decisions`, called directly. The survey cannot drift from the
CLI because it does not reimplement any of it.

Not every decision asks both questions. A node the beam abandoned was never judged on
stopping, so it goes straight to the candidates; an early-stopped node has no candidates by
construction, so it only asks about stopping. The rest ask both.

## Current build

10 decisions from `runs/20260728-014100_d2i_bench` (`carsales-easy`), seed 3 — 2 early-stopped
(20%, under the band), 8 offer candidates. The run holds 6 trajectories, so 4 decisions come
from trajectories already drawn from; the nodes are all distinct. Every question shows the
275-row dataset's 5-row sample.

```bash
python3 survey/build_survey.py -n 10 --seed 3        # exactly this build
```

A run is usable only if it has **both** `repository.json` and a non-empty `pruned_questions`
in `report.json`, and shows its data only if it also has `data_sample`; older runs predate
these and are skipped or noted. Add runs to the folder and rerun the build — the early-stopped
share climbs towards the band by itself.
