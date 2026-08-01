# Trajectory rating — human vs the G-Eval judge

The earlier study (`survey/`, now closed) asked which question an agent should ask next.
This one asks something else entirely: **given a whole trajectory, how good is it?** — the
same six criteria, the same 1–5 scale and the same rubric text the LLM judge in
`eval/geval_trajectory.py` was given, so a human score and a model score are directly
comparable and the thing being measured is their agreement.

```
trajectory-survey/
  build_survey.py     samples trajectories and freezes one study into data/
  index.html          the survey — static, no build step, no dependencies
  agreement.js        the agreement scoring the closing screen uses — deployed
  agreement.py        its twin, for the offline analysis
  score_survey.py     turns returned responses into the study's report
  data/
    survey.json       PUBLIC. the items, and which system wrote each — deployed
    truth.json        PUBLIC. the judge's scores and reasons — deployed
  private/
    build.json        which system is behind each item + the judge prompt hashes — gitignored
  responses/          where you drop the files people send back — gitignored
  tests/
    test_agreement.py the estimators' correctness suite, including JS↔Python parity
    run_agreement.mjs runs agreement.js from node, for that parity check
    synth_responses.py oracle / adversarial / random / noisy respondents
    harness.html      drives the whole page in a browser, as a smoke test
```

## Running it

```bash
python3 trajectory-survey/build_survey.py            # rebuild
python3 -m http.server 8000                          # from the site root
open http://localhost:8000/trajectory-survey/
```

Opening `index.html` off disk will not work — the browser blocks `fetch` on `file://`. The
page says so if you try.

Deploying is committing `index.html`, `agreement.js` and `data/`. **All three** —
`index.html` without `agreement.js` still runs the survey but shows no results at the end,
and the failure is silent.

## What the respondent is shown

Exactly what the judge was sent, and nothing else:

* the **analytical goal**, verbatim from the run;
* **the dataset itself** — its own table, folded away after the first look. This is one
  place the screen departs from the judge's prompt on purpose: the judge is given a schema
  profile, which is the wrong thing to put in front of a person, so the rows go in its
  place. The profile is still what the prompt receipts are computed from;
* the **trajectory**, rendered from the judge module's own `_format_trajectory`: each step's
  statement, the evidence attached to it, and — labelled as *not* evidence — any
  system-generated metadata;
* the **rubric**: each criterion's evaluation steps, its question and all five level
  descriptions, parsed straight out of the judge's `RUBRIC` constant.

The build hashes the exact user message the judge received for each item and records it in
`private/build.json`, so "the human saw the same thing" is a receipt rather than a claim.

No score is shown until the closing screen. Which system produced the trajectory is not
shown on the rating screen either — but it *is* in the bundle the page fetches, so it is
readable by anyone who looks; see "What a returned response identifies" for the trade and
for `--blind-systems`.

## The build cannot drift from the judge

`build_survey.py` imports `eval/geval_trajectory.py` from the D2I repository and calls it —
`detect_kind`, `load_run_trajectories`, `load_profile_text`, `_format_trajectory`,
`_build_user_prompt`. Nothing is reimplemented here.

Two things are then checked rather than assumed, and both are fatal:

1. **The instrument.** The module's `_prompt_hash()` must equal the `prompt_hash` recorded in
   the scoring run's `meta.json`. Point it at a module with an edited rubric and the build
   refuses, because the human half would be measuring a different instrument
   (`--allow-hash-mismatch` overrides it, for inspection only).
2. **The alignment.** A judge score is attached to a `trajectory_index`, so every loaded
   trajectory's step count and depth must match what the scoring run recorded for that
   index. A run directory that has changed since it was judged fails the build instead of
   silently pairing a human's rating with another trajectory's score.

```bash
python3 trajectory-survey/build_survey.py \
    --geval-run runs/geval_runs/20260731-133448_systems \
    --geval-module /Users/shaolinx/Desktop/D2I_repo/eval/geval_trajectory.py
```

Run directories are read from the site's own `runs/` copy, falling back to the path recorded
in the scoring output. Both were checked to load byte-identical trajectories.

## D2I trajectories arrive truncated, and are repaired here

`load_d2i_trajectories` groups `repository.json`'s records by `trajectory_id`. D2I mints a
**new** trajectory_id when the beam forks, and the ancestors keep the parent branch's id —
so that grouping returns the tail *segment* of a path, not the path.

Across this repository's D2I runs it truncates **95 of 129** reported trajectories: 44 of
them down to a single node sitting at depth 3, with no base insight in sight. On
`revenue_opportunities`, for instance, two of the five reported trajectories load as one
node each; both are really four-node chains.

This matters more here than almost anywhere. Resolution is defined as "does the trajectory
resolve the uncertainty introduced by *the initial finding*" and Information Gain as
progression "relative to the preceding trajectory" — so a lone depth-3 node is being rated,
by a person or by the judge, on progression it was never shown. Both criteria can only come
out low, for a reason that has nothing to do with D2I's search.

The real path is recoverable exactly: every record carries its own `id` and its
`parent_id`, so walking back from the deepest node of a reported trajectory reconstructs it.
`d2i_full_paths()` does that, rebuilding node dicts in the loader's own shape with evidence
from the judge's own `d2i_node_evidence`, and the build says what it repaired:

```
  d2i/cms_hospital_readmissions-level_2: rebuilt 5 truncated trajectory/ies from parent_id
```

Two consequences worth being explicit about:

* **The `20260731-133448_systems` scores were computed on the truncated trajectories.** They
  are not comparable with ratings collected from the repaired ones, and `--attach-scores`
  will refuse to pair them — its prompt-hash check fails, which is exactly what that check
  is for. Fix `load_d2i_trajectories` in the D2I repo the same way and re-run the judge
  before comparing anything.
* `--d2i-as-judged` turns the repair off, reproducing what the last scoring run saw. That is
  what you want only if you are reproducing that run, never for collecting fresh ratings.

## Collecting before anything has been judged

`runs/` moves faster than the scoring does. `--no-judge` builds from `runs/` alone: it
samples and renders trajectories exactly as above — same loaders, same rubric, same prompt
receipts — but attaches no model scores, so a rewritten `runs/` can be put in front of
people the same day it lands.

```bash
python3 trajectory-survey/build_survey.py --no-judge -d 3 -n 2
```

It implies `--no-reveal`: there is no key to publish, the page closes with a thank-you and
**no agreement summary**, and the response file's `results` block says why. Run discovery is
`sweeps.discover`, the judge's own, so which arm/dataset/config a run is comes from the same
place either way.

**Scoring those responses later.** Rebuilding to get a key would move `built_at` and every
item id, and `score_survey.py` would then rightly refuse the responses — they would be
answers to different questions. So the key is written for the bundle that is already
deployed instead:

```bash
python3 eval/geval_trajectory.py --group systems --repeats 3        # in the D2I repo
python3 trajectory-survey/build_survey.py --attach-scores \
        --geval-run runs/geval_runs/<the new run>
python3 trajectory-survey/score_survey.py trajectory-survey/responses/
```

`--attach-scores` keeps `built_at`, `seed` and every item id, and re-renders each item's
judge prompt to check it hashes to what `private/build.json` recorded at build time. If a
run directory has been rewritten since people rated it, that check fails loudly rather than
pairing a human's rating with a different trajectory's score:

```
build_survey: these trajectories are not what was shown when the ratings were collected —
  d2i/salesfact #1
```

Items the scoring run does not cover are reported and simply get no key; the rest are
scored.

## The data a respondent sees

`data/` is **gitignored** — covid.csv alone is 346 MB, past GitHub's 100 MB per-file limit —
so the CSVs never reach the deployed site. Anything a respondent is to see has to be written
into `trajectory-survey/data/tables/`, which is committed, and that is what the build does:

```
  carsales                     275 of       275 rows x   5 cols  ->   0.0 MB  (whole table)
  cases                     10,000 of    10,000 rows x  58 cols  ->   7.3 MB  (whole table)
  yelp_reviews               2,610 of     2,610 rows x  10 cols  ->   0.9 MB  (whole table)
  covid                      1,000 of 3,348,186 rows x  13 cols  ->   0.1 MB  ** TRUNCATED **
```

Whole where it fits, truncated where it cannot: covid is 3.35 million rows, so its first
`--truncate-rows` (default 1000) ship and the panel says so beside the true count, every
time it is opened. `--table-budget-mb` (default 8) is the line between the two.

Each table is its own file, fetched the first time someone opens the panel rather than by
every respondent whether they look or not, and rendered **windowed** — 10 000 × 58 is
580 000 cells, so only the ~40 rows around the scroll position are ever in the DOM.

## Sampling

```bash
python3 trajectory-survey/build_survey.py -d 4 -n 1   # all four pooled datasets, 1 per system
python3 trajectory-survey/build_survey.py -d 2 -n 2   # 2 of them, 2 per system    → 12 items
python3 trajectory-survey/build_survey.py --seed 7    # a different draw
```

* `--pool` is the set a draw may use, defaulting to the datasets whose tables are on hand:
  **covid, carsales, cases, yelp_reviews**. `--all-datasets` lifts it, at the cost of screens
  with no data panel.
* `-d/--datasets N` draws N datasets **shared by all three systems**. A system comparison is
  only a system comparison if every system was judged on the same data.
* `-n/--per-system N` then draws N trajectories per system **within** each dataset,
  independently per system — there is no reason the third trajectory of one system should be
  paired with the third of another. Total items = datasets × systems × N.

Items are presented **grouped by dataset**, because each dataset carries a ~6 000-character
profile the respondent has to read; interleaving datasets would mean re-reading it on every
screen, and that fatigue is a worse bias than any order effect grouping introduces. The
dataset order and the order within each dataset are both shuffled, so no system is
systematically judged first.

One seed, one bundle: every respondent rates the **same** trajectories, which is what makes
inter-rater agreement computable. Rebuilding moves `built_at` and the item ids, which
invalidates answers saved in a respondent's browser and makes `score_survey.py` refuse
responses from the older build — so rebuild *before* recruiting, not during.

Budget about three minutes per trajectory; the home page's "Participate" blurb quotes a
duration and is worth updating if you change `-n`.

## QUIS trustworthiness is assigned, not judged — and the respondent is told so

The judge is not asked for QUIS's trustworthiness. QUIS records no claim-level evidence —
every one of its steps reads *"No claim-level execution evidence was retained by the
system"* — so an evidence-grounded rubric could only ever score it 1–2, which would read as
"QUIS makes untrustworthy claims" when it means "QUIS persists no evidence". The criterion
is fixed at 5 by policy and that rubric section is cut from its prompt.

Asking a human for it while saying nothing would be worse than asking the model: they would
sit and score the absence of evidence, and their score would measure an output format. So
the criterion is **shown, marked "optional here", and explained on the screen**:

> Not required for this trajectory. This system records no claim-level evidence — every step
> above says so — because its statements are direct renderings of statistics computed over
> the stated subspace, and hold of the data by construction. There is nothing here for an
> evidence-grounded criterion to weigh, so it is fixed by policy rather than judged, and the
> model is not asked for it either. Answer it if you have a view; you can move on without it.

It does not block **Next**, the screen's counter reads "All 5 scored — ready to move on (1
optional on this one)", and the digit keys fill the required scales first. Anything given
anyway is kept, excluded from agreement, and reported separately — `score_survey.py` prints
what humans gave those cells against the assigned 5, which is worth knowing and is not
agreement.

The wording is a paraphrase of the judge's own, deliberately: its version names QUIS, and
naming the system on the screen would unblind the item. **The mark itself is a soft leak** —
a respondent who notices that only some trajectories carry it can tell those apart — which
is the price of not making people score something that was never recorded.

`--match-assigned` instead drops the criterion from those items altogether: the strictest
match to the judge's prompt, and a harder leak, since those screens then show five scales
where the others show six.

## The scoring

Each **cell** is one (trajectory, criterion) pair: a human's integer against the judge's
mean over its repeats.

| measure | what it answers |
|---|---|
| `exact` | you and the judge's rounded mean picked the same number — against the rate two raters with *these* marginals would hit by accident, which is well above 1/5 whenever both cluster mid-scale |
| `±1` | within one point of the unrounded mean. Measured unrounded, and its chance rate is computed unrounded too, so the baseline is not quietly more permissive than the measure |
| `MAE` | mean distance in scale points — the one number that says how far apart, not how often |
| `r`, `rho` | correlation across trajectories: whether the two **ranked** them alike, independently of whether either scored high or low. A judge can be badly calibrated and still order things correctly |
| `QWK` | chance-corrected agreement that counts a 2-point miss more than a 1-point one |
| `alpha` | Krippendorff's alpha, interval, over the unrounded judge mean |
| **judge vs itself** | the same measures computed between the judge's own repeats. **This is the ceiling.** Agreeing with a judge more often than it agrees with itself is not possible, so a human number means nothing until it is read against this column |

Every figure carries a 95% interval from a bootstrap that resamples **whole trajectories**,
not single cells — one trajectory yields six ratings from one person reading one screen, and
treating those as independent draws would make the intervals too narrow.

`score_survey.py` adds what needs more than one respondent or the unblinding:

* **by system** — each system's mean from each side, and whether the two **rank** the systems
  the same way. This is the finding a benchmark table rests on: a judge that disagrees cell
  by cell but orders the systems identically still supports the same conclusion, and the
  reverse is a much more serious problem than a low `exact`.
* **human vs human** — with two or more respondents, how often *they* agree. The second
  ceiling, and usually the more honest one.
* **where respondents split** — the trajectories people disagreed about most, worth reading
  before trusting any single number about them.

```bash
python3 trajectory-survey/score_survey.py trajectory-survey/responses/ --per-respondent
python3 trajectory-survey/score_survey.py trajectory-survey/responses/ --export -o trajectory-survey/results/
```

`--export` writes `metrics.csv`, `cells.csv`, `systems.csv` and a booktabs `metrics.tex`.
`--check-parity` diffs each response's browser-computed table against a recomputation here;
it should be near-tautological, so a failure means a page was served against a stale
`truth.json`.

### Two implementations, one table

`agreement.js` is deployed with the page; `agreement.py` is what the offline analysis uses.
They share a PRNG (mulberry32) and a resampling order, so a CI computed in a browser and one
computed here agree to the last bit — including some deliberate awkwardness:
`agreement.py` never calls the builtin `sum` on floats, because since Python 3.12 it applies
Neumaier compensation and would disagree with JavaScript's `reduce` in the last ulp.

```bash
python3 trajectory-survey/tests/test_agreement.py
```

Four groups: hand-worked estimator values (with a cross-check against scipy/sklearn where
installed); invariants, where an oracle respondent must score exact agreement 1 and an
adversarial one must correlate negatively; a baseline self-check that runs hundreds of
uniformly random respondents and requires each metric to converge on its own chance column —
which validates a metric and its baseline at once, with no ground truth; and the parity
check, which runs `agreement.js` under node over the same cells and diffs every cell at
1e-9, bootstrap intervals included.

`tests/harness.html` is the browser smoke test: it drives the page end to end inside an
iframe — Start, every scale on every screen, Next, through to the closing table — so a
runtime error on a screen that only appears after thirty clicks fails loudly. Instructions
are at the top of the file.

## What a returned response identifies

An answer names its trajectory twice:

* `item` — the build's id for a slot, `blake2b(seed | arm | dataset | level | index)`. It
  names **coordinates**, not content: rewrite `runs/` and `d2i/salesfact #1` still hashes to
  the same id while being a different trajectory. Unique within a build (asserted at build
  time), and *deliberately* the same across builds that share a seed and draw the same
  trajectory — which is why `score_survey.py` also checks `built_at`.
* `content` — the sha256 of the judge prompt that was on the screen. This one **is** the
  trajectory: it changes the moment anything about the goal, the profile or the steps does.

**A response records the respondent and nothing else.** No judge score, no agreement table:
those were never asked of the person, and are recomputed offline from a scoring run
whenever they are wanted. The closing screen still shows them — that is the page being
useful to whoever just sat through it — but they do not enter the file.

What the download does carry is **the screen itself**, per answer, under `screen`: the
position (`3 of 18`), the goal, the whole dataset profile as it was shown, every step with
its evidence and metadata, and each criterion with the rubric text that sat beside its
scale — its question, its five level descriptions, its evaluation steps, and the note on
one marked optional. It repeats the profile and the rubric on every screen on purpose: a
record of what someone was shown beats a normalised one, at a few hundred kilobytes in a
file nobody has to parse in a hurry (~60 KB for 3 items, ~350 KB for 18).

A **submitted** response drops `screen`: it has to fit in one 45 000-character spreadsheet
cell. `--identify --annotate` puts the trajectory back from `runs/`.

Both carry the system. Each answer names its `dataset` and `system`, and `shown`'s
per-trajectory entries add `arm`, the goal and the steps — so one file says what was rated,
what produced it, and how it was scored, with nothing to join against.

**This is a deliberate trade, and it costs the blinding.** The system name reaches the
download by being in `survey.json`, which the page fetches, so a respondent who opens the
network tab can see which agent wrote the trajectory in front of them — and, since ratings
can be revised, could act on it. The page itself never shows it before the closing screen,
but that is a convention, not a guarantee. Build with `--blind-systems` to keep it out of
the served bundle entirely; the download is then silent about the system too, and

```bash
python3 trajectory-survey/build_survey.py --identify responses/theirs.json \
        --annotate responses/theirs.annotated.json
```

puts it back afterwards, offline. That resolution works either way:

It re-renders every trajectory under `--runs-root` and matches on the content hash — no
private bundle, no answer key, no trust in the id. It prints the mapping and, with
`--annotate`, writes a copy in which every answer carries its `system`, `dataset`,
`trajectory_index`, `run_dir`, goal and step text:

```
  9f288f51  ->  quis/yelp_reviews trajectory #1
            runs/quis/yelp_reviews-level_2
  …
  18/18 resolved
```

A trajectory that has been rewritten since it was rated simply does not match, and says so
rather than resolving to something plausible. `score_survey.py` makes the same check, and
drops — loudly — any rating whose content hash disagrees with the key's.

## Blinding, and the one deliberate leak

`data/survey.json` names no system and carries no score; the build searches its own output
for the system names and for any score field before writing it, because a leak here would be
silent and would make every response worthless.

`data/truth.json` is the exception: showing respondents what the judge said at the close
needs the answer key in the browser, and it is a public URL. **Someone determined could read
it before answering.** If that matters more than the feedback:

```bash
python3 trajectory-survey/build_survey.py --no-reveal
```

No key is published, the page closes with a plain thank-you, and `score_survey.py` is
unaffected — it reads the key from `data/truth.json` or, with `--truth`, from
`private/build.json`.

Ratings are revealed only **at the end**, never after each trajectory. The earlier survey
revealed as it went, which cost it statistical independence; here the whole point is
agreement, so nothing about the judge is on screen while there are still items to rate.

## The Submit button

Out of the box the closing screen offers only a download, because `ENDPOINT` at the top of
`index.html` is empty. Fill it in and a **Submit my answers** button appears, posting each
response to the Apps Script in [`apps_script/Code.gs`](apps_script/Code.gs), which appends it
to a Google Sheet and emails it on with the `.json` attached. The setup steps are at the top
of that file; it is the earlier survey's script adapted to this response shape, and the
same notes about `text/plain`, the opaque retry and the Gmail quota apply.

Two sheets fill up: **responses**, one row per person with the headline agreement as its own
columns, and **ratings**, one row per (trajectory, criterion) — the shape to pivot on.

## Which build is deployed

`data/survey.json`'s `meta` block says, and is the only thing that does — `built_at`,
`seed`, `n_items`, `datasets`, `per_system`, and the scoring run it is paired with:

```bash
python3 -c "import json;print(json.load(open('trajectory-survey/data/survey.json'))['meta'])"
```

`meta.judged` says whether a key exists at all: `false` is a `--no-judge` build, which
collects ratings and shows no agreement summary. The intro screen quotes its own item count
and a three-minutes-per-trajectory estimate from that same block, so it never disagrees with
the bundle it is serving.

The scoring run `20260731-133448_systems` predates the current `runs/` and covers only the
datasets it was run on; anything built from today's `runs/` needs a fresh one before
`--attach-scores` has scores to attach.
