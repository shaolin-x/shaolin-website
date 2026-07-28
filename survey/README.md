# D2I human evaluation — the web survey

The static counterpart to `human_trajectory_web.py`. That tool is local: it holds `runs/`
open, samples a fresh session per judge, and enforces blinding inside the request handler.
A GitHub Pages site can do none of that, so the same work moves to build time and the
result is split into a public half and a private one.

```
survey/
  build_survey.py          freezes one sampled evaluation into the two halves
  index.html               the survey itself — static, no build step, no dependencies
  score_survey.py          turns returned responses into the evaluation's own report
  data/
    survey.json            PUBLIC. blinded decisions — deployed
    truth.json             PUBLIC. the answer key, for the closing screen — deployed
  private/
    decisions.json         the full decisions with the shuffle — gitignored, never deployed
  responses/               where you drop the files people send back — gitignored
```

## Running it

```bash
python3 survey/build_survey.py            # rebuild after runs/ changes
python3 -m http.server 8000               # from the site root
open http://localhost:8000/survey/
```

Opening `index.html` straight off disk will not work — the browser blocks `fetch` on
`file://`. The page says so if you try.

Deploying is just committing `survey/index.html` and `survey/data/`. `runs/` and
`survey/private/` are gitignored and are not needed at serve time.

## Scoring what comes back

```bash
python3 survey/score_survey.py survey/responses/ --per-respondent -o survey/results/
```

The tables are `human_trajectory_eval.report()` — the same function the CLI prints, not a
second implementation of the metrics — plus a pooled report over all respondents,
pairwise human-vs-human agreement, and a list of the decisions people split on.

## How blinding survives a static host

A response says only *"on decision X I picked slot 3"*. Slots are a shuffle, and the map
from slot back to candidate is in `private/decisions.json`, which is never published. So
`survey.json` can be read in full by anyone and still gives away nothing: it carries no
score, no candidate status, no `d2i_terminate`, and no shuffle order. Scoring happens
offline, in `score_survey.py`, against the private half.

The one deliberate exception is `truth.json`. The closing screen shows respondents how they
did — a real draw for participation — and that needs the answer key in the browser. The page
does not request it until the last decision is locked in, but it is a public URL: **someone
determined could fetch it early.** If that matters more than the closing screen, build with

```bash
python3 survey/build_survey.py --no-reveal
```

and no key is published at all. The page then closes with a plain thank-you, and
`score_survey.py` is unaffected — it never reads `truth.json`.

## What each build is

One seed, one bundle: every respondent judges the **same** decisions in the same order.
That is what makes their answers comparable and inter-rater agreement computable. Rebuilding
changes `built_at`, which invalidates any answers saved in a respondent's browser (the ids
and the shuffle have moved) and makes `score_survey.py` refuse responses from an older
seed — so rebuild *before* you start recruiting, not during.

Sampling, the candidate pools, the early-stopped mix and the shuffle are all
`human_trajectory_eval.build_decisions`, called directly. The survey cannot drift from the
CLI because it does not reimplement any of it.

About half of a build is early-stopped nodes. That is intentional and is the module's
doing: such nodes carry no candidates, so they only ask the stop/continue question, and
without deliberately over-sampling them that question would have "continue" as its answer
nearly every time. They make the survey faster than the decision count suggests.

## Current build

Whatever `runs/` held when you last ran the build script. As of writing that is one usable
run — `20260727-201135_d2i_bench/carsales-easy`, 8 distinct decision points, which is all
its 9 trajectories yield. A run is usable only if it has **both** `repository.json` and a
non-empty `pruned_questions` in `report.json`; older runs predate those and are skipped with
a note. Add more runs and rerun the build to lengthen the survey.
