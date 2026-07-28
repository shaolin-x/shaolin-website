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
```

The `raw_json` column of the responses sheet holds the same thing, if you would rather
export from there than from the mailbox.

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

## Feedback after each decision

By default, answering a decision reveals what the agent did there — its ranking of every
candidate, the per-term score breakdown, where your pick landed — and a **Next** button
carries on. Both halves land together, after the candidate pick rather than between the two
questions: knowing the agent continued here would imply it answered one of the candidates,
which would tilt the pick that follows.

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
