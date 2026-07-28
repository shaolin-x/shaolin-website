/* Aggregate one response through metrics.js and print it as JSON.
 *
 * The Python half of the parity test (`test_parity.py`) runs this under subprocess for
 * every synthetic respondent and diffs each cell against `human_trajectory_eval.aggregate`.
 * Kept as a thin argv wrapper so the thing under test is exactly the file the page loads.
 *
 *   node survey/tests/run_metrics.mjs <survey.json> <truth.json> <response.json>
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../metrics.js");

const [surveyPath, truthPath, responsePath] = process.argv.slice(2);
const read = (p) => JSON.parse(readFileSync(p, "utf8"));

const survey = read(surveyPath);
const truth = read(truthPath);
const response = read(responsePath);

const table = survey.meta.metrics.table;
const rows = M.rowsFor(response.verdicts || [], truth, survey.decisions);
const agg = M.aggregate(rows, table);

process.stdout.write(JSON.stringify({
  engine: M.VERSION,
  summary: agg,
  n_picks: rows.picks.length,
  n_stops: rows.stops.length,
  n_skipped: rows.skipped,
  terms: M.termAttribution(rows.picks, survey.meta.metrics.terms || ["llm", "sem", "cov", "imp", "traj"]),
  criteria: M.criterionAttribution(rows.stops),
  action_confusion: M.confusion(rows.picks, "top_action", "pick_action"),
  stop_confusion: M.confusion(rows.stops, "model", "human", ["false", "true"])
}, null, 2));
