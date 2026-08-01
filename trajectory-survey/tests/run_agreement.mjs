/* Runs agreement.js under node over cells handed in on argv, and prints the report as
 * JSON. tests/test_agreement.py diffs that against agreement.py cell by cell — the two
 * implementations are only allowed to exist because this check keeps them equal.
 *
 *   node tests/run_agreement.mjs cells.json [order.json] [bootstrap-reps]
 */
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const { TrajAgreement } = require(join(here, "..", "agreement.js"));

const [cellsPath, orderPath, reps] = process.argv.slice(2);
const cells = JSON.parse(readFileSync(cellsPath, "utf8"));
const order = orderPath && orderPath !== "-" ? JSON.parse(readFileSync(orderPath, "utf8")) : null;

const report = TrajAgreement.compute(cells, {
  order: order || undefined,
  bootstrap: reps === undefined ? undefined : Number(reps),
});
process.stdout.write(JSON.stringify({report, text: TrajAgreement.toText(report),
                                     csv: TrajAgreement.toCSV(report)}));
