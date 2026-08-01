/* Human-vs-judge agreement on the six G-Eval criteria.
 *
 * Deployed with the page (the closing screen reads its numbers from here) and run from
 * node by tests/run_agreement.mjs, which diffs every cell against agreement.py — the two
 * implementations share a PRNG and a resampling order so a CI computed in a browser and
 * one computed offline agree to the last bit.
 *
 * A "cell" is one (trajectory, criterion) pair: the respondent's integer 1-5 against the
 * judge's mean over its repeats. Cells the judge did not actually judge — QUIS's
 * trustworthiness, which is assigned 5 by policy — are dropped before anything is
 * computed and counted separately; averaging a human rating against a constant nobody
 * chose would flatter or punish the agreement for no reason.
 *
 * Everything is reported against a chance rate AND against the judge's agreement with
 * itself across its own repeats. The second is the one that matters: a human matching the
 * judge 60% of the time when the judge matches itself 65% of the time is a very different
 * finding from the same 60% against a self-consistent judge.
 */
(function (root) {
  "use strict";

  const BOOTSTRAP = 1000;
  const SEED = 20260731;

  /* ------------------------------------------------------------------ basics */

  const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
  // Away from zero at .5, which never actually arises for a mean of three integers
  // (n/3 is never x.5) but keeps JS and Python identical if the repeat count changes.
  const round1 = (x) => Math.sign(x) * Math.round(Math.abs(x));

  function pearson(xs, ys) {
    const n = xs.length;
    if (n < 2) return null;
    const mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - mx, dy = ys[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx <= 0 || syy <= 0) return null;     // a flat column has no correlation to report
    return sxy / Math.sqrt(sxx * syy);
  }

  /* Ranks with ties averaged — the judge's means take few distinct values, so ties are
     the rule here rather than an edge case. */
  function ranks(xs) {
    const idx = xs.map((v, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
    const out = new Array(xs.length);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && xs[idx[j + 1]] === xs[idx[i]]) j++;
      const r = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) out[idx[k]] = r;
      i = j + 1;
    }
    return out;
  }

  const spearman = (xs, ys) => (xs.length < 2 ? null : pearson(ranks(xs), ranks(ys)));

  /* Quadratically weighted kappa over the 1-5 scale, against the judge's rounded mean.
     Quadratic weights because the scale is ordinal: rating a 5 as a 4 is not the same
     mistake as rating it a 1, and unweighted kappa cannot tell those apart. */
  function qwk(a, b) {
    const K = 5, n = a.length;
    if (!n) return null;
    const O = Array.from({length: K}, () => new Array(K).fill(0));
    const ra = new Array(K).fill(0), rb = new Array(K).fill(0);
    for (let i = 0; i < n; i++) {
      const x = Math.min(K, Math.max(1, round1(a[i]))) - 1;
      const y = Math.min(K, Math.max(1, round1(b[i]))) - 1;
      O[x][y] += 1; ra[x] += 1; rb[y] += 1;
    }
    let num = 0, den = 0;
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      const w = ((i - j) * (i - j)) / ((K - 1) * (K - 1));
      num += w * O[i][j];
      den += w * (ra[i] * rb[j]) / n;
    }
    if (den <= 0) return null;                 // both raters constant and identical
    return 1 - num / den;
  }

  /* Krippendorff's alpha, interval difference, two coders, no missing values.
     Computed from the coincidence matrix rather than from a two-coder shortcut, so the
     judge's non-integer means need no rounding to enter it. */
  function alphaInterval(a, b) {
    const n = a.length;
    if (n < 2) return null;
    let Do = 0;
    for (let i = 0; i < n; i++) Do += (a[i] - b[i]) * (a[i] - b[i]);
    Do /= n;
    const all = a.concat(b), N = all.length;
    let De = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (i !== j) De += (all[i] - all[j]) * (all[i] - all[j]);
    }
    De /= N * (N - 1);
    if (De <= 0) return null;                  // every value identical: no disagreement to explain
    return 1 - Do / De;
  }

  /* ------------------------------------------------------------------ one table row */

  /* `rows` are {human, judge} pairs. `exact` compares against the judge's rounded mean,
     `within1` and `mae` against the unrounded one — rounding first would throw away the
     part of the judge's answer that says "the three repeats disagreed". */
  function stats(rows) {
    const h = rows.map(r => r.human), m = rows.map(r => r.judge);
    const n = rows.length;
    if (!n) return {n: 0};
    const mr = m.map(round1);
    const exact = mean(rows.map((r, i) => (r.human === mr[i] ? 1 : 0)));
    const within1 = mean(rows.map((r, i) => (Math.abs(r.human - m[i]) <= 1 ? 1 : 0)));
    const mae = mean(rows.map((r, i) => Math.abs(r.human - m[i])));

    // Chance for `exact` is not 1/5: it is what two raters with THESE marginals would hit
    // by accident, which is higher whenever both concentrate on the middle of the scale.
    const ph = new Array(6).fill(0), pm = new Array(6).fill(0);
    for (let i = 0; i < n; i++) {
      ph[Math.min(5, Math.max(1, h[i]))] += 1 / n;
      pm[Math.min(5, Math.max(1, mr[i]))] += 1 / n;
    }
    // `within1` is measured against the UNROUNDED judge mean, so its chance rate is too:
    // rounding first would quietly widen the band (a human 4 is within 1 of a rounded 3
    // but not of a 2.667) and the baseline would sit above the thing it baselines.
    let exactChance = 0, within1Chance = 0;
    for (let k = 1; k <= 5; k++) {
      exactChance += ph[k] * pm[k];
      within1Chance += ph[k] * mean(m.map(mi => (Math.abs(k - mi) <= 1 ? 1 : 0)));
    }

    return {
      n,
      human_mean: mean(h),
      judge_mean: mean(m),
      bias: mean(rows.map((r, i) => r.human - m[i])),
      exact, exact_chance: exactChance,
      within1, within1_chance: within1Chance,
      mae,
      pearson: pearson(h, m),
      spearman: spearman(h, m),
      qwk: qwk(h, m),
      alpha: alphaInterval(h, m),
    };
  }

  /* ------------------------------------------------------------------ the judge's own ceiling */

  /* The same table, computed between the judge's repeats instead of between a human and
     the judge: every unordered pair of repeats of every cell, entered in both orders so
     the result does not depend on which repeat was called "first". */
  function selfRows(cells) {
    const out = [];
    for (const c of cells) {
      const s = c.judge_scores || [];
      for (let i = 0; i < s.length; i++) for (let j = i + 1; j < s.length; j++) {
        out.push({criterion: c.criterion, item: c.item, human: s[i], judge: s[j]});
        out.push({criterion: c.criterion, item: c.item, human: s[j], judge: s[i]});
      }
    }
    return out;
  }

  /* ------------------------------------------------------------------ bootstrap */

  /* mulberry32 — small, exactly reproducible, and identical to agreement.py's. */
  function prng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const CI_METRICS = ["exact", "within1", "mae", "pearson", "spearman", "qwk", "alpha"];

  /* Resample TRAJECTORIES, not cells: one trajectory contributes six ratings from one
     person reading one screen, and treating those as six independent draws would make
     every interval too narrow. */
  function bootstrap(cells, byItem, keyOf, reps, seed) {
    // Sorted, not in insertion order: an all-digit item id is an integer-like object key,
    // which JS enumerates numerically first, so insertion order is not portable — and the
    // resample has to draw the same items here as it does in agreement.py.
    const items = Object.keys(byItem).sort();
    const rand = prng(seed);
    const draws = {};
    for (let b = 0; b < reps; b++) {
      const pool = [];
      for (let i = 0; i < items.length; i++) {
        pool.push(...byItem[items[Math.floor(rand() * items.length)]]);
      }
      const groups = {};
      for (const c of pool) (groups[keyOf(c)] = groups[keyOf(c)] || []).push(c);
      for (const k of Object.keys(groups).sort()) {
        const s = stats(groups[k]);
        for (const m of CI_METRICS) {
          const v = s[m];
          if (v === null || v === undefined || Number.isNaN(v)) continue;
          ((draws[k] = draws[k] || {})[m] = draws[k][m] || []).push(v);
        }
      }
    }
    const out = {};
    for (const k of Object.keys(draws)) {
      out[k] = {};
      for (const m of Object.keys(draws[k])) {
        const xs = draws[k][m].slice().sort((x, y) => x - y);
        if (xs.length < reps / 2) continue;    // too often unestimable to quote an interval
        out[k][m] = [xs[Math.floor(0.025 * xs.length)],
                     xs[Math.min(xs.length - 1, Math.floor(0.975 * xs.length))]];
      }
    }
    return out;
  }

  /* ------------------------------------------------------------------ the report */

  /* `cells`: {item, dataset, criterion, human, judge, judge_scores, assigned}.
     Returns the whole report — per criterion, pooled, the judge's self-agreement, and
     bootstrap intervals for both. */
  function compute(cells, opts) {
    opts = opts || {};
    const reps = opts.bootstrap === undefined ? BOOTSTRAP : opts.bootstrap;
    const usable = cells.filter(c => !c.assigned && c.human !== null && c.human !== undefined);
    const excluded = cells.length - usable.length;

    const byCriterion = {};
    for (const c of usable) (byCriterion[c.criterion] = byCriterion[c.criterion] || []).push(c);

    const byItem = {};
    for (const c of usable) (byItem[c.item] = byItem[c.item] || []).push(c);

    const criteria = (opts.order || Object.keys(byCriterion))
      .filter(k => byCriterion[k] && byCriterion[k].length);

    const ci = reps ? bootstrap(usable, byItem, c => c.criterion, reps, SEED) : {};
    const ciPooled = reps ? bootstrap(usable, byItem, () => "pooled", reps, SEED + 1) : {};

    const self = selfRows(usable);
    const selfByCriterion = {};
    for (const r of self) (selfByCriterion[r.criterion] = selfByCriterion[r.criterion] || []).push(r);

    return {
      version: 1,
      n_items: Object.keys(byItem).length,
      n_cells: usable.length,
      n_excluded_assigned: excluded,
      criteria: criteria.map(k => Object.assign(
        {criterion: k}, stats(byCriterion[k]),
        {ci: ci[k] || {}, judge_self: stats(selfByCriterion[k] || [])})),
      pooled: Object.assign({criterion: "all criteria"}, stats(usable),
                            {ci: ciPooled["pooled"] || {}, judge_self: stats(self)}),
      // Kept out of the headline: these are the cells the judge was never asked about, so
      // they say what a human thinks of QUIS's trustworthiness and nothing about agreement.
      assigned: cells.filter(c => c.assigned && c.human !== null && c.human !== undefined)
        .map(c => ({item: c.item, criterion: c.criterion, human: c.human, judge: c.judge})),
    };
  }

  /* ------------------------------------------------------------------ rendering */

  const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)
    ? "—" : (d === 0 ? String(Math.round(v)) : v.toFixed(d === undefined ? 3 : d)));
  const pct = (v) => (v === null || v === undefined || Number.isNaN(v)
    ? "—" : (100 * v).toFixed(1) + "%");

  const LABELS = {
    goal_relevance: "Goal relevance", information_gain: "Information gain",
    interestingness: "Interestingness", trustworthiness: "Trustworthiness",
    actionability: "Actionability", resolution: "Resolution",
  };
  const label = (k) => LABELS[k] || k;

  const COLUMNS = [
    ["n", r => String(r.n)],
    ["human", r => fmt(r.human_mean, 2)],
    ["judge", r => fmt(r.judge_mean, 2)],
    ["exact", r => pct(r.exact)],
    ["chance", r => pct(r.exact_chance)],
    ["±1", r => pct(r.within1)],
    ["MAE", r => fmt(r.mae, 2)],
    ["r", r => fmt(r.pearson)],
    ["rho", r => fmt(r.spearman)],
    ["QWK", r => fmt(r.qwk)],
    ["alpha", r => fmt(r.alpha)],
    ["judge vs itself", r => (r.judge_self && r.judge_self.n
      ? pct(r.judge_self.exact) + " exact, " + fmt(r.judge_self.mae, 2) + " MAE" : "—")],
  ];

  function rowsOf(report) {
    return report.criteria.map(r => [label(r.criterion)].concat(COLUMNS.map(c => c[1](r))))
      .concat([["ALL CRITERIA"].concat(COLUMNS.map(c => c[1](report.pooled)))]);
  }

  function toText(report) {
    const head = ["criterion"].concat(COLUMNS.map(c => c[0]));
    const rows = [head].concat(rowsOf(report));
    const w = head.map((_, i) => Math.max(...rows.map(r => r[i].length)));
    const line = (r) => r.map((v, i) => (i ? v.padStart(w[i]) : v.padEnd(w[i]))).join("  ");
    const out = [line(head), w.map(n => "-".repeat(n)).join("  ")]
      .concat(rows.slice(1).map(line));
    out.push("");
    out.push(`${report.n_items} trajectories, ${report.n_cells} ratings`
      + (report.n_excluded_assigned
        ? `, ${report.n_excluded_assigned} excluded (assigned by policy, not judged)` : ""));
    return out.join("\n");
  }

  function toMarkdown(report) {
    const head = ["criterion"].concat(COLUMNS.map(c => c[0]));
    const rows = rowsOf(report);
    return [`| ${head.join(" | ")} |`,
            `|${head.map(() => "---").join("|")}|`]
      .concat(rows.map(r => `| ${r.join(" | ")} |`)).join("\n");
  }

  /* One CSV number. Nine decimals rather than the default stringification: JavaScript
     prints 1 where Python prints 1.0, and agreement.py has to emit the same bytes. */
  const csvNum = (v) => (v === null || v === undefined || Number.isNaN(v) ? "" : v.toFixed(9));

  function toCSV(report) {
    const head = ["criterion", "n", "human_mean", "judge_mean", "bias", "exact",
                  "exact_chance", "within1", "within1_chance", "mae", "pearson",
                  "spearman", "qwk", "alpha", "judge_self_exact", "judge_self_mae"];
    const line = (r) => [r.criterion, String(r.n)].concat(
      [r.human_mean, r.judge_mean, r.bias, r.exact, r.exact_chance, r.within1,
       r.within1_chance, r.mae, r.pearson, r.spearman, r.qwk, r.alpha,
       (r.judge_self || {}).exact, (r.judge_self || {}).mae].map(csvNum))
      .join(",");
    return [head.join(",")].concat(report.criteria.map(line)).concat([line(report.pooled)]).join("\n");
  }

  root.TrajAgreement = {compute, stats, toText, toCSV, toMarkdown, label,
                        pearson, spearman, qwk, alphaInterval, ranks, prng, COLUMNS};
})(typeof module !== "undefined" && module.exports ? module.exports : window);
