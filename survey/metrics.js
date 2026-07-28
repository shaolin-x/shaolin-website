/* Scoring for the survey's completion page and download.
 *
 * The metric *definitions* do not live here. `build_survey.py` calls
 * `human_trajectory_eval.pick_row` once per candidate slot at build time and publishes the
 * result in `truth.json`, so a pick is scored by looking its slot up — this file never
 * evaluates a scoring formula and so cannot drift from the Python that defines them.
 *
 * What it does reimplement is the four estimators that reduce those rows to a number
 * (kappa, weighted mean, tie-corrected AUROC, Youden cut) plus the cluster bootstrap, all
 * of which depend on the respondent's own answers and so cannot be precomputed. Those are
 * ported line-for-line from `human_trajectory_eval.py`, down to a shared PRNG, and
 * `survey/tests/` asserts the two agree to 1e-9 on every synthetic respondent.
 *
 * Loaded as a plain script (`window.SurveyMetrics`) rather than an ES module so the node
 * test runner can `require` it without the page being involved.
 */
(function (root) {
  "use strict";

  // Bumped in lockstep with hte.METRICS_VERSION. `survey.json` carries the build's value;
  // a mismatch means the page is being served against a bundle built by a different
  // definition of the metrics, which is reported rather than silently averaged.
  var VERSION = "2";

  /* ------------------------------------------------------------------ estimators */

  function mean(xs) {
    if (!xs.length) return null;
    var s = 0;
    for (var i = 0; i < xs.length; i++) s += xs[i];
    return s / xs.length;
  }

  /* Chance-corrected agreement with a per-item baseline: (a - e) / (1 - e). The expected
     rate varies by decision because pools differ in size and in how many candidates share
     the top score, so a single scalar baseline would not be comparable across them. */
  function kappa(obs, exp) {
    var a = mean(obs), e = mean(exp);
    if (a === null || e === null || e >= 1.0) return null;
    return (a - e) / (1.0 - e);
  }

  /* Weight-normalised mean, falling back to the plain mean when every weight is zero —
     the limit as the weights vanish, rather than 0/0. */
  function wmean(vals, weights) {
    if (!vals.length) return null;
    var total = 0, acc = 0;
    for (var i = 0; i < weights.length; i++) total += weights[i];
    if (total <= 0) return mean(vals);
    for (var j = 0; j < vals.length; j++) acc += vals[j] * weights[j];
    return acc / total;
  }

  /* Tie-corrected AUC (Mann-Whitney, half credit for ties). The judge's utilities are
     heavily quantised, so the tie term is load-bearing rather than a rounding detail. */
  function auroc(scores, labels) {
    var pos = [], neg = [], i;
    for (i = 0; i < scores.length; i++) (labels[i] > 0 ? pos : neg).push(scores[i]);
    if (!pos.length || !neg.length) return null;
    var wins = 0;
    for (i = 0; i < pos.length; i++) {
      for (var j = 0; j < neg.length; j++) {
        wins += pos[i] > neg[j] ? 1.0 : pos[i] === neg[j] ? 0.5 : 0.0;
      }
    }
    return wins / (pos.length * neg.length);
  }

  /* The cut on `scores` maximising balanced agreement, plus the interval of cuts tying for
     it. "Continue" is predicted when score >= cut. Quantised utilities mean a bare argmax
     would be an artefact of which candidate cut was tried first. */
  function youden(scores, labels) {
    var pos = [], neg = [], i;
    for (i = 0; i < scores.length; i++) (labels[i] > 0 ? pos : neg).push(scores[i]);
    if (!pos.length || !neg.length) return null;
    var lo = Math.min.apply(null, scores), hi = Math.max.apply(null, scores);
    var seen = {}, cuts = [lo - 1e-9, hi + 1e-9].concat(scores);
    var uniq = [];
    for (i = 0; i < cuts.length; i++) {
      if (!Object.prototype.hasOwnProperty.call(seen, cuts[i])) {
        seen[cuts[i]] = 1; uniq.push(cuts[i]);
      }
    }
    uniq.sort(function (a, b) { return a - b; });
    var best = -2.0, keep = [];
    for (i = 0; i < uniq.length; i++) {
      var cut = uniq[i], p = 0, n = 0, k;
      for (k = 0; k < pos.length; k++) if (pos[k] >= cut) p++;
      for (k = 0; k < neg.length; k++) if (neg[k] >= cut) n++;
      var j = p / pos.length - n / neg.length;
      if (j > best + 1e-12) { best = j; keep = [cut]; }
      else if (Math.abs(j - best) <= 1e-12) keep.push(cut);
    }
    return [mean(keep), Math.min.apply(null, keep), Math.max.apply(null, keep)];
  }

  /* Wilson score interval — used for the plain rates, where it beats the bootstrap at
     small n and needs no resampling. */
  function wilson(k, n, z) {
    if (!n) return [null, null];
    z = z || 1.959963984540054;
    var p = k / n, d = 1 + z * z / n;
    var centre = (p + z * z / (2 * n)) / d;
    var half = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, centre - half), Math.min(1, centre + half)];
  }

  /* Byte-for-byte the PRNG in human_trajectory_eval.mulberry32, so a bootstrap run here
     and one run there return the same interval for the same answers. */
  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
      t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  var BOOTSTRAP_DRAWS = 2000;
  var BOOTSTRAP_SEED = 20260728;

  /* Percentile CI resampling *clusters*, not rows: two decisions can descend from one
     trajectory (one judge call, two questions), so treating them as independent would
     report an interval that is too narrow. Percentile rather than BCa — see the note in
     human_trajectory_eval.py. */
  function clusterBootstrap(rows, stat, seed, draws, alpha) {
    if (!rows.length) return [null, null];
    seed = seed === undefined ? BOOTSTRAP_SEED : seed;
    draws = draws || BOOTSTRAP_DRAWS;
    alpha = alpha === undefined ? 0.05 : alpha;
    var groups = {}, i;
    for (i = 0; i < rows.length; i++) {
      var key = String(rows[i].cluster !== undefined && rows[i].cluster !== null
        ? rows[i].cluster : rows[i].id);
      (groups[key] = groups[key] || []).push(rows[i]);
    }
    var keys = Object.keys(groups).sort();
    if (keys.length < 2) return [null, null];
    var rnd = mulberry32(seed), got = [];
    for (var d = 0; d < draws; d++) {
      var sample = [];
      for (i = 0; i < keys.length; i++) {
        var pickKey = keys[Math.floor(rnd() * keys.length) % keys.length];
        sample = sample.concat(groups[pickKey]);
      }
      var value = stat(sample);
      if (value !== null && value !== undefined && !isNaN(value)) got.push(value);
    }
    if (got.length < Math.floor(draws / 10)) return [null, null];
    got.sort(function (a, b) { return a - b; });
    function pick(q) {
      return got[Math.min(got.length - 1,
        Math.max(0, Math.floor(q * (got.length - 1) + 0.5)))];
    }
    return [pick(alpha / 2), pick(1 - alpha / 2)];
  }

  /* ------------------------------------------------------------------ rows */

  /* `{picks, stops, review, skipped}` for one response.
   *
   * A pick row is `truth[id].picks[slot]` verbatim — every scoring decision in it was made
   * by Python at build time. A stop row is assembled here only because it needs the
   * human's own answer, and mirrors human_trajectory_eval.stop_row exactly.
   */
  function rowsFor(verdicts, truth, decisions) {
    var picks = [], stops = [], review = [], skipped = 0;
    var byId = {};
    for (var i = 0; i < (decisions || []).length; i++) byId[decisions[i].id] = decisions[i];

    // Last answer wins, mirroring score_survey.py. The page cannot produce a duplicate but
    // a hand-edited response re-imported can, and both would otherwise be counted.
    var latest = {};
    for (i = 0; i < verdicts.length; i++) latest[verdicts[i].decision] = verdicts[i];
    var ordered = [];
    for (i = 0; i < verdicts.length; i++) {
      if (latest[verdicts[i].decision] === verdicts[i]) ordered.push(verdicts[i]);
    }

    for (i = 0; i < ordered.length; i++) {
      var v = ordered[i], t = truth && truth[v.decision], d = byId[v.decision];
      if (v.slot === null && v.terminate !== true) skipped++;
      if (!t) continue;

      if (v.terminate !== null && v.terminate !== undefined
          && t.stop && t.stop.model !== null && t.stop.model !== undefined) {
        var s = t.stop, human = !!v.terminate;
        var usable = s.margin !== null && s.margin !== undefined;
        stops.push({
          id: v.decision, run: s.run, depth: s.depth,
          cluster: s.cluster || v.decision, source: s.source,
          model: !!s.model, human: human,
          agree: !!s.model === human ? 1.0 : 0.0,
          human_continue: human ? 0.0 : 1.0,
          utility: usable ? s.utility : null,
          threshold: usable ? s.threshold : null,
          margin: usable ? s.margin : null,
          criteria: usable ? (s.criteria || null) : null
        });
      }

      var mine = null, top = null;
      if (v.slot !== null && v.slot !== undefined && t.picks) {
        var row = t.picks[String(v.slot)];
        if (row) picks.push(row);
        mine = findBy(t.rows, "slot", v.slot);
        top = findBy(t.rows, "rank", 1);
      }
      review.push({ d: d, v: v, t: t, mine: mine, top: top });
    }
    return { picks: picks, stops: stops, review: review, skipped: skipped };
  }

  function findBy(list, key, want) {
    for (var i = 0; i < (list || []).length; i++) if (list[i][key] === want) return list[i];
    return null;
  }

  /* ------------------------------------------------------------------ detail */

  /* One fully-resolved record per decision, for the downloaded file.
   *
   * A response on its own says "on decision X I picked slot 3", which is enough to score
   * against the private bundle and useless to read. This joins the three halves the page
   * already holds — the blinded decision, the answer key and the respondent's verdict —
   * into something self-contained: where the trajectory came from, what the candidates
   * were, what D2I scored each of them and which one the human took.
   *
   * Nothing here is new information. It is assembled at the close, once every answer is
   * locked in, out of files the page had already fetched.
   */
  function detail(verdicts, truth, decisions, runs, termNames) {
    var byId = {}, i;
    for (i = 0; i < (decisions || []).length; i++) byId[decisions[i].id] = decisions[i];
    var latest = {};
    for (i = 0; i < verdicts.length; i++) latest[verdicts[i].decision] = verdicts[i];

    var out = [];
    for (i = 0; i < (decisions || []).length; i++) {
      var d = decisions[i], v = latest[d.id], t = truth && truth[d.id];
      if (!t) continue;
      var stop = t.stop || {};
      var bySlot = {};
      for (var k = 0; k < (t.rows || []).length; k++) bySlot[t.rows[k].slot] = t.rows[k];

      var cands = (d.candidates || []).map(function (c) {
        var row = bySlot[c.slot] || {};
        var breakdown = {};
        for (var j = 0; j < (termNames || []).length; j++) {
          breakdown[termNames[j]] = row.terms ? row.terms[j] : null;
        }
        return {
          slot: c.slot,
          action: c.action,
          question: c.question,
          model_rank: row.rank === undefined ? null : row.rank,
          model_score: row.score === undefined ? null : row.score,
          score_breakdown: breakdown,
          model_top: row.rank === 1,
          answered_by_model: !!row.answered,
          picked_by_human: v ? v.slot === c.slot : false
        };
      });
      var top = cands.filter(function (c) { return c.model_top; })[0] || null;
      var human_stop = v && v.terminate !== null && v.terminate !== undefined ? !!v.terminate : null;

      out.push({
        id: d.id,
        // Where this trajectory came from — run, file, node and depth.
        source: {
          run: d.run,
          run_dir: (runs && runs[d.run] && runs[d.run].run_dir) || null,
          run_group: (runs && runs[d.run] && runs[d.run].parent) || null,
          path: (runs && runs[d.run] && runs[d.run].path) || null,
          model: (runs && runs[d.run] && runs[d.run].model) || null,
          node: d.node,
          depth: d.depth,
          candidate_depth: d.child_depth,
          trajectory: stop.cluster || null
        },
        // `goal` and `columns` are not repeated here — they are per *run*, identical across
        // every decision drawn from it, and sit in `results.runs[source.run]`.
        // The trajectory as the respondent saw it: every step from the base insight down.
        trajectory_so_far: d.path,
        answered: !!v,
        skipped: !!(v && v.slot === null && v.terminate !== true),
        note: (v && v.note) || "",
        termination: !d.ask_stop ? null : {
          asked: true,
          model_terminate: t.model_terminate,
          human_terminate: human_stop,
          agree: human_stop === null || t.model_terminate === null
            ? null : (t.model_terminate === human_stop),
          // Present only where the judge actually ruled; `source` says which.
          verdict_source: stop.source || null,
          continuation_utility: stop.utility === undefined ? null : stop.utility,
          threshold: stop.threshold === undefined ? null : stop.threshold,
          margin: stop.margin === undefined ? null : stop.margin,
          criteria: stop.criteria || null,
          model_rationale: t.terminate_reason || ""
        },
        selection: !cands.length ? null : {
          asked: true,
          n_candidates: cands.length,
          // Slots, not copies: both candidates are in the list below, flagged
          // `picked_by_human` and `model_top`.
          human_slot: v ? v.slot : null,
          model_top_slot: top ? top.slot : null,
          candidates: cands,
          // The per-decision metric row, so each figure in the summary can be traced back
          // to the decisions that produced it.
          metrics: (v && v.slot !== null && t.picks) ? (t.picks[String(v.slot)] || null) : null
        }
      });
    }
    return out;
  }

  /* How much of each kind of question was actually answered. Reported explicitly because
     the two families run over different subsets — a terminated node has no candidates, and
     a node the judge never ruled on has no stop question. */
  function counts(details, rows) {
    var trajectories = {}, candidatesShown = 0;
    var sel = {asked: 0, answered: 0, skipped: 0};
    var term = {asked: 0, answered: 0, judged: 0, inferred: 0};
    for (var i = 0; i < details.length; i++) {
      var d = details[i];
      if (d.source.trajectory) trajectories[d.source.trajectory] = 1;
      if (d.selection) {
        sel.asked++;
        candidatesShown += d.selection.n_candidates;
        if (d.selection.human_slot !== null && d.selection.human_slot !== undefined) sel.answered++;
        else sel.skipped++;
      }
      if (d.termination) {
        term.asked++;
        if (d.termination.human_terminate !== null) term.answered++;
        if (d.termination.margin !== null && d.termination.margin !== undefined) term.judged++;
        else term.inferred++;
      }
    }
    return {
      decisions: details.length,
      distinct_trajectories: Object.keys(trajectories).length,
      candidates_shown: candidatesShown,
      candidate_selection: sel,
      termination_selection: term,
      scored_picks: (rows.picks || []).length,
      scored_stops: (rows.stops || []).length
    };
  }

  /* The metric table keyed by a short stable name, so a reader does not have to match on
     display labels that may be reworded. */
  var HEADLINE_KEYS = [
    ["kappa_sel", "κ_sel"], ["raw_agreement", "raw agreement"],
    ["utility_attainment", "attainment"], ["standardized_regret", "standardized regret"],
    ["within_noise_rate", "within-noise"], ["action_agreement", "action agreement"],
    ["margin_weighted_agreement", "margin-weighted"], ["unweighted_agreement", "unweighted"],
    ["auroc", "AUROC"], ["implied_threshold", "implied human threshold"]
  ];

  function headlines(agg) {
    var out = {};
    for (var i = 0; i < HEADLINE_KEYS.length; i++) {
      var name = HEADLINE_KEYS[i][0], needle = HEADLINE_KEYS[i][1];
      for (var j = 0; j < agg.length; j++) {
        if (agg[j].label.indexOf(needle) >= 0) {
          out[name] = {
            value: agg[j].value, chance: agg[j].chance,
            ci: [agg[j].lo, agg[j].hi], n: agg[j].n,
            estimable: estimable(agg[j])
          };
          break;
        }
      }
    }
    return out;
  }

  /* ------------------------------------------------------------------ aggregate */

  /* `(value, chance, n)` for one METRIC_TABLE entry — the port of hte.estimate. */
  function estimate(spec, rows) {
    var only = spec.only, sel = [], i;
    for (i = 0; i < rows.length; i++) {
      var r = rows[i];
      var passOnly = only === null || only === undefined || r[only] !== null && r[only] !== undefined;
      if (passOnly && r[spec.key] !== null && r[spec.key] !== undefined) sel.push(r);
    }
    if (!sel.length) return { value: null, chance: null, n: 0, sel: sel };

    var vals = sel.map(function (r) { return Number(r[spec.key]); });
    var base = null;
    if (spec.base) {
      var ok = true;
      for (i = 0; i < sel.length; i++) {
        if (sel[i][spec.base] === null || sel[i][spec.base] === undefined) { ok = false; break; }
      }
      if (ok) base = sel.map(function (r) { return Number(r[spec.base]); });
    }
    var est = spec.est || "mean", value = null, chance = null;

    if (est === "mean") { value = mean(vals); chance = base ? mean(base) : null; }
    else if (est === "kappa") { value = kappa(vals, base || []); chance = 0.0; }
    else if (est === "wmean") {
      value = wmean(vals, sel.map(function (r) { return Math.abs(Number(r.margin)); }));
    } else if (est === "auroc") {
      value = auroc(vals, sel.map(function (r) { return Number(r.human_continue); }));
      chance = 0.5;
    } else if (est === "youden") {
      // Not identifiable when every utility sits on one side of D2I's own threshold:
      // there is no evidence about where the human would have cut on the other side.
      var taus = [];
      for (i = 0; i < sel.length; i++) {
        if (sel[i].threshold !== null && sel[i].threshold !== undefined) taus.push(Number(sel[i].threshold));
      }
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      if (taus.length && (hi < Math.min.apply(null, taus) || lo >= Math.max.apply(null, taus))) {
        value = null;
      } else {
        var got = youden(vals, sel.map(function (r) { return Number(r.human_continue); }));
        value = got ? got[0] : null;
      }
    } else {
      throw new Error("unknown estimator " + est);
    }
    return { value: value, chance: chance, n: sel.length, sel: sel };
  }

  /* The full table as data: one entry per METRIC_TABLE spec, with a cluster-bootstrap CI. */
  function aggregate(rows, table, seed) {
    var out = [];
    for (var i = 0; i < table.length; i++) {
      var spec = table[i];
      var src = spec.group === "selection" ? rows.picks : rows.stops;
      var got = estimate(spec, src);
      var lo = null, hi = null;
      if (got.value !== null) {
        var ci = clusterBootstrap(src, function (rs) { return estimate(spec, rs).value; }, seed);
        lo = ci[0]; hi = ci[1];
      }
      out.push({
        key: spec.key, label: spec.label, group: spec.group, fmt: spec.fmt,
        est: spec.est || "mean", value: got.value, chance: got.chance,
        lo: lo, hi: hi, n: got.n
      });
    }
    return out;
  }

  /* Component attribution: the two headline estimators recomputed against each single
     score term's ranking, so the table stays one-dimensional. */
  function termAttribution(picks, terms) {
    var out = [];
    for (var i = 0; i < terms.length; i++) {
      var short = terms[i], sel = [];
      for (var j = 0; j < picks.length; j++) {
        if (picks[j].terms && picks[j].terms[short]) sel.push(picks[j].terms[short]);
      }
      if (!sel.length) { out.push({ term: short, n: 0, kappa: null, attainment: null }); continue; }
      out.push({
        term: short, n: sel.length,
        kappa: kappa(sel.map(function (s) { return s.is_top; }),
                     sel.map(function (s) { return s.e_top; })),
        attainment: mean(sel.map(function (s) { return s.attainment; })),
        e_attainment: mean(sel.map(function (s) { return s.e_attainment; }))
      });
    }
    return out;
  }

  /* Continuation attribution: AUROC of each judge criterion against the human's label,
     for comparison with the AUROC of the aggregate utility. */
  function criterionAttribution(stops) {
    var sel = stops.filter(function (s) { return s.criteria; });
    var names = [], i, k;
    for (i = 0; i < sel.length; i++) {
      for (k in sel[i].criteria) {
        if (Object.prototype.hasOwnProperty.call(sel[i].criteria, k) && names.indexOf(k) < 0) {
          names.push(k);
        }
      }
    }
    var labels = sel.map(function (s) { return Number(s.human_continue); });
    return names.map(function (name) {
      var vals = sel.map(function (s) { return s.criteria[name]; });
      for (var j = 0; j < vals.length; j++) {
        if (typeof vals[j] !== "number") return { criterion: name, n: 0, auroc: null };
      }
      return { criterion: name, n: sel.length, auroc: auroc(vals, labels) };
    });
  }

  /* `{labels, matrix}` with matrix[i][j] = rows whose rowKey is labels[i] and colKey
     labels[j]. Serves both the action matrix and the 2x2 stop one. */
  function confusion(rows, rowKey, colKey, labels) {
    if (!labels) {
      var seen = {};
      for (var i = 0; i < rows.length; i++) {
        seen[String(rows[i][rowKey])] = 1; seen[String(rows[i][colKey])] = 1;
      }
      labels = Object.keys(seen).sort();
    }
    var index = {};
    labels.forEach(function (l, i) { index[l] = i; });
    var m = labels.map(function () { return labels.map(function () { return 0; }); });
    for (i = 0; i < rows.length; i++) {
      var a = index[String(rows[i][rowKey])], b = index[String(rows[i][colKey])];
      if (a !== undefined && b !== undefined) m[a][b]++;
    }
    return { labels: labels, matrix: m };
  }

  /* ------------------------------------------------------------------ rendering */

  function fmtValue(x, fmt) {
    if (x === null || x === undefined || isNaN(x)) return "—";
    return fmt === "pct" ? (100 * x).toFixed(1) + "%" : x.toFixed(3);
  }

  function fmtCI(row) {
    if (row.lo === null || row.lo === undefined || row.hi === null || row.hi === undefined) return "—";
    return fmtValue(row.lo, row.fmt) + " – " + fmtValue(row.hi, row.fmt);
  }

  /* Whether a metric has enough data to mean anything for a single respondent. Below these
     the cell renders as "not estimable" rather than a number nobody should read. */
  var MIN_N = { auroc: 4, youden: 6, kappa: 4, wmean: 3, mean: 1 };

  function estimable(row) {
    return row.value !== null && row.n >= (MIN_N[row.est] || 1);
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  var GROUP_TITLE = { selection: "Selection alignment", continuation: "Continuation alignment" };

  function renderTable(agg) {
    var html = '<div class="scroll"><table><thead><tr><th>metric</th><th>you</th>'
             + "<th>chance</th><th>95% CI</th><th>n</th></tr></thead><tbody>";
    var group = null;
    for (var i = 0; i < agg.length; i++) {
      var r = agg[i];
      if (r.group !== group) {
        group = r.group;
        html += '<tr><th colspan="5" class="grp">' + esc(GROUP_TITLE[group] || group) + "</th></tr>";
      }
      var cell = estimable(r)
        ? '<td class="mono">' + fmtValue(r.value, r.fmt) + "</td>"
          + '<td class="mono">' + fmtValue(r.chance, r.fmt) + "</td>"
          + '<td class="mono">' + fmtCI(r) + "</td>"
        : '<td class="mono" colspan="3">— not estimable at n=' + r.n + "</td>";
      html += "<tr><td>" + esc(r.label) + "</td>" + cell + '<td class="mono">' + r.n + "</td></tr>";
    }
    return html + "</tbody></table></div>";
  }

  function renderMatrix(conf, rowHead, colHead) {
    if (!conf.labels.length) return "";
    var html = '<div class="scroll"><table><thead><tr><th>' + esc(rowHead) + " \\ " + esc(colHead)
             + "</th>";
    conf.labels.forEach(function (l) { html += "<th>" + esc(l) + "</th>"; });
    html += "</tr></thead><tbody>";
    conf.labels.forEach(function (l, i) {
      html += "<tr><td>" + esc(l) + "</td>";
      conf.matrix[i].forEach(function (v, j) {
        html += '<td class="mono' + (i === j && v ? " ok" : "") + '">' + v + "</td>";
      });
      html += "</tr>";
    });
    return html + "</tbody></table></div>";
  }

  /* A fixed-width copy of the table, for the download and the clipboard. */
  function toText(agg) {
    var lines = ["metric".padEnd(36) + "you".padStart(9) + "chance".padStart(9)
                 + "95% CI".padStart(20) + "n".padStart(5), "-".repeat(79)];
    var group = null;
    for (var i = 0; i < agg.length; i++) {
      var r = agg[i];
      if (r.group !== group) { group = r.group; lines.push("-- " + (GROUP_TITLE[group] || group)); }
      lines.push(r.label.padEnd(36)
        + (estimable(r) ? fmtValue(r.value, r.fmt) : "—").padStart(9)
        + (estimable(r) ? fmtValue(r.chance, r.fmt) : "—").padStart(9)
        + (estimable(r) ? fmtCI(r) : "—").padStart(20)
        + String(r.n).padStart(5));
    }
    return lines.join("\n");
  }

  function toMarkdown(agg) {
    var lines = ["| metric | you | chance | 95% CI | n |", "| --- | --- | --- | --- | --- |"];
    var group = null;
    for (var i = 0; i < agg.length; i++) {
      var r = agg[i];
      if (r.group !== group) {
        group = r.group;
        lines.push("| **" + (GROUP_TITLE[group] || group) + "** | | | | |");
      }
      lines.push("| " + r.label + " | " + (estimable(r) ? fmtValue(r.value, r.fmt) : "—")
        + " | " + (estimable(r) ? fmtValue(r.chance, r.fmt) : "—")
        + " | " + (estimable(r) ? fmtCI(r) : "—") + " | " + r.n + " |");
    }
    return lines.join("\n");
  }

  function toCSV(agg, rows) {
    function q(v) {
      if (v === null || v === undefined) return "";
      var s = String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }
    var out = ["metric,group,value,chance,ci_lo,ci_hi,n"];
    agg.forEach(function (r) {
      out.push([r.label, r.group, r.value, r.chance, r.lo, r.hi, r.n].map(q).join(","));
    });
    out.push("");
    out.push("decision,run,depth,cluster,is_top,e_top,attainment,e_attainment,d_std,pool_sd,"
      + "top2_gap,n,pick_action,top_action");
    (rows.picks || []).forEach(function (r) {
      out.push([r.id, r.run, r.depth, r.cluster, r.is_top, r.e_top, r.attainment,
        r.e_attainment, r.d_std, r.pool_sd, r.top2_gap, r.n, r.pick_action,
        r.top_action].map(q).join(","));
    });
    out.push("");
    out.push("decision,run,depth,cluster,source,model_terminate,human_terminate,agree,utility,"
      + "threshold,margin");
    (rows.stops || []).forEach(function (r) {
      out.push([r.id, r.run, r.depth, r.cluster, r.source, r.model, r.human, r.agree,
        r.utility, r.threshold, r.margin].map(q).join(","));
    });
    return out.join("\n");
  }

  var SurveyMetrics = {
    VERSION: VERSION,
    mean: mean, kappa: kappa, wmean: wmean, auroc: auroc, youden: youden, wilson: wilson,
    mulberry32: mulberry32, clusterBootstrap: clusterBootstrap,
    rowsFor: rowsFor, estimate: estimate, aggregate: aggregate,
    detail: detail, counts: counts, headlines: headlines,
    termAttribution: termAttribution, criterionAttribution: criterionAttribution,
    confusion: confusion, estimable: estimable,
    fmtValue: fmtValue, fmtCI: fmtCI,
    renderTable: renderTable, renderMatrix: renderMatrix,
    toText: toText, toMarkdown: toMarkdown, toCSV: toCSV
  };

  root.SurveyMetrics = SurveyMetrics;
  if (typeof module !== "undefined" && module.exports) module.exports = SurveyMetrics;
})(typeof globalThis !== "undefined" ? globalThis : this);
