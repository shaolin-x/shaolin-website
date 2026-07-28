#!/usr/bin/env python
"""Browser front end for the human trajectory evaluation — clickable, one file.

Same evaluation as `eval/human_trajectory_eval.py`, same session.json, same metrics: this
only replaces the keyboard prompts with a page. Every bit of judging logic — loading runs,
reconstructing decision points, sampling, blinding, scoring, the report — is imported from
that module, so the two front ends cannot drift apart. A session started here can be
finished on the CLI with `--resume`, and vice versa.

Started with no arguments it opens on a picker: every folder under `runs/`, then the run
dirs inside the one you choose (`carsales-easy`, `data-55`, …), which you tick before
setting the sample size and starting. Give a path on the command line to skip it.

    python eval/human_trajectory_web.py                       # pick in the browser
    python eval/human_trajectory_web.py runs/20260727-201135_d2i_bench/carsales-easy 5
    python eval/human_trajectory_web.py --resume eval/human_trajectory_eval/<stamp>_<run>/session.json

It serves on 127.0.0.1 only and opens a browser tab. The page is a string constant below —
no template files, no CDN, no build step. Ctrl-C stops the server; the session is written
after every answer, so nothing is lost.

Blinding is enforced server-side, not by the page: `public_decision` strips the scores, the
statuses and the model's verdict, so a judge with devtools open sees exactly what a judge
without them sees. The truth for one decision is returned only in the response to that
decision's own answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))          # so the sibling evaluators import as modules

import human_trajectory_eval as hte         # noqa: E402

DEFAULT_PORT = 8765


# ----------------------------------------------------------------- browsing runs/

def groups(root: Path) -> list[dict]:
    """The top level of the picker: every folder under `runs/`, newest first.

    `n_runs` counts the run dirs inside — the flat layout (report.json sitting directly in
    the timestamped folder) counts as one, the benchmark layouts count their per-dataset
    children. A shallow look is enough for every layout on disk and keeps the listing
    instant even with a hundred folders.
    """
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if (child / "report.json").is_file():
            n, kind = 1, "run"
        else:
            n = sum(1 for g in child.iterdir() if g.is_dir() and (g / "report.json").is_file())
            kind = "group"
        out.append({"name": child.name, "path": str(child), "kind": kind, "n_runs": n})
    return out


def runs_in(group: Path) -> list[dict]:
    """The second level: the run dirs inside one folder, each with why it can or cannot be
    judged. The check is the real loader, so a folder listed as ready always yields a
    session."""
    dirs = [group] if (group / "report.json").is_file() else [
        d for d in sorted(group.iterdir()) if d.is_dir() and (d / "report.json").is_file()
    ]
    out = []
    for d in dirs:
        loaded = hte.load_run(d)
        ok = not isinstance(loaded, str)
        n_nodes = 0
        if ok:
            points, _, _ = hte.decision_points(d, *loaded, 2, False)
            n_nodes = len(points)
            if not n_nodes:
                ok, loaded = False, "no node offers a choice"
        out.append({
            "name": d.name, "path": str(d), "usable": ok and n_nodes > 0,
            "nodes": n_nodes, "reason": "" if ok else str(loaded),
        })
    return out


# ----------------------------------------------------------------- what the page may see

def public_decision(decision: dict, index: int, total: int) -> dict:
    """One decision with every tell removed.

    Candidates go out in *presentation* order (the blinded shuffle), carrying only their
    action and question — no score, no status, no breakdown, and no `d2i_terminate`. What
    survives is exactly what the terminal screen prints before an answer.
    """
    return {
        "id": decision["id"],
        "index": index,
        "total": total,
        "run": decision["run"],
        "node": decision["node_id"][:8],
        "depth": decision["depth"],
        "child_depth": decision["child_depth"],
        "goal": decision["goal"],
        "columns": decision["columns"],
        "path": decision["path"],
        # Whether the stop question applies at all — not what its answer is.
        "ask_stop": decision.get("d2i_terminate") is not None,
        "candidates": [
            {"slot": slot, "action": decision["candidates"][t]["action"],
             "question": decision["candidates"][t]["question"]}
            for slot, t in enumerate(decision["order"], 1)
        ],
    }


def reveal_payload(decision: dict, verdict: dict) -> dict:
    """The truth for one decision, released only once it has been answered."""
    slot_of = {t: k for k, t in enumerate(decision["order"], 1)}
    cands = decision["candidates"]
    rows = [
        {
            "rank": 1 + sum(1 for o in cands if o["score"] > c["score"]),
            "slot": slot_of[i],
            "score": c["score"],
            "answered": c["status"] in hte.SELECTED,
            "picked": i == verdict.get("true_index"),
            "action": c["action"],
            "question": c["question"],
            "terms": [(c.get("breakdown") or {}).get(key) for key, _ in hte.TERMS],
        }
        for i, c in enumerate(cands)
    ]
    picked = None if verdict.get("true_index") is None else cands[verdict["true_index"]]
    return {
        "terms": [short for _, short in hte.TERMS],
        "lambdas": decision.get("lambdas"),
        "model_terminate": decision.get("d2i_terminate"),
        "terminate_reason": decision.get("terminate_reason") or "",
        "human_terminate": verdict.get("terminate"),
        "rank": None if picked is None
                else 1 + sum(1 for c in cands if c["score"] > picked["score"]),
        "n": len(cands),
        "rows": rows,
    }


def state(ctx: dict) -> dict:
    """Everything the page needs: whether a session exists, progress, next decision."""
    session = ctx.get("session")
    if session is None:
        return {"started": False, "root": str(ctx["root"])}
    judged = {v["decision"] for v in session["verdicts"]}
    decisions = session["decisions"]
    nxt = next(((i, d) for i, d in enumerate(decisions) if d["id"] not in judged), None)
    return {
        "started": True,
        "judge": session.get("judge"),
        "path": session["path"],
        "feedback": bool(session.get("feedback", True)),
        "notes": session.get("notes") or [],
        "done": len(judged),
        "total": len(decisions),
        "can_undo": bool(session["verdicts"]),
        "decision": None if nxt is None else public_decision(nxt[1], nxt[0], len(decisions)),
    }


def start_session(ctx: dict, body: dict) -> None:
    """Build a session from the picker's selection and install it on the context."""
    paths = [Path(p) for p in body.get("paths") or []]
    if not paths:
        raise ValueError("pick at least one run")
    n = max(1, int(body.get("n") or 10))
    seed = body.get("seed")
    seed = random.randrange(2 ** 31) if seed in (None, "") else int(seed)
    decisions, notes = hte.build_decisions(
        paths, n, random.Random(seed), ctx["min_candidates"], ctx["drop_unanswerable"]
    )
    if not decisions:
        raise ValueError("no decision points in that selection")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = hte._UNSAFE.sub("-", body.get("label") or paths[0].name) or "runs"
    session = {
        "stamp": stamp,
        "judge": (body.get("judge") or "").strip() or None,
        "path": ", ".join(str(p) for p in paths),
        "runs": sorted({d["run_dir"] for d in decisions}),
        "n_requested": n,
        "min_candidates": ctx["min_candidates"],
        "drop_unanswerable": ctx["drop_unanswerable"],
        "feedback": bool(body.get("feedback", ctx["feedback"])),
        "seed": seed,
        "notes": notes,
        "decisions": decisions,
        "verdicts": [],
    }
    ctx["session"] = session
    ctx["by_id"] = {d["id"]: d for d in decisions}
    ctx["session_path"] = (ctx["session_dir"] or (hte.OUT_DIR / f"{stamp}_{label}")) / "session.json"
    hte.save_session(session, ctx["session_path"])
    for t in notes:
        print(t)
    print(f"session: {ctx['session_path']}   ({len(decisions)} decisions)")


# ----------------------------------------------------------------- the server

def make_handler(ctx: dict):
    """A request handler bound to one live context (which may not hold a session yet)."""
    lock = threading.Lock()

    def finish() -> list[str]:
        lines = hte.report(ctx["session"])
        out = ctx["session_path"].parent / "results.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if ctx["out_file"] is not None:
            ctx["out_file"].parent.mkdir(parents=True, exist_ok=True)
            ctx["out_file"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:          # keep the console for the run's own output
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8", code)

        def do_GET(self) -> None:                    # noqa: N802 (http.server's spelling)
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                with lock:
                    self._json(state(ctx))
            elif url.path == "/api/runs":
                group = (parse_qs(url.query).get("group") or [""])[0]
                with lock:
                    if group:
                        self._json({"group": group, "runs": runs_in(Path(group))})
                    else:
                        self._json({"root": str(ctx["root"]), "groups": groups(ctx["root"])})
            elif url.path == "/api/report":
                with lock:
                    if ctx.get("session") is None:
                        self._json({"error": "no session"}, 400)
                        return
                    self._json({"lines": finish(),
                                "results": str(ctx["session_path"].parent / "results.txt")})
            else:
                self.send_error(404)

        def do_POST(self) -> None:                   # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json({"error": "bad json"}, 400)
                return

            with lock:
                if self.path == "/api/start":
                    try:
                        start_session(ctx, body)
                    except (ValueError, OSError) as e:
                        self._json({"error": str(e)}, 400)
                        return
                    self._json(state(ctx))
                    return

                session = ctx.get("session")
                if session is None:
                    self._json({"error": "no session"}, 400)
                    return

                if self.path == "/api/undo":
                    if session["verdicts"]:
                        undone = session["verdicts"].pop()
                        hte.save_session(session, ctx["session_path"])
                        print(f"undid {undone['decision']}")
                    self._json(state(ctx))
                    return

                if self.path != "/api/verdict":
                    self.send_error(404)
                    return

                decision = ctx["by_id"].get(body.get("decision"))
                if decision is None:
                    self._json({"error": "unknown decision"}, 400)
                    return
                if any(v["decision"] == decision["id"] for v in session["verdicts"]):
                    self._json({"state": state(ctx), "reveal": None})   # a double submit
                    return

                slot = body.get("slot")
                slot = int(slot) if slot is not None else None
                if slot is not None and not 1 <= slot <= len(decision["candidates"]):
                    self._json({"error": "slot out of range"}, 400)
                    return
                verdict = hte.make_verdict(
                    decision,
                    body.get("terminate"),
                    slot,
                    str(body.get("choice") or ("s" if slot is None else slot)),
                    str(body.get("note") or "").strip(),
                )
                session["verdicts"].append(verdict)
                hte.save_session(session, ctx["session_path"])
                print(f"  {decision['id']}: "
                      + ("terminate" if verdict["terminate"]
                         else "skipped" if slot is None else f"picked {slot}"))
                self._json({
                    "state": state(ctx),
                    "reveal": reveal_payload(decision, verdict) if session.get("feedback") else None,
                })

    return Handler


# ----------------------------------------------------------------- the page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D2I human evaluation</title>
<style>
:root {
  --bg: #f6f7f9; --card: #fff; --ink: #16181d; --muted: #6b7280; --line: #e3e6ea;
  --accent: #2f6df6; --accent-soft: #eaf1ff; --ok: #12805c; --warn: #b4530a; --pick: #f0f6ff;
  --radius: 12px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --card: #1c1f25; --ink: #e8eaed; --muted: #9aa1ac; --line: #2c3038;
    --accent: #6ea0ff; --accent-soft: #21304d; --ok: #4dbd97; --warn: #e0964d; --pick: #202a3d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 24px 20px 80px; }
header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }
h1 { font-size: 17px; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; }
.bar { height: 6px; background: var(--line); border-radius: 99px; overflow: hidden; margin: 12px 0 22px; }
.bar > i { display: block; height: 100%; background: var(--accent); transition: width .25s ease; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px; margin-bottom: 16px;
}
.eyebrow {
  font-size: 11px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px;
}
.goal { font-size: 15px; }
.cols { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.col { font-size: 12px; padding: 2px 9px; border-radius: 99px; background: var(--accent-soft); color: var(--accent); }
.step { border-left: 2px solid var(--line); padding: 0 0 14px 16px; margin-left: 4px; position: relative; }
.step:last-child { padding-bottom: 0; }
.step::before {
  content: ""; position: absolute; left: -5px; top: 6px; width: 8px; height: 8px;
  border-radius: 99px; background: var(--line);
}
.step.head::before { background: var(--accent); }
.step .meta { font-size: 12px; color: var(--muted); margin-bottom: 3px; }
.step .q { color: var(--muted); margin-bottom: 3px; }
.step .stat { font-size: 12px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ask { font-weight: 600; margin-bottom: 12px; }
.row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
button {
  font: inherit; color: inherit; background: var(--card); border: 1px solid var(--line);
  border-radius: 9px; padding: 9px 16px; cursor: pointer; transition: .12s;
}
button:hover:not(:disabled) { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: .45; cursor: not-allowed; }
button.ghost { color: var(--muted); }
kbd {
  font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid var(--line);
  border-bottom-width: 2px; border-radius: 5px; padding: 1px 5px; color: var(--muted); margin-right: 6px;
}
.pick-row {
  display: flex; gap: 12px; align-items: center; width: 100%; text-align: left;
  padding: 10px 14px; margin-bottom: 6px;
}
.pick-row:hover:not(:disabled) { background: var(--pick); }
.pick-row.sel { border-color: var(--accent); background: var(--pick); }
.pick-row .grow { flex: 1; }
.cand { align-items: flex-start; }
.cand .n {
  flex: none; width: 22px; height: 22px; border-radius: 99px; background: var(--line);
  font-size: 12px; display: grid; place-items: center; margin-top: 1px;
}
.cand.sel .n { background: var(--accent); color: #fff; }
.box {
  flex: none; width: 18px; height: 18px; border-radius: 5px; border: 1.5px solid var(--line);
  display: grid; place-items: center; font-size: 12px; color: #fff;
}
.sel .box { background: var(--accent); border-color: var(--accent); }
.tag {
  font-size: 11px; padding: 1px 7px; border-radius: 99px; background: var(--accent-soft);
  color: var(--accent); margin-right: 7px; white-space: nowrap;
}
input[type=text], input[type=number] {
  font: inherit; color: inherit; background: transparent; border: 1px solid var(--line);
  border-radius: 9px; padding: 9px 12px; margin: 4px 0;
}
input[type=text] { width: 100%; }
label.field { display: block; font-size: 12px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--line); }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }
td.q, th.q { text-align: left; }
tr.picked td { background: var(--pick); font-weight: 600; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .muted { color: var(--muted); }
pre {
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-x: auto;
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 16px;
}
.hint { font-size: 12px; color: var(--muted); margin-top: 10px; }
.err { color: var(--warn); font-size: 13px; margin-top: 10px; }
</style>
</head>
<body>
<div class="wrap">
  <header><h1>D2I human evaluation</h1><span class="sub" id="counter"></span></header>
  <div class="sub" id="who"></div>
  <div class="bar" id="barwrap"><i id="prog" style="width:0"></i></div>
  <div id="app"></div>
</div>

<script>
const $ = (h) => { const t = document.createElement("template"); t.innerHTML = h.trim(); return t.content.firstChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const app = document.getElementById("app");

let S = null, stage = "stop", sel = null, revealed = null, lastStop = null;
let picker = {groups: [], group: null, runs: [], chosen: new Set(), error: ""};

const get  = async (u)    => (await fetch(u)).json();
const post = async (u, b) => (await fetch(u, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(b||{})})).json();

function chrome() {
  const c = document.getElementById("counter"), w = document.getElementById("who");
  const bar = document.getElementById("barwrap");
  if (!S.started) {
    c.textContent = ""; w.textContent = "choose what to evaluate"; bar.style.visibility = "hidden"; return;
  }
  bar.style.visibility = "visible";
  c.textContent = S.decision ? `decision ${S.decision.index + 1} of ${S.total}` : `${S.done} of ${S.total} done`;
  w.textContent = (S.judge ? S.judge + " · " : "") + S.path;
  document.getElementById("prog").style.width = (100 * S.done / Math.max(1, S.total)) + "%";
}

/* ---------------------------------------------------------------- picker */

async function loadGroups() {
  const r = await get("/api/runs");
  picker.groups = r.groups; picker.root = r.root; render();
}
async function loadRuns(g) {
  picker.group = g; picker.runs = []; picker.chosen = new Set(); picker.error = ""; render();
  const r = await get("/api/runs?group=" + encodeURIComponent(g));
  picker.runs = r.runs;
  picker.runs.filter(x => x.usable).forEach(x => picker.chosen.add(x.path));
  render();
}

function renderPicker() {
  let body = "";
  if (!picker.group) {
    body = `
      <div class="card">
        <div class="eyebrow">runs &nbsp;·&nbsp; <span class="mono">${esc(picker.root || "")}</span></div>
        ${picker.groups.length ? picker.groups.map(g => `
          <button class="pick-row" data-group="${esc(g.path)}">
            <span class="grow"><span class="mono">${esc(g.name)}</span></span>
            <span class="muted">${g.n_runs} run${g.n_runs === 1 ? "" : "s"}</span>
          </button>`).join("") : `<p class="muted">no run folders found.</p>`}
      </div>`;
  } else {
    const ready = picker.runs.filter(r => r.usable).length;
    body = `
      <div class="card">
        <div class="eyebrow">${esc(picker.group.split("/").pop())} &nbsp;·&nbsp; ${ready} of ${picker.runs.length} ready</div>
        ${picker.runs.map(r => `
          <button class="pick-row ${picker.chosen.has(r.path) ? "sel" : ""}" data-run="${esc(r.path)}" ${r.usable ? "" : "disabled"}>
            <span class="box">${picker.chosen.has(r.path) ? "✓" : ""}</span>
            <span class="grow"><span class="mono">${esc(r.name)}</span></span>
            <span class="muted">${r.usable ? r.nodes + " decision points" : esc(r.reason)}</span>
          </button>`).join("") || `<p class="muted">nothing judgeable in here.</p>`}
        <div class="row" style="margin-top:14px">
          <button class="ghost" data-act="back">← all runs</button>
          <button class="ghost" data-act="all">select all</button>
          <button class="ghost" data-act="none">clear</button>
        </div>
      </div>
      <div class="card">
        <div class="eyebrow">session</div>
        <div class="row">
          <label class="field">trajectories<br><input type="number" id="n" value="10" min="1" style="width:90px"></label>
          <label class="field">judge<br><input type="text" id="judge" placeholder="your name" style="width:180px"></label>
          <label class="field">seed<br><input type="text" id="seed" placeholder="random" style="width:110px" class="mono"></label>
        </div>
        <div class="row" style="margin-top:14px">
          <button class="primary" id="start" ${picker.chosen.size ? "" : "disabled"}>Start evaluation</button>
          <span class="muted">${picker.chosen.size} run${picker.chosen.size === 1 ? "" : "s"} selected</span>
        </div>
        ${picker.error ? `<div class="err">${esc(picker.error)}</div>` : ""}
      </div>`;
  }
  app.innerHTML = ""; app.append($(`<div>${body}</div>`));
  app.querySelectorAll("[data-group]").forEach(b => b.onclick = () => loadRuns(b.dataset.group));
  app.querySelectorAll("[data-run]").forEach(b => b.onclick = () => {
    const p = b.dataset.run;
    picker.chosen.has(p) ? picker.chosen.delete(p) : picker.chosen.add(p);
    render();
  });
  app.querySelectorAll("[data-act]").forEach(b => b.onclick = () => {
    const a = b.dataset.act;
    if (a === "back") { picker.group = null; return render(); }
    picker.chosen = a === "all" ? new Set(picker.runs.filter(r => r.usable).map(r => r.path)) : new Set();
    render();
  });
  const s = document.getElementById("start");
  if (s) s.onclick = async () => {
    s.disabled = true; s.textContent = "building…";
    const res = await post("/api/start", {
      paths: [...picker.chosen],
      n: Number(document.getElementById("n").value) || 10,
      judge: document.getElementById("judge").value,
      seed: document.getElementById("seed").value.trim(),
      label: picker.group.split("/").pop(),
    });
    if (res.error) { picker.error = res.error; return render(); }
    S = res; stage = S.decision && S.decision.ask_stop ? "stop" : "pick"; render();
  };
}

/* ---------------------------------------------------------------- judging */

function trajectory(d) {
  const steps = d.path.map((s, i) => `
    <div class="step ${i === d.path.length - 1 ? "head" : ""}">
      <div class="meta">depth ${s.depth} · ${esc(s.action)}</div>
      ${s.question ? `<div class="q">${esc(s.question)}</div>` : ""}
      <div>${esc(s.label)}</div>
      <div class="stat">${esc(s.statistic)}</div>
    </div>`).join("");
  return `
    <div class="card">
      <div class="eyebrow">goal</div>
      <div class="goal">${esc(d.goal)}</div>
      <div class="cols">${d.columns.map(c => `<span class="col">${esc(c)}</span>`).join("")}</div>
    </div>
    <div class="card">
      <div class="eyebrow">trajectory so far &nbsp;·&nbsp; ${esc(d.run)} · node ${esc(d.node)} · depth ${d.depth} → ${d.child_depth}</div>
      ${steps}
    </div>`;
}

function renderDecision(d) {
  let body = trajectory(d);
  if (stage === "stop") {
    body += `
      <div class="card">
        <div class="ask">Is this trajectory worth continuing?</div>
        <div class="row">
          <button class="primary" data-act="continue"><kbd>C</kbd>Continue</button>
          <button data-act="terminate"><kbd>T</kbd>Terminate</button>
          <button class="ghost" data-act="skip"><kbd>S</kbd>Skip</button>
          ${S.can_undo ? `<button class="ghost" data-act="undo"><kbd>U</kbd>Undo last</button>` : ""}
        </div>
        <div class="hint">Answered on the trajectory alone — the candidates come next.</div>
      </div>`;
  } else {
    body += `
      <div class="card">
        <div class="ask">Which question should be asked next?</div>
        ${d.candidates.map(c => `
          <button class="pick-row cand" data-slot="${c.slot}">
            <span class="n">${c.slot}</span>
            <span class="grow"><span class="tag">${esc(c.action)}</span>${esc(c.question)}</span>
          </button>`).join("")}
        <input type="text" id="note" placeholder="note (optional)">
        <div class="row">
          <button class="primary" id="submit" disabled>Submit</button>
          <button class="ghost" data-act="skip"><kbd>S</kbd>Skip</button>
          <button class="ghost" data-act="back"><kbd>U</kbd>Back</button>
        </div>
        <div class="hint">Press <kbd>1</kbd>–<kbd>${d.candidates.length}</kbd> to select, <kbd>↵</kbd> to submit.</div>
      </div>`;
  }
  app.innerHTML = ""; app.append($(`<div>${body}</div>`));
  app.querySelectorAll("[data-act]").forEach(b => b.onclick = () => act(b.dataset.act));
  app.querySelectorAll(".cand").forEach(b => b.onclick = () => {
    sel = Number(b.dataset.slot);
    app.querySelectorAll(".cand").forEach(x => x.classList.toggle("sel", Number(x.dataset.slot) === sel));
    document.getElementById("submit").disabled = false;
  });
  const s = document.getElementById("submit");
  if (s) s.onclick = () => send({slot: sel, terminate: lastStop});
}

async function act(what) {
  if (what === "undo") {
    revealed = null; sel = null; lastStop = null;
    S = await post("/api/undo");
    stage = S.decision && S.decision.ask_stop ? "stop" : "pick";
    return render();
  }
  if (what === "back")      { if (S.decision.ask_stop) { stage = "stop"; sel = null; return render(); } return act("undo"); }
  if (what === "skip")      return send({slot: null, terminate: stage === "pick" ? lastStop : null, choice: "s"});
  if (what === "continue")  {
    lastStop = false;
    if (!S.decision.candidates.length) return send({slot: null, terminate: false, choice: "c"});
    stage = "pick"; sel = null; return render();
  }
  if (what === "terminate") { lastStop = true; return send({slot: null, terminate: true, choice: "t"}); }
}

async function send(body) {
  const note = document.getElementById("note");
  const res = await post("/api/verdict", {decision: S.decision.id, note: note ? note.value : "", ...body});
  if (res.error) return;
  S = res.state; revealed = res.reveal; sel = null;
  if (!revealed) { lastStop = null; stage = S.decision && S.decision.ask_stop ? "stop" : "pick"; }
  render();
}

function renderReveal() {
  const r = revealed;
  let head = "";
  if (r.model_terminate !== null) {
    const agree = r.model_terminate === r.human_terminate;
    head = `<p><strong class="${agree ? "ok" : "warn"}">${agree ? "✓ agreement" : "✗ you disagreed"}</strong> —
            the model ${r.model_terminate ? "terminated the line here" : "kept exploring this line"}.</p>` +
           (r.terminate_reason ? `<p class="muted">${esc(r.terminate_reason)}</p>` : "");
  }
  if (r.rank !== null) {
    head += `<p>Your pick ranked <strong>${r.rank}</strong> of ${r.n}${r.rank === 1 ? " — the model's top choice." : "."}</p>`;
  }
  const rows = r.rows.map(x => `
    <tr class="${x.picked ? "picked" : ""}">
      <td>${x.rank}</td><td>${x.slot}</td>
      <td>${x.answered ? "✓" : ""}${x.picked ? "←" : ""}</td>
      <td class="mono">${x.score.toFixed(4)}</td>
      ${x.terms.map(t => `<td class="mono">${t === null ? "—" : t.toFixed(2)}</td>`).join("")}
      <td class="q"><span class="tag">${esc(x.action)}</span></td>
    </tr>`).join("");
  const w = r.lambdas;
  app.innerHTML = "";
  app.append($(`
    <div><div class="card">
      <div class="eyebrow">what the model did</div>
      ${head}
      ${r.rows.length ? `<table>
        <thead><tr><th>#</th><th>slot</th><th></th><th>score</th>
          ${r.terms.map(t => `<th>${t}</th>`).join("")}<th class="q">action</th></tr></thead>
        <tbody>${rows}</tbody></table>
        ${w ? `<div class="hint mono">score = ${r.terms.map(t => `${w[t]}·${t}`).join(" + ")}</div>` : ""}` : ""}
      <div class="row" style="margin-top:16px">
        <button class="primary" data-act="next"><kbd>↵</kbd>Next</button>
        <button class="ghost" data-act="undo"><kbd>U</kbd>Undo</button>
      </div>
    </div></div>`));
  app.querySelectorAll("[data-act]").forEach(b => b.onclick = () => {
    if (b.dataset.act === "undo") return act("undo");
    revealed = null; lastStop = null;
    stage = S.decision && S.decision.ask_stop ? "stop" : "pick";
    render();
  });
}

async function finish() {
  const r = await get("/api/report");
  app.innerHTML = "";
  app.append($(`
    <div>
      <div class="card">
        <div class="eyebrow">done</div>
        <p>All ${S.total} decisions judged. Results written to <span class="mono">${esc(r.results)}</span>.</p>
        ${S.can_undo ? `<div class="row"><button class="ghost" data-act="undo">Undo last</button></div>` : ""}
      </div>
      <pre>${esc(r.lines.join("\n"))}</pre>
    </div>`));
  app.querySelectorAll("[data-act]").forEach(b => b.onclick = () => act("undo"));
}

function render() {
  chrome();
  if (!S.started) return renderPicker();
  if (revealed) return renderReveal();
  if (!S.decision) return finish();
  renderDecision(S.decision);
}

document.addEventListener("keydown", (e) => {
  if (!S || !S.started) return;
  if (e.target.tagName === "INPUT") { if (e.key === "Enter") document.getElementById("submit")?.click(); return; }
  const k = e.key.toLowerCase();
  if (revealed) {
    if (k === "enter") app.querySelector('[data-act="next"]')?.click();
    if (k === "u") act("undo");
    return;
  }
  if (!S.decision) return;
  if (stage === "stop") {
    if (k === "c") act("continue"); else if (k === "t") act("terminate");
    else if (k === "s") act("skip"); else if (k === "u" && S.can_undo) act("undo");
    return;
  }
  if (/^[1-9]$/.test(k)) app.querySelector(`.cand[data-slot="${k}"]`)?.click();
  else if (k === "enter") document.getElementById("submit")?.click();
  else if (k === "s") act("skip");
  else if (k === "u") act("back");
});

(async () => {
  S = await get("/api/state");
  if (!S.started) { await loadGroups(); return; }
  stage = S.decision && S.decision.ask_stop ? "stop" : "pick";
  render();
})();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- entry point

def main() -> None:
    p = argparse.ArgumentParser(
        description="Browser front end for the D2I human trajectory evaluation.")
    p.add_argument("path", type=Path, nargs="?",
                   help="a run dir or a parent of run dirs. Omit it to pick in the browser.")
    p.add_argument("n", type=int, nargs="?", default=10,
                   help="trajectories to sample, one decision point each (default: 10)")
    p.add_argument("--root", type=Path, default=hte.REPO / "runs",
                   help=f"where the picker looks for runs (default: {hte.REPO / 'runs'})")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (recorded in the session)")
    p.add_argument("--judge", default=None, help="name of the person judging")
    p.add_argument("--min-candidates", type=int, default=2,
                   help="skip decision points offering fewer candidates (default: 2)")
    p.add_argument("--drop-unanswerable", action="store_true",
                   help="leave out candidates the model selected but failed to answer")
    p.add_argument("--no-feedback", dest="feedback", action="store_false",
                   help="do not reveal the model's ranking after each answer")
    p.add_argument("--session-dir", type=Path, default=None,
                   help=f"where to write session.json (default: under {hte.OUT_DIR})")
    p.add_argument("--resume", type=Path, default=None, help="continue an existing session.json")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"(default: {DEFAULT_PORT})")
    p.add_argument("--no-browser", dest="browser", action="store_false",
                   help="do not open a browser tab")
    p.add_argument("-o", "--out", type=Path, default=hte.OUT_FILE,
                   help=f"also write the results here (default: {hte.OUT_FILE}); - to skip")
    args = p.parse_args()

    if args.min_candidates < 2:
        raise SystemExit("--min-candidates must be at least 2 — a pool of one is not a choice")

    ctx: dict = {
        "session": None,
        "session_path": None,
        "by_id": {},
        "root": args.root,
        "min_candidates": args.min_candidates,
        "drop_unanswerable": args.drop_unanswerable,
        "feedback": args.feedback,
        "session_dir": args.session_dir,
        "out_file": None if str(args.out) == "-" else args.out,
    }

    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"no session file at {args.resume}")
        session = json.loads(args.resume.read_text(encoding="utf-8"))
        ctx.update(session=session, session_path=args.resume,
                   by_id={d["id"]: d for d in session["decisions"]})
    elif args.path:
        # A path on the command line skips the picker: build the session up front.
        paths = hte.run_dirs(args.path)
        if not paths:
            raise SystemExit(f"no run dirs (nothing holding a report.json) under {args.path}")
        start_session(ctx, {
            "paths": [str(x) for x in paths], "n": args.n, "judge": args.judge,
            "seed": args.seed, "feedback": args.feedback, "label": args.path.name,
        })

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ctx))
    url = f"http://127.0.0.1:{server.server_port}/"
    if ctx["session"] is None:
        print(f"\npick a run in the browser  (looking in {args.root})")
    else:
        done = len({v["decision"] for v in ctx["session"]["verdicts"]})
        print(f"\n{len(ctx['session']['decisions']) - done} decision(s) to judge  ({done} done)")
        print(f"session: {ctx['session_path']}")
    print(f"open:    {url}   (Ctrl-C to stop; the session is saved after every answer)")

    if args.browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — session saved")
    finally:
        server.server_close()
        if ctx["session"] is not None:
            print("\n" + "\n".join(hte.report(ctx["session"])))


if __name__ == "__main__":
    main()
