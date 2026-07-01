#!/usr/bin/env python3
"""Image-level labeller for robot survey captures (stdlib only, no install).

A tiny local web app for image-level annotation of a mission's strip captures,
with two switchable tasks over the very same frames:

  * **Species** (multilabel) — for each frame, which of the 8 target species
    (4 wildflowers + 4 weeds) are present. Grouped Wildflowers / Weeds; nothing
    checked by default, so the labels are unbiased human ground truth (the model's
    predictions are NOT loaded into the UI). Saved to ``labels/image_multilabel.json``.
  * **Image quality** (single-select) — one ordinal judgement per frame: how much
    of the image is degraded by visual artifacts, in quartile bands (0–25 … 75–100),
    stored behind the scenes as 1..4. Saved to ``labels/image_quality.json``.

Both tasks and every completed run under ``missions/`` are switchable live from the
header — no relaunch. Each keeps its own labels file, its own cursor, and its own
progress. Everything autosaves and is resumable: stop any time, rerun, and you land
back on the last frame of whichever task/mission you pick.

Run (from anywhere):
    python -m multimodal_dataset.labeling label                          # first mission, Species task
    python -m multimodal_dataset.labeling label --mission site_1/strip_5 # start on a specific mission
    python -m multimodal_dataset.labeling label --task quality           # start on the Quality task

then open the printed http://127.0.0.1:8765 URL.

Keyboard (Species):  1-4 wildflowers, A/S/D/F weeds
Keyboard (Quality):  1-4 artifact-prevalence band
Both:  ->/Space/Enter confirm + next  |  <- prev  |  c copy previous frame  |  u next unreviewed
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Focal species (display + vector order), the wildflower/weed grouping, the
# fresh-frame default, and the image-quality bands all come from the package's
# single canonical source. Saved species labels migrate by NAME, so that order
# can change without scrambling them.
from ..classes import (
    CLASSES as CLASS_ORDER,
    DEFAULT_ON,
    GROUPS,
    QUALITY_BINS,
    QUALITY_LABEL,
)

# Keyboard shortcut rows for the Species task, one per GROUPS entry: wildflowers
# -> number keys, weeds -> the ASDF home row. Displayed on each species row.
GROUP_KEYS = [["1", "2", "3", "4"], ["a", "s", "d", "f"]]
# Image-quality bands are keyed 1..4 in order.
QUALITY_KEYS = ["1", "2", "3", "4"]

# The two tasks, in header-dropdown order. Each maps to a Store class and its own
# per-run labels file (so the tasks never collide on disk).
TASKS = [
    {"id": "species", "name": "Species", "file": "image_multilabel.json"},
    {"id": "quality", "name": "Image quality", "file": "image_quality.json"},
]
TASKS_META = [{"id": t["id"], "name": t["name"]} for t in TASKS]
TASK_FILE = {t["id"]: t["file"] for t in TASKS}

LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Data assembly                                                               #
# --------------------------------------------------------------------------- #
class BaseStore:
    """Frames + navigation shared by every task; subclasses own the label schema."""

    task = None       # "species" | "quality"
    task_kind = None  # "multi" | "single"

    def __init__(self, captures: str, out_path: str):
        self.captures = captures
        self.out_path = out_path

        # Frames are shown in capture/leg order, straight from captures.geojson.
        geo = json.load(open(os.path.join(captures, "captures.geojson")))
        self.frames: list[dict] = []
        for feat in geo["features"]:
            p = feat["properties"]
            if p.get("corrupt"):
                continue
            self.frames.append({
                "id": os.path.splitext(p["file"])[0],
                "file": p["file"],
                "leg": p.get("leg"),
                "mark": p.get("mark_index"),
                "along": p.get("along_track_m"),
            })

        self.by_id = {f["id"]: f for f in self.frames}
        self.order = [f["id"] for f in self.frames]
        self.labels: dict[str, dict] = {}          # filled by the subclass
        self.cursor = self.order[0] if self.order else None

    # -- shared bookkeeping ------------------------------------------------- #
    def counts(self) -> dict:
        rev = sum(1 for l in self.labels.values() if l["reviewed"])
        edited = sum(1 for l in self.labels.values() if l["source"] == "human")
        return {"total": len(self.frames), "reviewed": rev, "edited": edited}

    def _atomic_write(self, payload: dict):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.out_path)

    def apply_save(self, fid: str, body: dict):
        """Apply one frame's edit; returns None on success or (code, msg) on error."""
        lab = self.labels[fid]
        err = self.set_value(lab, body)
        if err:
            return err
        lab["reviewed"] = bool(body.get("reviewed", lab["reviewed"]))
        lab["source"] = body.get("source", lab["source"])
        lab["updated"] = now_iso()
        if body.get("cursor") in self.by_id:
            self.cursor = body["cursor"]
        self.write()
        return None

    def init_payload(self) -> dict:
        # Deliberately ships NO model scores/predictions to the client.
        p = {
            "task": self.task,
            "task_kind": self.task_kind,
            "tasks": TASKS_META,
            "out_path": self.out_path,
            "mission": CURRENT.mission_name if CURRENT else None,
            "run_id": CURRENT.run_id if CURRENT else None,
            "run_label": (f"{CURRENT.site}/{CURRENT.strip}" if CURRENT else None),
            "runs": [_run_choice(r) for r in RUNS],
            "frames": [{"id": f["id"], "leg": f["leg"], "mark": f["mark"], "along": f["along"]} for f in self.frames],
            "order": self.order,
            "cursor": self.cursor,
            "counts": self.counts(),
            "labels": self.labels_payload(),
        }
        p.update(self.extra_payload())
        return p

    # -- subclass hooks ----------------------------------------------------- #
    def set_value(self, lab: dict, body: dict):        # noqa: D401 - hook
        raise NotImplementedError

    def write(self):
        raise NotImplementedError

    def labels_payload(self) -> dict:
        raise NotImplementedError

    def extra_payload(self) -> dict:
        return {}


class Store(BaseStore):
    """Species multilabel: a present/absent vector per frame, in pipeline order."""

    task = "species"
    task_kind = "multi"

    def __init__(self, captures: str, out_path: str):
        super().__init__(captures, out_path)

        # Self-contained label set — the target species in display order. No model
        # predictions are loaded; labels are pure human ground truth (CLAUDE.md).
        self.classes: list[str] = list(CLASS_ORDER)
        # Fresh frames start blank unless a canonical default species is set.
        self.default_vec = [1 if c == DEFAULT_ON else 0 for c in self.classes]

        # Labels: resume from disk if present, else seed from the default prior.
        if os.path.exists(out_path):
            saved = json.load(open(out_path))
            self.cursor = saved.get("meta", {}).get("cursor") or self.cursor
            saved_classes = saved.get("classes")
            for fid, lab in saved.get("labels", {}).items():
                # Keep only real human work; auto-seeded defaults are dropped and
                # reseeded fresh below.
                keep = lab.get("reviewed") or lab.get("source") == "human"
                if not (fid in self.by_id and keep):
                    continue
                # Rebuild the vector by species NAME (from the stored present-list,
                # falling back to the saved class order) so reordering or appending
                # classes never misaligns a saved label.
                names = lab.get("present")
                if names is None and saved_classes:
                    names = [saved_classes[i] for i, v in enumerate(lab.get("vector", []))
                             if i < len(saved_classes) and v]
                names = set(names or [])
                self.labels[fid] = {
                    "vector": [1 if c in names else 0 for c in self.classes],
                    "reviewed": bool(lab.get("reviewed", False)),
                    "source": lab.get("source", "human"),
                    "updated": lab.get("updated"),
                }
        for f in self.frames:
            self.labels.setdefault(
                f["id"],
                {"vector": list(self.default_vec), "reviewed": False, "source": "default", "updated": None},
            )

    def present(self, vec: list[int]) -> list[str]:
        return [self.classes[i] for i, v in enumerate(vec) if v]

    def set_value(self, lab: dict, body: dict):
        names = body.get("names")
        if not isinstance(names, list):
            # Older clients posted a positional vector, which is unsafe if the class
            # order changed since the page loaded. Require the species-name list.
            return (409, "stale client: reload the page")
        nameset = set(names)
        lab["vector"] = [1 if c in nameset else 0 for c in self.classes]
        return None

    def labels_payload(self) -> dict:
        return {fid: {k: lab[k] for k in ("vector", "reviewed", "source")} for fid, lab in self.labels.items()}

    def extra_payload(self) -> dict:
        return {
            "classes": self.classes,
            "groups": [{"name": name, "members": members,
                        "keys": (GROUP_KEYS[gi] if gi < len(GROUP_KEYS) else [""] * len(members))}
                       for gi, (name, members) in enumerate(GROUPS)],
            "default_vec": self.default_vec,
        }

    def write(self):
        with LOCK:
            self._atomic_write({
                "schema": "strip-image-multilabel/v1",
                "task": "species",
                "classes": self.classes,
                "default_on": [self.classes[i] for i, v in enumerate(self.default_vec) if v],
                "captures_dir": self.captures,
                "meta": {"updated": now_iso(), "cursor": self.cursor, **self.counts()},
                "labels": {
                    fid: {
                        "vector": lab["vector"],
                        "present": self.present(lab["vector"]),
                        "reviewed": lab["reviewed"],
                        "source": lab["source"],
                        "updated": lab["updated"],
                    }
                    for fid, lab in self.labels.items()
                },
            })


class QualityStore(BaseStore):
    """Image quality: one ordinal artifact-prevalence band (1..4, 0=unset) per frame."""

    task = "quality"
    task_kind = "single"

    def __init__(self, captures: str, out_path: str):
        super().__init__(captures, out_path)
        self.bands = [b[0] for b in QUALITY_BINS]     # e.g. "0–25%" .. "75–100%"

        if os.path.exists(out_path):
            saved = json.load(open(out_path))
            self.cursor = saved.get("meta", {}).get("cursor") or self.cursor
            saved_bands = saved.get("bins")
            for fid, lab in saved.get("labels", {}).items():
                keep = lab.get("reviewed") or lab.get("source") == "human"
                if not (fid in self.by_id and keep):
                    continue
                # Prefer the integer bin; fall back to matching the band string, so a
                # relabelled/reordered band list can't silently corrupt saved work.
                b = lab.get("bin") or 0
                if not b and lab.get("band") and saved_bands:
                    try:
                        b = saved_bands.index(lab["band"]) + 1
                    except ValueError:
                        b = 0
                b = b if 0 <= b <= len(self.bands) else 0
                self.labels[fid] = {
                    "bin": int(b),
                    "reviewed": bool(lab.get("reviewed", False)),
                    "source": lab.get("source", "human"),
                    "updated": lab.get("updated"),
                }
        for f in self.frames:
            self.labels.setdefault(
                f["id"],
                {"bin": 0, "reviewed": False, "source": "default", "updated": None},
            )

    def _band(self, b: int):
        return self.bands[b - 1] if 1 <= b <= len(self.bands) else None

    def set_value(self, lab: dict, body: dict):
        try:
            b = int(body.get("bin", 0))
        except (TypeError, ValueError):
            return (400, "bad bin")
        if not (0 <= b <= len(self.bands)):
            return (400, "bin out of range")
        lab["bin"] = b
        return None

    def labels_payload(self) -> dict:
        return {fid: {k: lab[k] for k in ("bin", "reviewed", "source")} for fid, lab in self.labels.items()}

    def extra_payload(self) -> dict:
        return {
            "label_name": QUALITY_LABEL,
            "bins": [{"label": lab, "desc": desc, "key": QUALITY_KEYS[i]}
                     for i, (lab, desc) in enumerate(QUALITY_BINS)],
        }

    def write(self):
        with LOCK:
            self._atomic_write({
                "schema": "strip-image-quality/v1",
                "task": "quality",
                "label": QUALITY_LABEL,
                "bins": self.bands,
                "bin_meaning": "share of the frame degraded by visual artifacts (quartile band, 1..4; 0=unset)",
                "captures_dir": self.captures,
                "meta": {"updated": now_iso(), "cursor": self.cursor, **self.counts()},
                "labels": {
                    fid: {
                        "bin": lab["bin"],
                        "band": self._band(lab["bin"]),
                        "reviewed": lab["reviewed"],
                        "source": lab["source"],
                        "updated": lab["updated"],
                    }
                    for fid, lab in self.labels.items()
                },
            })


STORE_CLASSES = {"species": Store, "quality": QualityStore}

STORE: BaseStore | None = None
RUNS: list = []          # every completed run under missions/, for the in-app switcher
CURRENT = None           # the active RunInfo
CURRENT_TASK = "species"  # the active task id


def _run_choice(r) -> dict:
    """A run as the header dropdown sees it: stable id + a human label."""
    return {"run_id": r.run_id, "mission": r.mission_name,
            "label": f"{r.site}/{r.strip} · {r.run_date or r.run_id}"}


def activate(run_id: str, task: str | None = None) -> bool:
    """Point STORE at the given completed run + task — rebuilds its frames + labels."""
    global STORE, CURRENT, CURRENT_TASK
    match = next((r for r in RUNS if r.run_id == run_id), None)
    if match is None:
        return False
    task = task or CURRENT_TASK
    cls = STORE_CLASSES.get(task)
    if cls is None:
        return False
    out = str(match.run_dir / "labels" / TASK_FILE[task])
    os.makedirs(os.path.dirname(out), exist_ok=True)   # first label of a new mission/task
    with LOCK:
        STORE = cls(str(match.captures_dir), out)
        CURRENT = match
        CURRENT_TASK = task
    return True


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return None

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/api/init":
            return self._send(200, STORE.init_payload())
        if u.path == "/img":
            q = urllib.parse.parse_qs(u.query)
            fid = (q.get("id") or [""])[0]
            f = STORE.by_id.get(fid)
            if not f:
                return self._send(404, {"error": "unknown id"})
            path = os.path.join(STORE.captures, f["file"])
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return self._send(404, {"error": "file missing"})
            return self._send(200, data, "image/jpeg", {"Cache-Control": "public, max-age=86400"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)

        # Switch the active mission/run and/or task without a relaunch; returns a
        # fresh init. Either field is optional — omit run_id to just swap tasks.
        if u.path == "/api/select":
            body = self._read_json()
            if body is None:
                return self._send(400, {"error": "bad json"})
            run_id = body.get("run_id") or (CURRENT.run_id if CURRENT else None)
            task = body.get("task") or CURRENT_TASK
            if not activate(run_id, task):
                return self._send(400, {"error": "unknown run or task"})
            return self._send(200, STORE.init_payload())

        if u.path != "/api/save":
            return self._send(404, {"error": "not found"})
        body = self._read_json()
        if body is None:
            return self._send(400, {"error": "bad json"})
        fid = body.get("id")
        if fid not in STORE.by_id:
            return self._send(400, {"error": "unknown id"})
        err = STORE.apply_save(fid, body)
        if err:
            code, msg = err
            return self._send(code, {"error": msg})
        return self._send(200, {"ok": True, "counts": STORE.counts()})


# --------------------------------------------------------------------------- #
# Frontend                                                                    #
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>image labels</title>
<style>
  :root{--bg:#14171c;--panel:#1d222b;--line:#2c333f;--ink:#e7ecf3;--mut:#8b97a8;
        --on:#3fb950;--onbg:#16341f;--accent:#4493f8;}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;flex-direction:column}
  header{display:flex;align-items:center;gap:14px;padding:10px 16px;border-bottom:1px solid var(--line);
         background:var(--panel)}
  header .title{font-weight:600}
  header .grow{flex:1}
  .sortlab{color:var(--mut);font-size:12.5px}
  select{background:#222a35;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 6px;font:inherit}
  /* prominent Task / Mission switchers -- color-coded chips, glow on hover/focus */
  .switch{display:inline-flex;align-items:center;gap:9px;background:#232b37;border:2px solid var(--sw);
          border-radius:10px;padding:6px 12px;cursor:pointer;transition:box-shadow .15s,transform .05s;
          box-shadow:0 0 0 0 transparent}
  .switch:hover,.switch:focus-within{box-shadow:0 0 0 3px var(--glow),0 2px 10px #0006}
  .switch:active{transform:translateY(1px)}
  .switch .lab{font-size:10px;letter-spacing:.11em;text-transform:uppercase;font-weight:800;color:var(--sw)}
  .switch select{background:transparent;border:0;color:var(--ink);font-size:15px;font-weight:700;
                 padding:1px 16px 1px 0;cursor:pointer;max-width:240px;appearance:none;-webkit-appearance:none;
                 background-image:linear-gradient(45deg,transparent 50%,var(--sw) 50%),
                                  linear-gradient(135deg,var(--sw) 50%,transparent 50%);
                 background-position:calc(100% - 8px) center,calc(100% - 4px) center;
                 background-size:4px 4px,4px 4px;background-repeat:no-repeat}
  .switch select:focus{outline:none}
  .switch.task{--sw:#4493f8;--glow:rgba(68,147,248,.28)}
  .switch.mission{--sw:#3fb950;--glow:rgba(63,185,80,.28)}
  .pill{padding:3px 10px;border:1px solid var(--line);border-radius:999px;color:var(--mut);cursor:pointer;
        user-select:none;font-size:12.5px}
  .pill:hover{color:var(--ink);border-color:var(--accent)}
  #progwrap{width:160px;height:6px;background:#222a35;border-radius:4px;overflow:hidden}
  #prog{height:100%;width:0;background:var(--on);transition:width .2s}
  #progLbl{font-size:12.5px;color:var(--mut)}
  main{flex:1;display:flex;min-height:0}
  #imgwrap{flex:1;display:flex;align-items:center;justify-content:center;padding:14px;min-width:0;background:#0e1116}
  #img{max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;box-shadow:0 4px 24px #0008}
  #side{width:430px;border-left:1px solid var(--line);display:flex;flex-direction:column;background:var(--panel)}
  #meta{padding:12px 16px;border-bottom:1px solid var(--line)}
  #meta b{font-size:14px}
  #meta .sub{color:var(--mut);font-size:12.5px;margin-top:3px}
  #rows{flex:1;overflow:auto;padding:10px}
  .grp{color:var(--mut);font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:700;
       margin:12px 6px 7px}
  .grp:first-child{margin-top:2px}
  .row{display:flex;align-items:center;gap:12px;padding:13px 12px;border:1px solid var(--line);
       border-radius:9px;margin-bottom:8px;cursor:pointer;background:#1a1f28}
  .row:hover{border-color:var(--accent)}
  .row.on{background:var(--onbg);border-color:var(--on)}
  .key{width:22px;height:22px;flex:none;display:flex;align-items:center;justify-content:center;
       border:1px solid var(--line);border-radius:5px;font-size:12px;color:var(--mut)}
  .row.on .key{color:var(--on);border-color:var(--on)}
  .box{width:20px;height:20px;flex:none;border:2px solid var(--mut);border-radius:5px;position:relative}
  .row.on .box{background:var(--on);border-color:var(--on)}
  .row.on .box:after{content:"";position:absolute;left:6px;top:1px;width:5px;height:11px;
       border:solid #0c1a10;border-width:0 2px 2px 0;transform:rotate(45deg)}
  /* single-select (quality) uses a round radio + a filled dot instead of a tick */
  .box.radio{border-radius:50%}
  .row.on .box.radio:after{content:"";position:absolute;left:4px;top:4px;width:8px;height:8px;
       border:0;border-radius:50%;background:#0c1a10;transform:none}
  .name{flex:1;font-size:15px}.name i{font-style:italic}
  .name .dsc{display:block;color:var(--mut);font-size:12px;font-style:normal;margin-top:2px}
  /* bottom navigation -- the primary way to move between frames */
  #nav{display:flex;gap:10px;padding:12px;border-top:1px solid var(--line);background:#171b22}
  .btn{flex:1;padding:12px 10px;border:1px solid var(--line);border-radius:9px;background:#222a35;color:var(--ink);
       font:inherit;font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px}
  .btn:hover{border-color:var(--accent)}
  .btn.primary{flex:1.7;background:var(--accent);border-color:var(--accent);color:#06131f;font-weight:700;font-size:15px}
  .btn.primary:hover{filter:brightness(1.08)}
  .btn small{opacity:.7;font-weight:400}
  footer{padding:8px 16px;border-top:1px solid var(--line);color:var(--mut);font-size:12px;background:var(--panel)}
  kbd{background:#222a35;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--ink)}
  .rev{color:var(--on)} .unrev{color:#d9a441}
</style></head>
<body>
<header>
  <span class="title" id="title">image labels</span>
  <label class="switch task"><span class="lab">Task</span><select id="task"></select></label>
  <label class="switch mission"><span class="lab">Mission</span><select id="mission"></select></label>
  <span class="pill" id="unrevBtn" title="u">jump to next unreviewed</span>
  <span class="grow"></span>
  <span id="posLbl"></span>
  <div id="progwrap"><div id="prog"></div></div>
  <span id="progLbl"></span>
</header>
<main>
  <div id="imgwrap"><img id="img" alt=""></div>
  <div id="side">
    <div id="meta"></div>
    <div id="rows"></div>
    <div id="nav">
      <button class="btn" id="prevBtn" title="←">← Prev</button>
      <button class="btn" id="copyBtn" title="c">Copy prev <small>c</small></button>
      <button class="btn primary" id="nextBtn" title="→ / Space / Enter">Next → <small>confirm</small></button>
    </div>
  </div>
</main>
<footer><span id="help"></span> &nbsp;·&nbsp; autosaves to <span id="outp"></span></footer>
<script>
let S=null, pos=0, frameById={}, keyAction={};
const $=s=>document.querySelector(s);
const curId=()=>S.order[pos];
const curLab=()=>S.labels[curId()];

async function boot(){
  bindUI();                                   // listeners are attached once, survive task/mission switches
  apply(await (await fetch('/api/init')).json());
}

// (Re)load the whole view from an init payload — used on first load AND after a
// live task/mission switch, so all paths share one code path.
function apply(p){
  S=p; frameById={}; S.frames.forEach(f=>frameById[f.id]=f);
  buildKeys();
  $('#outp').textContent=S.out_path;
  const tName=(S.tasks.find(t=>t.id===S.task)||{}).name||'labels';
  $('#title').textContent=(S.run_label?S.run_label+' — ':'')+tName;
  document.title=(S.run_label?S.run_label+' · ':'')+tName;
  $('#help').innerHTML=helpText();
  fillTasks(); fillMissions();
  const i=S.order.indexOf(S.cursor); pos=i<0?0:i;
  show(); progress();
}

// char -> action, per task. Species: 1-4/ASDF toggle a species. Quality: 1-4 set a band.
function buildKeys(){
  keyAction={};
  if(S.task_kind==='single'){
    S.bins.forEach((b,j)=>{const k=(b.key||'').toLowerCase(); if(k) keyAction[k]=()=>setBin(j+1);});
  }else{
    S.groups.forEach(g=>g.members.forEach((name,j)=>{
      const k=(g.keys[j]||'').toLowerCase(); const idx=S.classes.indexOf(name);
      if(k) keyAction[k]=()=>toggle(idx);
    }));
  }
}
function helpText(){
  const common='<kbd>→</kbd>/<kbd>Space</kbd>/<kbd>Enter</kbd> confirm + next &nbsp;·&nbsp; '
             +'<kbd>←</kbd> prev &nbsp;·&nbsp; <kbd>c</kbd> copy previous frame &nbsp;·&nbsp; <kbd>u</kbd> next unreviewed';
  if(S.task_kind==='single')
    return '<kbd>1</kbd>–<kbd>4</kbd> artifact-prevalence band &nbsp;·&nbsp; '+common;
  return '<kbd>1</kbd>–<kbd>4</kbd> wildflowers &nbsp;·&nbsp; <kbd>A</kbd><kbd>S</kbd><kbd>D</kbd><kbd>F</kbd> weeds &nbsp;·&nbsp; '+common;
}

function fillTasks(){
  $('#task').innerHTML=S.tasks.map(t=>
    `<option value="${t.id}"${t.id===S.task?' selected':''}>${t.name}</option>`).join('');
}
function fillMissions(){
  $('#mission').innerHTML=S.runs.map(r=>
    `<option value="${r.run_id}"${r.run_id===S.run_id?' selected':''}>${r.label}</option>`).join('');
}
async function selectPost(payload){
  let p;
  try{ p=await (await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},
                                           body:JSON.stringify(payload)})).json(); }
  catch(e){ p={error:'request failed'}; }
  if(p.error){$('#progLbl').textContent='⚠ '+p.error;$('#progLbl').style.color='#d9a441';
              fillTasks();fillMissions();return;}
  apply(p);
}
async function switchTask(tid){ if(!tid||tid===S.task)return; await selectPost({task:tid}); $('#task').blur(); }
async function switchMission(rid){ if(!rid||rid===S.run_id)return; await selectPost({run_id:rid}); $('#mission').blur(); }

function show(){
  const id=curId(), f=frameById[id], lab=S.labels[id];
  $('#img').src='/img?id='+encodeURIComponent(id);
  const rev = lab.reviewed?'<span class="rev">✓ reviewed</span>':'<span class="unrev">● unreviewed</span>';
  $('#meta').innerHTML=`<b>${id}</b><div class="sub">leg ${f.leg} · mark ${f.mark} · ${f.along} m &nbsp;&nbsp; ${rev}</div>`;
  $('#posLbl').textContent=`${pos+1} / ${S.order.length}`;
  rows();
}

function rows(){ return S.task_kind==='single'?rowsSingle():rowsMulti(); }

function rowsMulti(){
  const lab=curLab();
  let h='';
  S.groups.forEach(g=>{
    h+=`<div class="grp">${g.name}</div>`;
    g.members.forEach((name,j)=>{
      const i=S.classes.indexOf(name);          // vector slot
      const key=(g.keys[j]||'').toUpperCase();  // shortcut shown on the row
      h+=`<div class="row ${lab.vector[i]?'on':''}" data-i="${i}">
        <div class="key">${key}</div><div class="box"></div>
        <div class="name"><i>${name}</i></div></div>`;
    });
  });
  $('#rows').innerHTML=h;
  $('#rows').querySelectorAll('.row').forEach(r=>r.onclick=()=>toggle(+r.dataset.i));
}

function rowsSingle(){
  const lab=curLab();
  let h=`<div class="grp">${S.label_name} · share of frame with visual artifacts</div>`;
  S.bins.forEach((b,j)=>{
    const on=lab.bin===(j+1);
    h+=`<div class="row ${on?'on':''}" data-b="${j+1}">
      <div class="key">${b.key}</div><div class="box radio"></div>
      <div class="name">${b.label}<span class="dsc">${b.desc}</span></div></div>`;
  });
  $('#rows').innerHTML=h;
  $('#rows').querySelectorAll('.row').forEach(r=>r.onclick=()=>setBin(+r.dataset.b));
}

function toggle(i){const lab=curLab();lab.vector[i]=lab.vector[i]?0:1;lab.source='human';rows();save(false);}
function setBin(b){const lab=curLab();lab.bin=(lab.bin===b?0:b);lab.source='human';rows();save(false);}

function save(markRev){
  const id=curId(), lab=S.labels[id]; if(markRev) lab.reviewed=true;
  const body={id,reviewed:lab.reviewed,source:lab.source,cursor:id};
  if(S.task_kind==='single') body.bin=lab.bin||0;
  else body.names=S.classes.filter((c,i)=>lab.vector[i]);   // name-based: safe across class reorders
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
   .then(r=>r.json()).then(d=>{
     if(d.counts){setCounts(d.counts);}
     else{$('#progLbl').textContent='⚠ reload page';$('#progLbl').style.color='#d9a441';}
   });
  progress();
}
function go(d){save(true);pos=Math.min(S.order.length-1,Math.max(0,pos+d));show();}
function copyPrev(){if(pos===0)return;const src=S.labels[S.order[pos-1]],lab=curLab();
  if(S.task_kind==='single') lab.bin=src.bin; else lab.vector=src.vector.slice();
  lab.source='human';rows();save(false);}
function nextUnreviewed(){const a=S.order;for(let k=1;k<=a.length;k++){const j=(pos+k)%a.length;
  if(!S.labels[a[j]].reviewed){pos=j;show();return;}}}

function progress(){let r=0;for(const k in S.labels)if(S.labels[k].reviewed)r++;setCounts({reviewed:r,total:S.frames.length});}
function setCounts(c){const pct=c.total?Math.round(100*c.reviewed/c.total):0;
  $('#prog').style.width=pct+'%';$('#progLbl').textContent=`${c.reviewed} / ${c.total}`;$('#progLbl').style.color='';}

function bindUI(){
  $('#task').onchange=e=>switchTask(e.target.value);
  $('#mission').onchange=e=>switchMission(e.target.value);
  $('#unrevBtn').onclick=nextUnreviewed;
  $('#prevBtn').onclick=()=>go(-1);
  $('#nextBtn').onclick=()=>go(1);
  $('#copyBtn').onclick=copyPrev;
  document.addEventListener('keydown',e=>{
    if(e.metaKey||e.ctrlKey||e.altKey)return;
    if(e.target.tagName==='SELECT')return;
    const k=e.key.length===1?e.key.toLowerCase():e.key;
    if(k in keyAction){keyAction[k]();e.preventDefault();return;}
    switch(e.key){
      case'ArrowRight':case' ':case'Enter':go(1);e.preventDefault();break;
      case'ArrowLeft':go(-1);e.preventDefault();break;
      case'c':copyPrev();break;
      case'u':nextUnreviewed();break;
    }
  });
}
boot();
</script>
</body></html>
"""


def main(argv=None):
    global RUNS
    from . import select_run, _MISSIONS_ROOT
    from ..discover import completed_runs
    ap = argparse.ArgumentParser(description="Image-level labeller (species + image quality)")
    ap.add_argument("--mission", help="start on this mission: site_x/strip_y (optional; switchable in-app)")
    ap.add_argument("--run", help="start on this exact run id (optional; switchable in-app)")
    ap.add_argument("--task", choices=[t["id"] for t in TASKS], default="species",
                    help="start on this task (optional; switchable in-app)")
    ap.add_argument("--missions-root", default=None, help="missions/ tree (default: the repo's)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args(argv)

    root = args.missions_root or _MISSIONS_ROOT
    RUNS = completed_runs(root)
    if not RUNS:
        raise SystemExit(f"no completed runs under {root}")

    # Initial run: honor --mission/--run if given (nice for jumping straight in),
    # else start on the first completed run — any mission/task is switchable in-app.
    start = select_run(mission=args.mission, run=args.run, missions_root=root) \
        if (args.mission or args.run) else RUNS[0]
    activate(start.run_id, args.task)

    c = STORE.counts()
    task_name = next(t["name"] for t in TASKS if t["id"] == args.task)
    url = f"http://127.0.0.1:{args.port}"
    print(f"labelling {CURRENT.mission_name} / {CURRENT.run_id}  ({len(RUNS)} missions available — switch in-app)")
    print(f"task: {task_name}  ·  {c['total']} frames  ·  {c['reviewed']} already reviewed")
    if isinstance(STORE, Store):
        print(f"default-on: {', '.join(STORE.present(STORE.default_vec)) or '(none)'}")
    print(f"labels -> {STORE.out_path}")
    print(f"open    {url}")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. labels saved at", STORE.out_path)


if __name__ == "__main__":
    main()
