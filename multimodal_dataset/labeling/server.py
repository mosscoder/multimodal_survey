#!/usr/bin/env python3
"""Strip image-level multilabel labeller (stdlib only, no install).

A tiny local web app for image-level multilabel annotation of the strip
captures: for each frame, which of the 8 focal species are present. Checkboxes
start blank except *Thinopyrum intermedium*, which is on by default (it occurs
in essentially every frame). The model's predictions are NOT loaded into the UI,
so the labels are an unbiased human ground truth. Everything autosaves to a JSON
file in the exact class order the pipeline uses, and the tool is resumable: stop
any time, rerun, and you land back on the last frame.

Run (from anywhere):
    mamba run -n pixelflora python /Users/kdoherty/multimodal_survey/dev/strip5_image_labeler/label_server.py

then open the printed http://127.0.0.1:8765 URL.

Keyboard:  1-9 toggle species  |  ->/Space/Enter confirm + next  |  <- prev
           c copy previous frame  |  u jump to next unreviewed  |  o toggle sort
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

# Focal species (display + vector order) and the always-present default come from
# the package's single canonical source. Saved labels migrate by NAME, so this
# order can change without scrambling them.
from ..classes import CLASSES as CLASS_ORDER, DEFAULT_ON
BAND = 0.2  # |score - threshold| < BAND  =>  the model's call here is "borderline" (used only to order the optional "hardest first" sort)

LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Data assembly                                                               #
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, captures: str, preds_path: str, out_path: str):
        self.captures = captures
        self.preds_path = preds_path
        self.out_path = out_path

        # Self-contained label set — the 9 target species in display order. No model
        # predictions are loaded; labels are pure human ground truth (CLAUDE.md).
        self.classes: list[str] = list(CLASS_ORDER)
        self.thresholds: dict[str, float] = {}
        by_id = {}

        # Fresh frames start blank except the always-present grass.
        self.default_vec = [1 if DEFAULT_ON in c else 0 for c in self.classes]

        # Canonical frame order comes from captures.geojson (capture/leg order).
        geo = json.load(open(os.path.join(captures, "captures.geojson")))
        self.frames: list[dict] = []
        for feat in geo["features"]:
            p = feat["properties"]
            if p.get("corrupt"):
                continue
            fid = os.path.splitext(p["file"])[0]
            # Model scores are used ONLY to compute the optional "hardest first"
            # ordering; they are never sent to the UI or used to seed a label.
            scores = by_id.get(fid, {}).get("scores", {})
            ambiguity = sum(max(0.0, BAND - abs(scores.get(c, 0.0) - self.thresholds[c]))
                            for c in self.classes if c in self.thresholds)
            self.frames.append(
                {
                    "id": fid,
                    "file": p["file"],
                    "leg": p.get("leg"),
                    "mark": p.get("mark_index"),
                    "along": p.get("along_track_m"),
                    "_ambiguity": ambiguity,
                }
            )

        self.by_id = {f["id"]: f for f in self.frames}
        seq_index = {f["id"]: i for i, f in enumerate(self.frames)}
        self.order_sequence = [f["id"] for f in self.frames]
        self.order_uncertain = sorted(
            (f["id"] for f in self.frames),
            key=lambda fid: (-self.by_id[fid]["_ambiguity"], seq_index[fid]),
        )

        # Labels: resume from disk if present, else seed from the default prior.
        self.labels: dict[str, dict] = {}
        self.cursor = self.order_sequence[0] if self.order_sequence else None
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

    def counts(self) -> dict:
        rev = sum(1 for l in self.labels.values() if l["reviewed"])
        edited = sum(1 for l in self.labels.values() if l["source"] == "human")
        return {"total": len(self.frames), "reviewed": rev, "edited": edited}

    def present(self, vec: list[int]) -> list[str]:
        return [self.classes[i] for i, v in enumerate(vec) if v]

    def write(self):
        with LOCK:
            payload = {
                "schema": "strip-image-multilabel/v1",
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
            }
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            tmp = self.out_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.out_path)

    def init_payload(self) -> dict:
        # Deliberately ships NO model scores/predictions to the client.
        return {
            "classes": self.classes,
            "default_vec": self.default_vec,
            "out_path": self.out_path,
            "frames": [{"id": f["id"], "leg": f["leg"], "mark": f["mark"], "along": f["along"]} for f in self.frames],
            "orders": {"sequence": self.order_sequence, "uncertain": self.order_uncertain},
            "labels": {fid: {k: lab[k] for k in ("vector", "reviewed", "source")} for fid, lab in self.labels.items()},
            "cursor": self.cursor,
            "counts": self.counts(),
        }


STORE: Store | None = None


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
        if u.path != "/api/save":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        fid = body.get("id")
        if fid not in STORE.by_id:
            return self._send(400, {"error": "unknown id"})
        names = body.get("names")
        if not isinstance(names, list):
            # Older clients posted a positional vector, which is unsafe if the class
            # order changed since the page loaded. Require the species-name list.
            return self._send(409, {"error": "stale client: reload the page"})
        nameset = set(names)
        lab = STORE.labels[fid]
        lab["vector"] = [1 if c in nameset else 0 for c in STORE.classes]
        lab["reviewed"] = bool(body.get("reviewed", lab["reviewed"]))
        lab["source"] = body.get("source", lab["source"])
        lab["updated"] = now_iso()
        if body.get("cursor") in STORE.by_id:
            STORE.cursor = body["cursor"]
        STORE.write()
        return self._send(200, {"ok": True, "counts": STORE.counts()})


# --------------------------------------------------------------------------- #
# Frontend                                                                    #
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>strip-5 image labels</title>
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
  .name{flex:1;font-size:15px}.name i{font-style:italic}
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
  <span class="title">strip-5 image labels</span>
  <span class="sortlab">Sort
    <select id="sort">
      <option value="sequence">capture order — walk the strip</option>
      <option value="uncertain">hardest first — model least sure</option>
    </select>
  </span>
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
<footer>
  <kbd>1</kbd>–<kbd>9</kbd> toggle species &nbsp;·&nbsp; <kbd>→</kbd>/<kbd>Space</kbd>/<kbd>Enter</kbd> confirm + next
  &nbsp;·&nbsp; <kbd>←</kbd> prev &nbsp;·&nbsp; <kbd>c</kbd> copy previous frame
  &nbsp;·&nbsp; <kbd>u</kbd> next unreviewed &nbsp;·&nbsp; autosaves to <span id="outp"></span>
</footer>
<script>
let S=null, order='sequence', pos=0, frameById={};
const $=s=>document.querySelector(s);

async function init(){
  S=await (await fetch('/api/init')).json();
  S.frames.forEach(f=>frameById[f.id]=f);
  $('#outp').textContent=S.out_path;
  let i=S.orders[order].indexOf(S.cursor); pos=i<0?0:i;
  bindUI(); show(); progress();
}
const ids=()=>S.orders[order];
const curId=()=>ids()[pos];

function show(){
  const id=curId(), f=frameById[id], lab=S.labels[id];
  $('#img').src='/img?id='+encodeURIComponent(id);
  const rev = lab.reviewed?'<span class="rev">✓ reviewed</span>':'<span class="unrev">● unreviewed</span>';
  $('#meta').innerHTML=`<b>${id}</b><div class="sub">leg ${f.leg} · mark ${f.mark} · ${f.along} m &nbsp;&nbsp; ${rev}</div>`;
  $('#posLbl').textContent=`${pos+1} / ${ids().length}`;
  rows();
}

function rows(){
  const lab=S.labels[curId()];
  let h='';
  S.classes.forEach((c,i)=>{
    h+=`<div class="row ${lab.vector[i]?'on':''}" data-i="${i}">
      <div class="key">${i+1}</div><div class="box"></div>
      <div class="name"><i>${c}</i></div></div>`;
  });
  $('#rows').innerHTML=h;
  $('#rows').querySelectorAll('.row').forEach(r=>r.onclick=()=>toggle(+r.dataset.i));
}

function toggle(i){const lab=S.labels[curId()];lab.vector[i]=lab.vector[i]?0:1;lab.source='human';rows();save(false);}
function save(markRev){
  const id=curId(), lab=S.labels[id]; if(markRev) lab.reviewed=true;
  const names=S.classes.filter((c,i)=>lab.vector[i]);   // name-based: safe across class reorders
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,names,reviewed:lab.reviewed,source:lab.source,cursor:id})})
   .then(r=>r.json()).then(d=>{
     if(d.counts){setCounts(d.counts);}
     else{$('#progLbl').textContent='⚠ reload page';$('#progLbl').style.color='#d9a441';}
   });
  progress();
}
function go(d){save(true);pos=Math.min(ids().length-1,Math.max(0,pos+d));show();}
function copyPrev(){if(pos===0)return;const src=S.labels[ids()[pos-1]].vector;const lab=S.labels[curId()];
  lab.vector=src.slice();lab.source='human';rows();save(false);}
function nextUnreviewed(){const a=ids();for(let k=1;k<=a.length;k++){const j=(pos+k)%a.length;
  if(!S.labels[a[j]].reviewed){pos=j;show();return;}}}
function setOrder(o){const id=curId();order=o;$('#sort').value=o;const j=ids().indexOf(id);pos=j<0?0:j;show();}

function progress(){let r=0;for(const k in S.labels)if(S.labels[k].reviewed)r++;setCounts({reviewed:r,total:S.frames.length});}
function setCounts(c){const pct=c.total?Math.round(100*c.reviewed/c.total):0;
  $('#prog').style.width=pct+'%';$('#progLbl').textContent=`${c.reviewed} / ${c.total}`;}

function bindUI(){
  $('#sort').onchange=e=>setOrder(e.target.value);
  $('#unrevBtn').onclick=nextUnreviewed;
  $('#prevBtn').onclick=()=>go(-1);
  $('#nextBtn').onclick=()=>go(1);
  $('#copyBtn').onclick=copyPrev;
  document.addEventListener('keydown',e=>{
    if(e.metaKey||e.ctrlKey||e.altKey)return;
    if(e.target.tagName==='SELECT')return;
    if(e.key>='1'&&e.key<='9'){const n=+e.key-1; if(n<S.classes.length){toggle(n);e.preventDefault();} return;}
    switch(e.key){
      case'ArrowRight':case' ':case'Enter':go(1);e.preventDefault();break;
      case'ArrowLeft':go(-1);e.preventDefault();break;
      case'c':copyPrev();break;
      case'u':nextUnreviewed();break;
      case'o':setOrder(order==='sequence'?'uncertain':'sequence');break;
    }
  });
}
init();
</script>
</body></html>
"""


def main(argv=None):
    global STORE
    import os
    from . import select_run
    ap = argparse.ArgumentParser(description="Strip image-level multilabel labeller")
    ap.add_argument("--mission", help="which mission to label: site_x/strip_y")
    ap.add_argument("--run", help="exact run id (use when a mission has multiple completed runs)")
    ap.add_argument("--missions-root", default=None, help="missions/ tree (default: the repo's)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args(argv)

    run = select_run(mission=args.mission, run=args.run, missions_root=args.missions_root)
    captures = str(run.captures_dir)
    out = str(run.run_dir / "labels" / "image_multilabel.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)   # first label of a new mission

    STORE = Store(captures, None, out)
    c = STORE.counts()
    url = f"http://127.0.0.1:{args.port}"
    print(f"labelling {run.mission_name} / {run.run_id}")
    print(f"strip image labeller  ·  {c['total']} frames  ·  {c['reviewed']} already reviewed")
    print(f"default-on: {', '.join(STORE.present(STORE.default_vec)) or '(none)'}")
    print(f"labels -> {out}")
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
        print("\nstopped. labels saved at", out)


if __name__ == "__main__":
    main()
