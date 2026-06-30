#!/usr/bin/env python3
"""Strip FALSE-POSITIVE / FALSE-NEGATIVE review app (stdlib + numpy).

A sibling of label_server.py. Where the labeller hides the model and collects an
UNBIASED human ground truth, this app does the opposite: it puts the model's
prediction front-and-centre so you can AUDIT the disagreements and fix the labels.

It loads the SAME label source the labeller writes — `labels/image_multilabel.json`
— so the two apps share one source of truth (an edit here shows up there, and vice
versa). Edits made here are tagged `source:"review"` so the labeller's blind work and
the review corrections stay distinguishable in the one file.

For each frame it compares the deployed model's call (inference_viz/summary.json,
any-patch @ frozen macro+patch thresholds) against the human GT and builds a queue of
DISAGREEMENTS per species:
  * FALSE POSITIVE  — model says PRESENT, GT says ABSENT   (is it a real plant the labeller missed? -> GT gap)
  * FALSE NEGATIVE  — model says ABSENT,  GT says PRESENT  (did the model miss it? confirm/keep GT)
The macro+patch overlay is shown so you can see exactly where the model fired; toggle
to the raw capture to judge unbiased. Pick the species + which errors to review; the
queue is sorted by model confidence (most-confident first). Fixing a checkbox writes
the whole GT file atomically (a one-time .bak is made on startup).

Run:
    mamba run -n pixelflora python dev/lupins_dinov3_demo/labeling/review_server.py
then open http://127.0.0.1:8766 .

Keyboard:  1-9 toggle species | a accept model for the queued species | ->/Enter next
           <- prev | r toggle overlay/raw | f cycle FP/FN/both
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# Model-prediction artifacts (summary.json / overlays / scores) are passed in as
# args, so this app stays decoupled from the MIL program. Scored species come from
# the package's single canonical source (matches the MIL program by value).
from ..classes import CLASSES as CLASS_ORDER

DEFAULT_SPECIES = "Gaillardia aristata"   # initial hunt; --species overrides; switchable in the UI
LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, captures, gt_path, summary_path, overlays, scores_path):
        self.captures, self.gt_path = captures, gt_path
        self.overlays = overlays
        self.classes = list(CLASS_ORDER)

        # frame order from captures.geojson (same canonical order as the labeller)
        geo = json.load(open(os.path.join(captures, "captures.geojson")))
        self.frames, self.by_id = [], {}
        for feat in geo["features"]:
            p = feat["properties"]
            if p.get("corrupt"):
                continue
            fid = os.path.splitext(p["file"])[0]
            f = {"id": fid, "file": p["file"], "leg": p.get("leg"), "mark": p.get("mark_index"),
                 "along": p.get("along_track_m")}
            self.frames.append(f); self.by_id[fid] = f

        # SHARED human GT (the labeller's output). Map present-list -> vector by NAME.
        self.labels = {}
        gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {"labels": {}}
        self.gt_meta = gt.get("meta", {})
        for f in self.frames:
            lab = gt.get("labels", {}).get(f["id"], {})
            names = set(lab.get("present", []))
            self.labels[f["id"]] = {
                "vector": [1 if c in names else 0 for c in self.classes],
                "reviewed": bool(lab.get("reviewed", False)),
                "source": lab.get("source", "default"),
                "updated": lab.get("updated"),
            }

        # MODEL predictions: any-patch present-list per frame (deployed macro+patch).
        self.tiling = "?"; self.thresholds = {}
        self.model_present = {f["id"]: set() for f in self.frames}
        if os.path.exists(summary_path):
            s = json.load(open(summary_path))
            self.tiling = s.get("tiling", "?"); self.thresholds = s.get("thresholds", {})
            for r in s.get("results", []):
                if r["image"] in self.model_present:
                    self.model_present[r["image"]] = set(r.get("present", []))

        # MODEL scores (optional, for display + queue ranking): macro+patch any-patch max.
        self.scores = {f["id"]: {} for f in self.frames}
        if os.path.exists(scores_path):
            z = np.load(scores_path, allow_pickle=True)
            key = "macro+patch" if "macro+patch" in z.files else next((k for k in z.files if "patch" in k), None)
            if key is not None and "fids" in z.files:
                sfids = [str(x) for x in z["fids"]]; arr = z[key]
                col = {c: i for i, c in enumerate(CLASS_ORDER)}   # npz cols are in CLASS_ORDER/NAMES order
                for i, fid in enumerate(sfids):
                    if fid in self.scores:
                        self.scores[fid] = {c: float(arr[i, col[c]]) for c in self.classes}

        self.has_overlay = os.path.isdir(overlays)

    # ---- disagreement model ------------------------------------------------
    def status(self, fid, sp):
        """'FP' | 'FN' | 'TP' | 'TN' for one (frame, species)."""
        si = self.classes.index(sp)
        gt = self.labels[fid]["vector"][si] == 1
        md = sp in self.model_present.get(fid, ())
        return ("TP" if md else "FN") if gt else ("FP" if md else "TN")

    def queue(self, sp, kind):
        """fids with a disagreement of `kind` ('FP'|'FN'|'both') for species `sp`,
        sorted by model confidence (desc)."""
        want = {"both": {"FP", "FN"}}.get(kind, {kind})
        q = [f["id"] for f in self.frames if self.status(f["id"], sp) in want]
        q.sort(key=lambda fid: self.scores.get(fid, {}).get(sp, 0.0), reverse=True)
        return q

    def counts(self):
        rev = sum(1 for l in self.labels.values() if l["reviewed"])
        rcorr = sum(1 for l in self.labels.values() if l["source"] == "review")
        return {"total": len(self.frames), "reviewed": rev, "review_edits": rcorr}

    def present(self, vec):
        return [self.classes[i] for i, v in enumerate(vec) if v]

    def write(self):
        """Write the FULL shared GT file atomically (preserves every frame; only
        edited frames change). Schema-compatible with label_server.py."""
        with LOCK:
            payload = {
                "schema": "strip-image-multilabel/v1",
                "classes": self.classes,
                "default_on": ["Thinopyrum intermedium"],
                "captures_dir": self.captures,
                "meta": {"updated": now_iso(), "cursor": self.gt_meta.get("cursor"), **self.counts()},
                "labels": {fid: {"vector": lab["vector"], "present": self.present(lab["vector"]),
                                 "reviewed": lab["reviewed"], "source": lab["source"],
                                 "updated": lab["updated"]}
                           for fid, lab in self.labels.items()},
            }
            tmp = self.gt_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.gt_path)

    def init_payload(self):
        return {
            "classes": self.classes,
            "tiling": self.tiling, "thresholds": self.thresholds,
            "default_species": DEFAULT_SPECIES if DEFAULT_SPECIES in self.classes else self.classes[0],
            "has_overlay": self.has_overlay, "has_scores": any(self.scores.values()),
            "gt_path": self.gt_path,
            "frames": {f["id"]: {"leg": f["leg"], "mark": f["mark"], "along": f["along"]} for f in self.frames},
            "labels": {fid: {"vector": l["vector"], "reviewed": l["reviewed"], "source": l["source"]}
                       for fid, l in self.labels.items()},
            "model_present": {fid: sorted(s) for fid, s in self.model_present.items()},
            "scores": self.scores,
            "counts": self.counts(),
        }


STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/api/init":
            return self._send(200, STORE.init_payload())
        if u.path == "/api/queue":
            q = urllib.parse.parse_qs(u.query)
            sp = (q.get("species") or [""])[0]; kind = (q.get("kind") or ["FP"])[0]
            if sp not in STORE.classes:
                return self._send(400, {"error": "unknown species"})
            return self._send(200, {"species": sp, "kind": kind, "ids": STORE.queue(sp, kind)})
        if u.path == "/img":
            q = urllib.parse.parse_qs(u.query)
            fid = (q.get("id") or [""])[0]; kind = (q.get("kind") or ["overlay"])[0]
            f = STORE.by_id.get(fid)
            if not f:
                return self._send(404, {"error": "unknown id"})
            if kind == "overlay" and STORE.has_overlay:
                path = os.path.join(STORE.overlays, fid + ".jpg")
                if not os.path.exists(path):
                    path = os.path.join(STORE.captures, f["file"])
            else:
                path = os.path.join(STORE.captures, f["file"])
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return self._send(404, {"error": "file missing"})
            return self._send(200, data, "image/jpeg", {"Cache-Control": "no-store"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/save":
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
            return self._send(409, {"error": "stale client: reload"})
        nameset = set(names)
        lab = STORE.labels[fid]
        lab["vector"] = [1 if c in nameset else 0 for c in STORE.classes]
        lab["reviewed"] = bool(body.get("reviewed", lab["reviewed"]))
        lab["source"] = "review"          # this edit came from the review app
        lab["updated"] = now_iso()
        STORE.write()
        return self._send(200, {"ok": True, "counts": STORE.counts(),
                                "source": lab["source"], "reviewed": lab["reviewed"]})


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>strip-5 FP/FN review</title>
<style>
  :root{--bg:#14171c;--panel:#1d222b;--line:#2c333f;--ink:#e7ecf3;--mut:#8b97a8;
        --on:#3fb950;--onbg:#16341f;--accent:#4493f8;--fp:#e87869;--fn:#d9a441;}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;display:flex;flex-direction:column}
  header{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
  header .title{font-weight:600}
  select{background:#222a35;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 6px;font:inherit}
  .pill{padding:3px 10px;border:1px solid var(--line);border-radius:999px;color:var(--mut);cursor:pointer;user-select:none;font-size:12.5px}
  .pill:hover{color:var(--ink);border-color:var(--accent)}
  .pill.act{color:#06131f;background:var(--accent);border-color:var(--accent);font-weight:600}
  .grow{flex:1}
  #progLbl{font-size:12.5px;color:var(--mut)}
  main{flex:1;display:flex;min-height:0}
  #imgwrap{flex:1;display:flex;align-items:center;justify-content:center;padding:14px;min-width:0;background:#0e1116;position:relative}
  #img{max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;box-shadow:0 4px 24px #0008}
  #kind{position:absolute;top:18px;left:18px;font-size:12px;color:var(--mut);background:#0e1116cc;border:1px solid var(--line);border-radius:6px;padding:3px 8px}
  #side{width:440px;border-left:1px solid var(--line);display:flex;flex-direction:column;background:var(--panel)}
  #banner{padding:12px 16px;border-bottom:1px solid var(--line);font-size:14px}
  #banner .tag{font-weight:700;padding:1px 7px;border-radius:5px}
  .tag.FP{background:#3a1c18;color:var(--fp);border:1px solid var(--fp)}
  .tag.FN{background:#3a2e12;color:var(--fn);border:1px solid var(--fn)}
  #banner .sub{color:var(--mut);font-size:12.5px;margin-top:4px}
  #rows{flex:1;overflow:auto;padding:10px}
  .row{display:flex;align-items:center;gap:10px;padding:11px 12px;border:1px solid var(--line);border-radius:9px;margin-bottom:7px;cursor:pointer;background:#1a1f28}
  .row:hover{border-color:var(--accent)} .row.on{background:var(--onbg);border-color:var(--on)}
  .key{width:20px;height:20px;flex:none;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:5px;font-size:11px;color:var(--mut)}
  .row.on .key{color:var(--on);border-color:var(--on)}
  .box{width:18px;height:18px;flex:none;border:2px solid var(--mut);border-radius:5px;position:relative}
  .row.on .box{background:var(--on);border-color:var(--on)}
  .row.on .box:after{content:"";position:absolute;left:5px;top:1px;width:4px;height:10px;border:solid #0c1a10;border-width:0 2px 2px 0;transform:rotate(45deg)}
  .name{flex:1}.name i{font-style:italic}
  .mdl{font-size:11px;color:var(--mut)} .mdl b{color:var(--accent)}
  .mfire{color:var(--accent)}
  #nav{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line);background:#171b22}
  .btn{flex:1;padding:11px 8px;border:1px solid var(--line);border-radius:9px;background:#222a35;color:var(--ink);font:inherit;cursor:pointer}
  .btn:hover{border-color:var(--accent)}
  .btn.primary{flex:1.5;background:var(--accent);border-color:var(--accent);color:#06131f;font-weight:700}
  .btn.accept{background:#16341f;border-color:var(--on);color:var(--on);font-weight:700}
  footer{padding:8px 16px;border-top:1px solid var(--line);color:var(--mut);font-size:12px;background:var(--panel)}
  kbd{background:#222a35;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--ink)}
</style></head>
<body>
<header>
  <span class="title">strip-5 FP/FN review</span>
  <span>species <select id="species"></select></span>
  <span>errors
    <span class="pill" data-kind="FP">false positive</span>
    <span class="pill" data-kind="FN">false negative</span>
    <span class="pill" data-kind="both">both</span>
  </span>
  <span class="pill" id="viewBtn" title="r">overlay ⇄ raw</span>
  <span class="grow"></span>
  <span id="posLbl"></span><span id="progLbl"></span>
</header>
<main>
  <div id="imgwrap"><img id="img" alt=""><div id="kind"></div></div>
  <div id="side">
    <div id="banner"></div>
    <div id="rows"></div>
    <div id="nav">
      <button class="btn" id="prevBtn" title="←">← Prev</button>
      <button class="btn accept" id="acceptBtn" title="a">Accept model <small>a</small></button>
      <button class="btn primary" id="nextBtn" title="→ / Enter">Next →</button>
    </div>
  </div>
</main>
<footer>
  <kbd>1</kbd>–<kbd>9</kbd> toggle &nbsp;·&nbsp; <kbd>a</kbd> accept model for the queued species &nbsp;·&nbsp;
  <kbd>→</kbd>/<kbd>Enter</kbd> next &nbsp;·&nbsp; <kbd>←</kbd> prev &nbsp;·&nbsp; <kbd>r</kbd> overlay/raw
  &nbsp;·&nbsp; <kbd>f</kbd> cycle FP/FN/both &nbsp;·&nbsp; edits → <span id="gtp"></span> (source:"review")
</footer>
<script>
let S=null, species=null, kind='FP', view='overlay', Q=[], pos=0;
const $=s=>document.querySelector(s);

async function init(){
  S=await (await fetch('/api/init')).json();
  species=S.default_species; $('#gtp').textContent=S.gt_path;
  const sel=$('#species'); S.classes.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);});
  sel.value=species;
  if(!S.has_overlay){view='raw';}
  bindUI(); await loadQueue();
}
async function loadQueue(){
  const r=await (await fetch(`/api/queue?species=${encodeURIComponent(species)}&kind=${kind}`)).json();
  Q=r.ids; pos=0;
  document.querySelectorAll('.pill[data-kind]').forEach(p=>p.classList.toggle('act',p.dataset.kind===kind));
  show();
}
function show(){
  if(!Q.length){$('#img').removeAttribute('src');$('#banner').innerHTML=`<b>No ${kind} disagreements for</b> <i>${species}</i> 🎉`;
    $('#rows').innerHTML='';$('#posLbl').textContent='0 / 0';$('#kind').textContent='';return;}
  const id=Q[pos], f=S.frames[id], lab=S.labels[id];
  $('#img').src=`/img?id=${encodeURIComponent(id)}&kind=${view}&t=${Date.now()}`;
  $('#kind').textContent=view==='overlay'?'overlay (model firing painted)':'raw capture';
  const st=status(id), sc=(S.scores[id]||{})[species];
  const scTxt=(sc==null)?'':` · score ${sc.toFixed(3)}`;
  const thr=S.thresholds[species]; const thrTxt=(thr==null)?'':` (τ ${(+thr).toFixed(2)})`;
  const expl=st==='FP'?`model says PRESENT${scTxt}${thrTxt}, GT says ABSENT — is it really there (GT gap) or a model FP?`
            :`model says ABSENT${scTxt}${thrTxt}, GT says PRESENT — did the model miss it, or is the GT wrong?`;
  $('#banner').innerHTML=`<span class="tag ${st}">${st==='FP'?'FALSE POSITIVE':'FALSE NEGATIVE'}</span>
     &nbsp; <i>${species}</i><div class="sub"><b>${id}</b> · leg ${f.leg} · mark ${f.mark}
     · src <b>${lab.source}</b></div><div class="sub">${expl}</div>`;
  $('#posLbl').textContent=`${pos+1} / ${Q.length}`;
  rows(); progress();
}
function status(id){const si=S.classes.indexOf(species);const gt=S.labels[id].vector[si]===1;
  const md=(S.model_present[id]||[]).includes(species);return gt?(md?'TP':'FN'):(md?'FP':'TN');}
function rows(){
  const id=Q[pos], lab=S.labels[id], mp=new Set(S.model_present[id]||[]);
  let h='';
  S.classes.forEach((c,i)=>{
    const fired=mp.has(c), sc=(S.scores[id]||{})[c];
    const badge=fired?`<span class="mfire">model ✓${sc!=null?' '+sc.toFixed(2):''}</span>`
                     :`<span class="mdl">model –${sc!=null?' '+sc.toFixed(2):''}</span>`;
    h+=`<div class="row ${lab.vector[i]?'on':''}" data-i="${i}">
        <div class="key">${i+1}</div><div class="box"></div>
        <div class="name"><i>${c}</i></div><div class="mdl">${badge}</div></div>`;
  });
  $('#rows').innerHTML=h;
  $('#rows').querySelectorAll('.row').forEach(r=>r.onclick=()=>toggle(+r.dataset.i));
}
function toggle(i){const lab=S.labels[Q[pos]];lab.vector[i]=lab.vector[i]?0:1;save(false);rows();}
function acceptModel(){const id=Q[pos];const si=S.classes.indexOf(species);
  const md=(S.model_present[id]||[]).includes(species);S.labels[id].vector[si]=md?1:0;save(true);next();}
function save(markRev){
  const id=Q[pos], lab=S.labels[id]; if(markRev)lab.reviewed=true;
  const names=S.classes.filter((c,i)=>lab.vector[i]);
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,names,reviewed:lab.reviewed,cursor:id})})
   .then(r=>r.json()).then(d=>{if(d.source){lab.source=d.source;}if(d.counts)setCounts(d.counts);});
}
function next(){save(true);if(pos<Q.length-1){pos++;show();}else{show();}}
function prev(){if(pos>0){pos--;show();}}
function progress(){fetch('/api/init');}
function setCounts(c){$('#progLbl').textContent=` · ${c.review_edits} review edits`;}
function bindUI(){
  $('#species').onchange=e=>{species=e.target.value;loadQueue();};
  document.querySelectorAll('.pill[data-kind]').forEach(p=>p.onclick=()=>{kind=p.dataset.kind;loadQueue();});
  $('#viewBtn').onclick=()=>{view=view==='overlay'?'raw':'overlay';show();};
  $('#prevBtn').onclick=prev; $('#nextBtn').onclick=next; $('#acceptBtn').onclick=acceptModel;
  document.addEventListener('keydown',e=>{
    if(e.metaKey||e.ctrlKey||e.altKey||e.target.tagName==='SELECT')return;
    if(e.key>='1'&&e.key<='9'){const n=+e.key-1;if(n<S.classes.length){toggle(n);e.preventDefault();}return;}
    switch(e.key){
      case'ArrowRight':case'Enter':next();e.preventDefault();break;
      case'ArrowLeft':prev();e.preventDefault();break;
      case'a':acceptModel();break;
      case'r':view=view==='overlay'?'raw':'overlay';show();break;
      case'f':kind=kind==='FP'?'FN':kind==='FN'?'both':'FP';loadQueue();break;
    }
  });
}
init();
</script>
</body></html>
"""


def main(argv=None):
    global STORE, DEFAULT_SPECIES
    from . import select_run
    ap = argparse.ArgumentParser(description="Strip FP/FN review app")
    ap.add_argument("--mission", help="which mission to review: site_x/strip_y")
    ap.add_argument("--run", help="exact run id (when a mission has multiple completed runs)")
    ap.add_argument("--missions-root", default=None)
    ap.add_argument("--summary", required=True, help="MIL predictions: inference_viz/summary.json")
    ap.add_argument("--overlays", required=True, help="MIL macro+patch overlays dir")
    ap.add_argument("--scores", default=None, help="optional per-frame any-patch scores .npz")
    ap.add_argument("--species", default=DEFAULT_SPECIES, help="species to hunt first (switchable in UI)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    DEFAULT_SPECIES = args.species
    run = select_run(mission=args.mission, run=args.run, missions_root=args.missions_root)
    captures = str(run.captures_dir)
    gt = str(run.run_dir / "labels" / "image_multilabel.json")

    # one-time safety backup of the shared GT before this app can write it
    if os.path.exists(gt):
        bak = gt + ".prereview"
        if not os.path.exists(bak):
            shutil.copy2(gt, bak)
            print(f"backed up GT -> {bak}")

    STORE = Store(captures, gt, args.summary, args.overlays, args.scores)
    c = STORE.counts()
    url = f"http://127.0.0.1:{args.port}"
    nfp = sum(1 for f in STORE.frames if STORE.status(f["id"], DEFAULT_SPECIES) == "FP")
    nfn = sum(1 for f in STORE.frames if STORE.status(f["id"], DEFAULT_SPECIES) == "FN")
    print(f"reviewing {run.mission_name} / {run.run_id}")
    print(f"FP/FN review · model tiling={STORE.tiling} · {c['total']} frames")
    print(f"{DEFAULT_SPECIES}: {nfp} FP, {nfn} FN to review")
    print(f"shared GT (edited by both apps) -> {gt}")
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
        print("\nstopped. GT saved at", gt)


if __name__ == "__main__":
    main()
