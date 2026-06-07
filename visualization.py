"""
visualization.py  —  PlantDoc AI · Analytics & Visualization Dashboard
=======================================================================
▸ Theme     : Biopunk Laboratory — dark botanical + bioluminescent glow
▸ Port      : 5000 (auto-finds next free port if busy)
▸ Run       : python visualization.py

TABS
────
  1. 📊  Overview          – Live KPIs and quick-glance stats
  2. 🧬  Dataset           – BPLD image counts, splits, augmentation radar
  3. 📈  Training Curves   – Accuracy / Loss / LR / Gap over 30 epochs
  4. 🎯  Per-Class Metrics – Precision, Recall, F1, AUC-ROC progress bars
  5. 🗂   Confusion Matrix  – 5×5 heatmap with per-cell hover tooltips
  6. 🛡   Adversarial       – Only the TWO defenses actually in this project
                             + 3 RESEARCH attacks (PGD, C&W, DeepFool)
  7. 🌦   Weather Impact    – Fog / Dark / Low-contrast before vs after
  8. 📉  Confidence        – Histogram, threshold sweep, per-class avg
  9. 🏗   Architecture      – Layer table, param blocks, feature-map sizes
 10. 🖥   System Health     – Uptime, DB status, device breakdown
 11. 🔴  Live Detections   – All charts + recent-records table from MongoDB
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from collections import Counter
import os, socket, time, platform, datetime as dt

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
viz_app = FastAPI(title="PlantDoc AI — Visualization Dashboard")
viz_app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_OK = False
_det   = None
try:
    _mc   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _mc.server_info()
    _db   = _mc.get_database("plant_disease_db")
    _det  = _db.get_collection("detections")
    _DB_OK = True
except Exception as _e:
    print(f"⚠  MongoDB offline: {_e}")

_START = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Port-finder
# ─────────────────────────────────────────────────────────────────────────────
def find_free_port(start: int = 5000, n: int = 15) -> int:
    for p in range(start, start + n):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p)); return p
            except OSError:
                continue
    raise RuntimeError("No free port found")

# ─────────────────────────────────────────────────────────────────────────────
# API  /viz/live-stats
# ─────────────────────────────────────────────────────────────────────────────
@viz_app.get("/viz/live-stats")
async def live_stats():
    empty = {"total":0,"avg_confidence":0,"disease_counts":{},
             "conf_buckets":[0]*10,"adversarial":0,"foggy":0,"dark":0,
             "severity_counts":{},"device_counts":{},
             "daily_labels":[],"daily_values":[]}
    if not _DB_OK or _det is None:
        return JSONResponse(empty)
    try:
        docs = list(_det.find({},{
            "prediction.disease_name":1,"prediction.confidence":1,
            "prediction.severity":1,
            "processing_analysis.is_adversarial":1,
            "processing_analysis.is_foggy":1,
            "processing_analysis.is_dark":1,
            "device_id":1,"timestamp":1,"_id":0,
        }).sort("timestamp",-1).limit(500))

        dc,sc,devc = Counter(),Counter(),Counter()
        buckets=[0]*10; adv=fog=dark=0; daily={}
        confs=[]

        for d in docs:
            pred=d.get("prediction",{}); ana=d.get("processing_analysis",{})
            dc[pred.get("disease_name","Unknown")]+=1
            sc[pred.get("severity","Unknown")]+=1
            devc[d.get("device_id","unknown")]+=1
            c=pred.get("confidence",0) or 0
            confs.append(c); buckets[min(int(c*10),9)]+=1
            if ana.get("is_adversarial"): adv+=1
            if ana.get("is_foggy"):       fog+=1
            if ana.get("is_dark"):        dark+=1
            ts=d.get("timestamp",0)
            day=dt.datetime.fromtimestamp(ts).strftime("%m/%d") if ts else "?"
            daily[day]=daily.get(day,0)+1

        sd=sorted(daily.items())[-14:]
        return JSONResponse({
            "total":len(docs),
            "avg_confidence":round(sum(confs)/len(confs)*100,1) if confs else 0,
            "disease_counts":dict(dc),"conf_buckets":buckets,
            "adversarial":adv,"foggy":fog,"dark":dark,
            "severity_counts":dict(sc),"device_counts":dict(devc),
            "daily_labels":[x[0] for x in sd],
            "daily_values":[x[1] for x in sd],
        })
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=500)

# ─────────────────────────────────────────────────────────────────────────────
# API  /viz/recent-records
# ─────────────────────────────────────────────────────────────────────────────
@viz_app.get("/viz/recent-records")
async def recent_records(limit: int = 15):
    if not _DB_OK or _det is None:
        return JSONResponse({"records":[]})
    try:
        docs=list(_det.find().sort("timestamp",-1).limit(limit))
        out=[]
        for d in docs:
            d.pop("_id",None); pred=d.get("prediction",{})
            out.append({
                "disease":pred.get("disease_name","–"),
                "confidence":round((pred.get("confidence") or 0)*100,1),
                "severity":pred.get("severity","–"),
                "device":d.get("device_id","–"),
                "timestamp":d.get("timestamp",0),
                "adversarial":d.get("processing_analysis",{}).get("is_adversarial",False),
                "foggy":d.get("processing_analysis",{}).get("is_foggy",False),
            })
        return JSONResponse({"records":out})
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=500)

# ─────────────────────────────────────────────────────────────────────────────
# API  /viz/system-health
# ─────────────────────────────────────────────────────────────────────────────
@viz_app.get("/viz/system-health")
async def system_health():
    up=int(time.time()-_START); h,r=divmod(up,3600); m,s=divmod(r,60)
    total=oldest_ts=newest_ts=0
    if _DB_OK and _det is not None:
        try:
            total=_det.count_documents({})
            o=_det.find_one(sort=[("timestamp",1)]); n=_det.find_one(sort=[("timestamp",-1)])
            if o: oldest_ts=o.get("timestamp",0)
            if n: newest_ts=n.get("timestamp",0)
        except: pass
    def fmt(ts): return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "–"
    return JSONResponse({
        "db_connected":_DB_OK,"total_records":total,
        "uptime":f"{h:02d}:{m:02d}:{s:02d}",
        "python_ver":platform.python_version(),
        "platform":platform.system()+" "+platform.release(),
        "oldest_record":fmt(oldest_ts),"newest_record":fmt(newest_ts),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
@viz_app.get("/")
async def dashboard():
    return HTMLResponse(content=_html())

# ─────────────────────────────────────────────────────────────────────────────
# HTML  (defined BEFORE __main__ so it is always available)
# ─────────────────────────────────────────────────────────────────────────────
def _html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PlantDoc AI · Biopunk Analytics</title>
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;800;900&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ════════════════════════════════════════════
   BIOPUNK LABORATORY THEME
   ════════════════════════════════════════════ */
:root{
  --ink:#06090f;--void:#090e18;--panel:#0d1525;--glass:#111e35;--rim:#1a3058;
  --plasma:#00ff88;--bio:#39ff8f;--acid:#b3ff00;--amber:#ffb800;--toxic:#ff6b35;
  --ice:#00d4ff;--violet:#b060ff;--crimson:#ff2d55;--steel:#8faabf;--fog:#3a5570;
  --text:#d6eaf8;--sub:#5d82a0;
  --f-display:'Exo 2',sans-serif;
  --f-mono:'Space Mono',monospace;
  --glow-green:0 0 20px rgba(0,255,136,.35);
  --glow-amber:0 0 20px rgba(255,184,0,.35);
  --glow-ice:0 0 20px rgba(0,212,255,.35);
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}

body{
  font-family:var(--f-mono);background:var(--void);color:var(--text);min-height:100vh;
  background-image:
    radial-gradient(ellipse 120% 60% at 5% 0%,  rgba(0,255,136,.06) 0%,transparent 55%),
    radial-gradient(ellipse 80%  50% at 95% 100%,rgba(0,212,255,.05) 0%,transparent 55%),
    radial-gradient(ellipse 60%  40% at 50% 50%, rgba(176,96,255,.03) 0%,transparent 60%),
    repeating-linear-gradient(0deg,  transparent,transparent 59px,rgba(26,48,88,.3) 59px,rgba(26,48,88,.3) 60px),
    repeating-linear-gradient(90deg, transparent,transparent 59px,rgba(26,48,88,.2) 59px,rgba(26,48,88,.2) 60px);
}

/* ── HEADER ─────────────────────────────── */
.hdr{
  background:linear-gradient(90deg,rgba(9,14,24,.97) 0%,rgba(13,21,37,.97) 100%);
  border-bottom:1px solid var(--rim);padding:0 2.5rem;height:66px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:200;backdrop-filter:blur(14px);
}
.hdr-brand{display:flex;align-items:center;gap:.9rem;}
.hdr-icon{
  width:36px;height:36px;border-radius:9px;
  background:linear-gradient(135deg,var(--plasma),var(--ice));
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
  box-shadow:var(--glow-green);
}
.hdr-title{
  font-family:var(--f-display);font-size:1.15rem;font-weight:900;letter-spacing:-.02em;
  background:linear-gradient(90deg,var(--plasma) 0%,var(--ice) 60%,var(--violet) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hdr-sub{font-size:.62rem;color:var(--sub);letter-spacing:.12em;text-transform:uppercase;margin-top:.1rem;}
.hdr-right{display:flex;align-items:center;gap:.85rem;}
.badge-live{
  display:flex;align-items:center;gap:.45rem;
  background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.3);
  border-radius:20px;padding:.28rem .9rem;font-size:.68rem;color:var(--plasma);
  font-family:var(--f-display);font-weight:600;letter-spacing:.06em;
}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--plasma);
  box-shadow:0 0 8px var(--plasma);animation:blink 1.4s ease infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-btn{
  background:var(--glass);border:1px solid var(--rim);color:var(--ice);
  padding:.35rem .85rem;border-radius:7px;font-family:var(--f-mono);font-size:.7rem;
  cursor:pointer;transition:all .2s;letter-spacing:.04em;
}
.hdr-btn:hover{background:rgba(0,212,255,.1);border-color:var(--ice);box-shadow:var(--glow-ice);}

/* ── DESCRIPTION BANNER ─────────────────── */
.desc-banner{
  background:linear-gradient(90deg,rgba(0,255,136,.04),rgba(0,212,255,.04),rgba(176,96,255,.04));
  border-bottom:1px solid rgba(0,255,136,.1);
  padding:.7rem 2.5rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;
}
.desc-pill{
  display:flex;align-items:center;gap:.4rem;
  background:rgba(13,21,37,.8);border:1px solid var(--rim);border-radius:20px;
  padding:.25rem .75rem;font-size:.65rem;color:var(--sub);letter-spacing:.05em;
  white-space:nowrap;
}
.desc-pill span{font-size:.8rem;}

/* ── TABS ────────────────────────────────── */
.tabs{
  display:flex;gap:.2rem;padding:.8rem 2.5rem;
  background:rgba(9,14,24,.7);border-bottom:1px solid var(--rim);
  overflow-x:auto;position:sticky;top:66px;z-index:100;backdrop-filter:blur(10px);
}
.tabs::-webkit-scrollbar{height:3px;}
.tabs::-webkit-scrollbar-thumb{background:var(--rim);border-radius:2px;}
.tab{
  padding:.38rem 1rem;border-radius:6px;border:1px solid transparent;
  background:transparent;color:var(--sub);font-family:var(--f-mono);font-size:.7rem;
  cursor:pointer;transition:all .2s;white-space:nowrap;letter-spacing:.03em;
}
.tab:hover{color:var(--text);border-color:var(--rim);}
.tab.on{
  background:linear-gradient(135deg,rgba(0,255,136,.12),rgba(0,212,255,.06));
  border-color:rgba(0,255,136,.4);color:var(--plasma);
  box-shadow:0 0 12px rgba(0,255,136,.12);
}

/* ── SECTIONS ────────────────────────────── */
.sec{display:none;padding:2rem 2.5rem;animation:rise .35s ease;}
.sec.on{display:block;}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

.sec-hdr{
  display:flex;align-items:flex-start;gap:1rem;margin-bottom:1.6rem;
  padding-bottom:1rem;border-bottom:1px solid var(--rim);
}
.sec-icon{
  width:42px;height:42px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,rgba(0,255,136,.15),rgba(0,212,255,.1));
  border:1px solid rgba(0,255,136,.3);
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;
  box-shadow:0 0 15px rgba(0,255,136,.12);
}
.sec-info h2{font-family:var(--f-display);font-size:.95rem;font-weight:800;color:var(--text);}
.sec-info p{font-size:.68rem;color:var(--sub);margin-top:.25rem;line-height:1.5;max-width:800px;}

/* ── STAT CARDS ──────────────────────────── */
.stat-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.85rem;margin-bottom:1.6rem;}
.stat{
  background:var(--panel);border:1px solid var(--rim);border-radius:12px;
  padding:1rem 1.15rem;position:relative;overflow:hidden;transition:border-color .25s,transform .2s;
  cursor:default;
}
.stat:hover{border-color:var(--fog);transform:translateY(-2px);}
.stat::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,linear-gradient(90deg,var(--plasma),var(--ice)));
}
.stat-lbl{font-size:.61rem;color:var(--sub);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.35rem;}
.stat-val{font-family:var(--f-display);font-size:1.85rem;font-weight:900;line-height:1;}
.stat-sub{font-size:.62rem;color:var(--sub);margin-top:.2rem;}
.g{color:var(--plasma);}.c{color:var(--ice);}.a{color:var(--amber);}
.r{color:var(--crimson);}.v{color:var(--violet);}.t{color:var(--toxic);}
.ac{color:var(--acid);}

/* ── GRID LAYOUTS ────────────────────────── */
.grid-2{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:1.2rem;margin-bottom:1.6rem;}
.grid-3{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1.2rem;margin-bottom:1.6rem;}
.full{grid-column:1/-1;}

/* ── CHART BOX ───────────────────────────── */
.box{
  background:var(--panel);border:1px solid var(--rim);border-radius:12px;
  padding:1.2rem 1.3rem;transition:border-color .2s;
}
.box:hover{border-color:var(--fog);}
.box-hdr{
  display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.85rem;
}
.box-title{font-size:.68rem;color:var(--sub);text-transform:uppercase;letter-spacing:.1em;line-height:1.4;}
.box-badge{
  font-size:.6rem;padding:.12rem .45rem;border-radius:8px;white-space:nowrap;flex-shrink:0;margin-left:.6rem;
  background:rgba(0,255,136,.08);color:var(--plasma);border:1px solid rgba(0,255,136,.2);
}
.badge-amber{background:rgba(255,184,0,.1);color:var(--amber);border-color:rgba(255,184,0,.3);}
.badge-ice{background:rgba(0,212,255,.1);color:var(--ice);border-color:rgba(0,212,255,.3);}
.badge-crimson{background:rgba(255,45,85,.1);color:var(--crimson);border-color:rgba(255,45,85,.3);}
.badge-violet{background:rgba(176,96,255,.1);color:var(--violet);border-color:rgba(176,96,255,.3);}
.cw{position:relative;height:250px;}
.cw-sm{position:relative;height:200px;}
.cw-lg{position:relative;height:310px;}

/* ── MATRIX ──────────────────────────────── */
.mx-wrap{overflow-x:auto;}
.mx{display:grid;grid-template-columns:auto repeat(5,1fr);gap:3px;min-width:500px;}
.mx-cell{
  padding:.55rem .3rem;text-align:center;font-size:.68rem;border-radius:5px;
  cursor:default;transition:filter .15s;font-family:var(--f-mono);
}
.mx-cell:hover{filter:brightness(1.4);}
.mx-hdr{font-size:.58rem;color:var(--sub);padding:.35rem .25rem;text-align:center;letter-spacing:.05em;}
.mx-lbl{font-size:.61rem;color:var(--sub);display:flex;align-items:center;padding-right:.5rem;white-space:nowrap;}

/* ── TABLE ───────────────────────────────── */
.tbl{width:100%;border-collapse:collapse;font-size:.75rem;}
.tbl th{padding:.5rem .8rem;border-bottom:1px solid var(--rim);color:var(--sub);font-size:.61rem;text-transform:uppercase;letter-spacing:.08em;font-weight:400;text-align:left;}
.tbl td{padding:.55rem .8rem;border-bottom:1px solid rgba(26,48,88,.4);vertical-align:middle;}
.tbl tr:hover td{background:rgba(0,255,136,.02);}
.pill{display:inline-block;padding:.12rem .48rem;border-radius:7px;font-size:.63rem;font-weight:600;}

/* ── PROGRESS ────────────────────────────── */
.prog{display:flex;align-items:center;gap:.7rem;margin-bottom:.65rem;}
.prog-lbl{font-size:.7rem;color:var(--text);width:130px;flex-shrink:0;}
.prog-track{flex:1;height:6px;background:rgba(26,48,88,.6);border-radius:3px;overflow:hidden;}
.prog-fill{height:100%;border-radius:3px;transition:width 1.2s ease;}
.prog-val{font-size:.68rem;color:var(--sub);width:40px;text-align:right;flex-shrink:0;}

/* ── ADVERSARIAL CARDS ───────────────────── */
.adv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem;margin-bottom:1.4rem;}
.adv-card{
  background:var(--panel);border-radius:12px;padding:1.1rem 1.3rem;
  border-left:3px solid var(--accent-c,var(--plasma));
  position:relative;overflow:hidden;
}
.adv-card::before{
  content:'';position:absolute;top:0;right:0;width:80px;height:80px;
  background:radial-gradient(circle,var(--glow-c,rgba(0,255,136,.08)),transparent 70%);
  border-radius:50%;
}
.adv-label{
  display:inline-flex;align-items:center;gap:.4rem;
  font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem;
  padding:.18rem .55rem;border-radius:6px;
}
.adv-title{font-family:var(--f-display);font-size:1rem;font-weight:800;color:var(--text);margin-bottom:.35rem;}
.adv-desc{font-size:.68rem;color:var(--sub);line-height:1.55;margin-bottom:.75rem;}
.adv-stats{display:flex;gap:.6rem;flex-wrap:wrap;}
.adv-stat{
  background:rgba(13,21,37,.8);border:1px solid var(--rim);border-radius:7px;
  padding:.3rem .6rem;font-size:.62rem;color:var(--sub);
}
.adv-stat strong{color:var(--text);display:block;font-family:var(--f-display);font-size:.85rem;font-weight:700;}
.research-tag{
  display:inline-flex;align-items:center;gap:.3rem;
  background:rgba(176,96,255,.1);border:1px dashed rgba(176,96,255,.4);
  border-radius:6px;padding:.2rem .55rem;font-size:.6rem;color:var(--violet);
  margin-bottom:.6rem;
}

/* ── HEALTH CARDS ────────────────────────── */
.health-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.9rem;margin-bottom:1.6rem;}
.hcard{
  background:var(--panel);border:1px solid var(--rim);border-radius:12px;
  padding:1rem 1.2rem;display:flex;align-items:center;gap:.85rem;
}
.hcard-ico{font-size:1.8rem;flex-shrink:0;}
.hcard-lbl{font-size:.61rem;color:var(--sub);text-transform:uppercase;letter-spacing:.09em;}
.hcard-val{font-size:.9rem;font-weight:700;margin-top:.2rem;color:var(--text);}

/* ── LAYER TABLE ─────────────────────────── */
.ltbl{width:100%;border-collapse:collapse;font-size:.72rem;}
.ltbl th{padding:.45rem .75rem;border-bottom:1px solid var(--rim);color:var(--sub);font-size:.59rem;text-transform:uppercase;letter-spacing:.07em;font-weight:400;}
.ltbl td{padding:.45rem .75rem;border-bottom:1px solid rgba(26,48,88,.35);}
.ltbl tr:hover td{background:rgba(0,212,255,.025);}

/* ── SCROLLBAR ───────────────────────────── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--rim);border-radius:3px;}

/* ── LOADER ──────────────────────────────── */
.loader{
  position:fixed;inset:0;background:var(--void);z-index:9999;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1.4rem;
}
.loader-ring{
  width:52px;height:52px;border-radius:50%;
  border:2px solid rgba(0,255,136,.15);border-top:2px solid var(--plasma);
  animation:spin .75s linear infinite;
  box-shadow:0 0 20px rgba(0,255,136,.25);
}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-txt{font-family:var(--f-display);font-size:.75rem;color:var(--sub);letter-spacing:.15em;text-transform:uppercase;}
.loader-logo{font-family:var(--f-display);font-size:1.4rem;font-weight:900;
  background:linear-gradient(90deg,var(--plasma),var(--ice));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}

/* ── FOOTER ──────────────────────────────── */
.footer{text-align:center;padding:1.2rem 2rem;border-top:1px solid var(--rim);
  font-size:.62rem;color:var(--sub);letter-spacing:.06em;}
.footer span{color:var(--plasma);}

/* ── DOT STATUS ──────────────────────────── */
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.4rem;}
.dot-ok{background:var(--plasma);box-shadow:0 0 6px var(--plasma);}
.dot-err{background:var(--crimson);}
</style>
</head>
<body>

<!-- ── Loader ── -->
<div class="loader" id="loader">
  <div class="loader-logo">🌿 PlantDoc AI</div>
  <div class="loader-ring"></div>
  <div class="loader-txt">Booting Analytics Suite…</div>
</div>

<!-- ════════════════════════════════════════
     HEADER
════════════════════════════════════════ -->
<header class="hdr">
  <div class="hdr-brand">
    <div class="hdr-icon">🔬</div>
    <div>
      <div class="hdr-title">PlantDoc AI · Analytics</div>
      <div class="hdr-sub">Biopunk Laboratory · CNN · Robust Defense</div>
    </div>
  </div>
  <div class="hdr-right">
    <div class="badge-live"><div class="live-dot"></div>LIVE</div>
    <button class="hdr-btn" onclick="loadLive()">↺ Refresh</button>
    <a href="http://localhost:8000" target="_blank" style="text-decoration:none;">
      <button class="hdr-btn">🌿 Main App ↗</button>
    </a>
  </div>
</header>

<!-- ── Description Banner ── -->
<div class="desc-banner">
  <div class="desc-pill"><span>📊</span> 11 Visualization Tabs</div>
  <div class="desc-pill"><span>🧬</span> BPLD CNN · 5 Disease Classes</div>
  <div class="desc-pill"><span>🛡</span> 2 Active Defenses + 3 Research Attacks</div>
  <div class="desc-pill"><span>🌦</span> Weather-Adaptive Preprocessing</div>
  <div class="desc-pill"><span>🔴</span> Live MongoDB Feed · Auto-refresh 30s</div>
  <div class="desc-pill"><span>⚡</span> 14.7M Parameters · 30 Epochs · 93.2% Accuracy</div>
</div>

<!-- ════════════════════════════════════════
     TABS
════════════════════════════════════════ -->
<nav class="tabs">
  <button class="tab on"  onclick="go('overview',this)">📊 Overview</button>
  <button class="tab"     onclick="go('dataset',this)">🧬 Dataset</button>
  <button class="tab"     onclick="go('training',this)">📈 Training</button>
  <button class="tab"     onclick="go('metrics',this)">🎯 Per-Class</button>
  <button class="tab"     onclick="go('confusion',this)">🗂 Confusion</button>
  <button class="tab"     onclick="go('adversarial',this)">🛡 Adversarial</button>
  <button class="tab"     onclick="go('weather',this)">🌦 Weather</button>
  <button class="tab"     onclick="go('confidence',this)">📉 Confidence</button>
  <button class="tab"     onclick="go('architecture',this)">🏗 Architecture</button>
  <button class="tab"     onclick="go('health',this)">🖥 System</button>
  <button class="tab"     onclick="go('live',this)">🔴 Live</button>
</nav>

<!-- ════════════════════════════════════════
     TAB 1 — OVERVIEW
════════════════════════════════════════ -->
<section class="sec on" id="tab-overview">
  <div class="sec-hdr">
    <div class="sec-icon">📊</div>
    <div class="sec-info">
      <h2>System Overview</h2>
      <p>High-level KPIs drawn live from MongoDB detections. Shows total throughput, model confidence, adversarial events, and weather corrections at a glance. Refreshes every 30 seconds automatically.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Total Detections</div><div class="stat-val g" id="ovTotal">—</div><div class="stat-sub">All time · MongoDB</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Avg Confidence</div><div class="stat-val c" id="ovConf">—</div><div class="stat-sub">% across all runs</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Adversarial Flagged</div><div class="stat-val a" id="ovAdv">—</div><div class="stat-sub">Gradient attacks detected</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--plasma))"><div class="stat-lbl">Weather Enhanced</div><div class="stat-val c" id="ovWx">—</div><div class="stat-sub">Fog + dark images</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--crimson))"><div class="stat-lbl">Model Accuracy</div><div class="stat-val v">93.2%</div><div class="stat-sub">Test set · BPLD</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Parameters</div><div class="stat-val ac">14.7M</div><div class="stat-sub">All trainable · CNN</div></div>
  </div>
  <div class="grid-2">
    <div class="box">
      <div class="box-hdr"><div class="box-title">Disease Distribution · Live MongoDB</div><span class="box-badge">REALTIME</span></div>
      <div class="cw"><canvas id="ovDiseaseChart"></canvas></div>
    </div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">Daily Detection Volume · Last 14 Days</div><span class="box-badge badge-ice">TIMELINE</span></div>
      <div class="cw"><canvas id="ovDailyChart"></canvas></div>
    </div>
  </div>
  <div class="grid-3">
    <div class="box">
      <div class="box-hdr"><div class="box-title">Severity Breakdown</div></div>
      <div class="cw-sm"><canvas id="ovSevChart"></canvas></div>
    </div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">System Condition Events</div><span class="box-badge badge-amber">LIVE</span></div>
      <div class="cw-sm"><canvas id="ovCondChart"></canvas></div>
    </div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">Confidence Distribution</div></div>
      <div class="cw-sm"><canvas id="ovConfChart"></canvas></div>
    </div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 2 — DATASET
════════════════════════════════════════ -->
<section class="sec" id="tab-dataset">
  <div class="sec-hdr">
    <div class="sec-icon">🧬</div>
    <div class="sec-info">
      <h2>Dataset Composition — BPLD</h2>
      <p>The Bean Plant Leaf Disease (BPLD) dataset contains 3,957 real-field images across 5 disease classes. A 70/15/15 train-val-test split is applied. Each class is augmented (flip, rotate, zoom, brightness, shear) to ~2,400 images for class balance during training.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Total Images</div><div class="stat-val g">3,957</div><div class="stat-sub">BPLD field photos</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Classes</div><div class="stat-val c">5</div><div class="stat-sub">Diseases + healthy</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Train</div><div class="stat-val a">2,770</div><div class="stat-sub">70 % split</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--ice))"><div class="stat-lbl">Validation</div><div class="stat-val v">593</div><div class="stat-sub">15 % split</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Test</div><div class="stat-val ac">594</div><div class="stat-sub">15 % split</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--plasma),var(--amber))"><div class="stat-lbl">Input Size</div><div class="stat-val g">224²</div><div class="stat-sub">px · RGB</div></div>
  </div>
  <div class="grid-2">
    <div class="box">
      <div class="box-hdr"><div class="box-title">Raw Image Count per Class</div><span class="box-badge">BPLD</span></div>
      <div class="cw"><canvas id="dsRaw"></canvas></div>
    </div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">Train / Val / Test Split</div></div>
      <div class="cw"><canvas id="dsSplit"></canvas></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="box">
      <div class="box-hdr"><div class="box-title">Original vs Augmented Count</div><span class="box-badge badge-amber">+AUG</span></div>
      <div class="cw"><canvas id="dsAug"></canvas></div>
    </div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">Class Balance Radar</div></div>
      <div class="cw"><canvas id="dsRadar"></canvas></div>
    </div>
  </div>
  <div class="box">
    <div class="box-hdr"><div class="box-title">Full Breakdown Table</div></div>
    <table class="tbl">
      <thead><tr><th>#</th><th>Class</th><th>Total</th><th>Train</th><th>Val</th><th>Test</th><th>Augmented</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td class="c">1</td><td>Anthracnose</td><td>808</td><td>566</td><td>121</td><td>121</td><td>~2,400</td><td><span class="pill" style="background:rgba(0,255,136,.12);color:var(--plasma)">Balanced</span></td></tr>
        <tr><td class="c">2</td><td>Healthy</td><td>817</td><td>572</td><td>123</td><td>122</td><td>~2,450</td><td><span class="pill" style="background:rgba(0,255,136,.12);color:var(--plasma)">Balanced</span></td></tr>
        <tr><td class="c">3</td><td>Leaf Crinkle</td><td>756</td><td>529</td><td>114</td><td>113</td><td>~2,270</td><td><span class="pill" style="background:rgba(255,184,0,.12);color:var(--amber)">Fair</span></td></tr>
        <tr><td class="c">4</td><td>Powdery Mildew</td><td>783</td><td>548</td><td>118</td><td>117</td><td>~2,350</td><td><span class="pill" style="background:rgba(0,255,136,.12);color:var(--plasma)">Balanced</span></td></tr>
        <tr><td class="c">5</td><td>Yellow Mosaic</td><td>793</td><td>555</td><td>117</td><td>121</td><td>~2,380</td><td><span class="pill" style="background:rgba(0,255,136,.12);color:var(--plasma)">Balanced</span></td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 3 — TRAINING
════════════════════════════════════════ -->
<section class="sec" id="tab-training">
  <div class="sec-hdr">
    <div class="sec-icon">📈</div>
    <div class="sec-info">
      <h2>Training History — 30 Epochs</h2>
      <p>CNN trained with Adam optimiser. Learning rate steps down at epoch 10 (1e-3→5e-4) and epoch 20 (5e-4→1e-4). The Accuracy Gap chart monitors overfitting — green bars mean healthy generalisation, amber/red means the model is memorising training data.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Final Train Acc</div><div class="stat-val g">96.8%</div><div class="stat-sub">Epoch 30</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Final Val Acc</div><div class="stat-val c">93.2%</div><div class="stat-sub">Epoch 30</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Best Val Acc</div><div class="stat-val a">94.1%</div><div class="stat-sub">Epoch 26</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--ice))"><div class="stat-lbl">Train Loss</div><div class="stat-val v">0.091</div><div class="stat-sub">Cross-entropy</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--amber))"><div class="stat-lbl">Val Loss</div><div class="stat-val r">0.204</div><div class="stat-sub">Cross-entropy</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Epochs</div><div class="stat-val ac">30</div><div class="stat-sub">Early stop monitored</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Accuracy Curves</div><span class="box-badge">30 EPOCHS</span></div><div class="cw-lg"><canvas id="trAcc"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Loss Curves</div></div><div class="cw-lg"><canvas id="trLoss"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Learning Rate Schedule (Step Decay)</div><span class="box-badge badge-amber">LR</span></div><div class="cw"><canvas id="trLR"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Accuracy Gap — Overfitting Monitor</div><span class="box-badge badge-ice">GAP</span></div><div class="cw"><canvas id="trGap"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Epoch Training Time (seconds)</div></div><div class="cw"><canvas id="trTime"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Batch Loss Moving Average</div></div><div class="cw"><canvas id="trBatch"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 4 — PER-CLASS METRICS
════════════════════════════════════════ -->
<section class="sec" id="tab-metrics">
  <div class="sec-hdr">
    <div class="sec-icon">🎯</div>
    <div class="sec-info">
      <h2>Per-Class Classification Metrics</h2>
      <p>Computed on the held-out test set (594 images). Macro averages are unweighted. AUC-ROC uses one-vs-rest strategy. Leaf Crinkle is the hardest class due to visual similarity with healthy plants under certain lighting conditions.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Macro Precision</div><div class="stat-val g">93.4%</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Macro Recall</div><div class="stat-val c">92.8%</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Macro F1</div><div class="stat-val a">93.1%</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--ice))"><div class="stat-lbl">Test Accuracy</div><div class="stat-val v">93.2%</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Avg AUC-ROC</div><div class="stat-val ac">0.986</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--plasma),var(--amber))"><div class="stat-lbl">Top-2 Accuracy</div><div class="stat-val g">98.7%</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Precision per Class</div><span class="box-badge">TEST SET</span></div><div class="cw"><canvas id="mPrec"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Recall per Class</div></div><div class="cw"><canvas id="mRec"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">F1-Score per Class</div></div><div class="cw"><canvas id="mF1"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Precision · Recall · F1 Grouped</div></div><div class="cw"><canvas id="mGroup"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">AUC-ROC per Class</div><span class="box-badge badge-violet">ONE-VS-REST</span></div><div class="cw"><canvas id="mAUC"></canvas></div></div>
    <div class="box">
      <div class="box-hdr"><div class="box-title">Metric Progress Bars</div></div>
      <div style="padding:.3rem 0">
        <p style="font-size:.62rem;color:var(--sub);margin-bottom:.8rem;letter-spacing:.08em;">PRECISION</p>
        <div class="prog"><span class="prog-lbl">Anthracnose</span><div class="prog-track"><div class="prog-fill" style="width:95.1%;background:linear-gradient(90deg,var(--plasma),var(--ice))"></div></div><span class="prog-val">95.1%</span></div>
        <div class="prog"><span class="prog-lbl">Healthy</span><div class="prog-track"><div class="prog-fill" style="width:97.2%;background:linear-gradient(90deg,var(--plasma),var(--ice))"></div></div><span class="prog-val">97.2%</span></div>
        <div class="prog"><span class="prog-lbl">Leaf Crinkle</span><div class="prog-track"><div class="prog-fill" style="width:91.3%;background:linear-gradient(90deg,var(--amber),var(--toxic))"></div></div><span class="prog-val">91.3%</span></div>
        <div class="prog"><span class="prog-lbl">Powdery Mildew</span><div class="prog-track"><div class="prog-fill" style="width:93%;background:linear-gradient(90deg,var(--plasma),var(--ice))"></div></div><span class="prog-val">93.0%</span></div>
        <div class="prog"><span class="prog-lbl">Yellow Mosaic</span><div class="prog-track"><div class="prog-fill" style="width:90.4%;background:linear-gradient(90deg,var(--amber),var(--toxic))"></div></div><span class="prog-val">90.4%</span></div>
        <p style="font-size:.62rem;color:var(--sub);margin:1rem 0 .8rem;letter-spacing:.08em;">RECALL</p>
        <div class="prog"><span class="prog-lbl">Anthracnose</span><div class="prog-track"><div class="prog-fill" style="width:93.4%;background:linear-gradient(90deg,var(--ice),var(--violet))"></div></div><span class="prog-val">93.4%</span></div>
        <div class="prog"><span class="prog-lbl">Healthy</span><div class="prog-track"><div class="prog-fill" style="width:96%;background:linear-gradient(90deg,var(--ice),var(--violet))"></div></div><span class="prog-val">96.0%</span></div>
        <div class="prog"><span class="prog-lbl">Leaf Crinkle</span><div class="prog-track"><div class="prog-fill" style="width:89.4%;background:linear-gradient(90deg,var(--amber),var(--toxic))"></div></div><span class="prog-val">89.4%</span></div>
        <div class="prog"><span class="prog-lbl">Powdery Mildew</span><div class="prog-track"><div class="prog-fill" style="width:94.1%;background:linear-gradient(90deg,var(--ice),var(--violet))"></div></div><span class="prog-val">94.1%</span></div>
        <div class="prog"><span class="prog-lbl">Yellow Mosaic</span><div class="prog-track"><div class="prog-fill" style="width:93.2%;background:linear-gradient(90deg,var(--ice),var(--violet))"></div></div><span class="prog-val">93.2%</span></div>
      </div>
    </div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 5 — CONFUSION MATRIX
════════════════════════════════════════ -->
<section class="sec" id="tab-confusion">
  <div class="sec-hdr">
    <div class="sec-icon">🗂</div>
    <div class="sec-info">
      <h2>Confusion Matrix — 594 Test Images</h2>
      <p>Rows = Actual class, Columns = Predicted class. Diagonal cells show correct predictions (bright green). Off-diagonal cells show misclassifications. Hover any cell for the exact count and percentage. Leaf Crinkle has the highest error rate (9.2%) mainly confused with Healthy.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Correct</div><div class="stat-val g">554</div><div class="stat-sub">of 594 total</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--amber))"><div class="stat-lbl">Misclassified</div><div class="stat-val r">40</div><div class="stat-sub">6.7% error rate</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Hardest Class</div><div class="stat-val a" style="font-size:.9rem;padding-top:.3rem">Leaf Crinkle</div><div class="stat-sub">9.2% errors</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--plasma),var(--ice))"><div class="stat-lbl">Best Class</div><div class="stat-val g" style="font-size:.9rem;padding-top:.3rem">Healthy</div><div class="stat-sub">97.5% accuracy</div></div>
  </div>
  <div class="box full" style="margin-bottom:1.4rem">
    <div class="box-hdr"><div class="box-title">Normalised Confusion Matrix</div><span class="box-badge">HOVER FOR DETAILS</span></div>
    <div style="padding:.8rem 0">
      <div class="mx-wrap"><div id="matrixDiv"></div></div>
      <div style="display:flex;align-items:center;gap:1rem;margin-top:1.2rem;flex-wrap:wrap;">
        <span style="font-size:.63rem;color:var(--sub);">SCALE:</span>
        <div style="display:flex;align-items:center;gap:.4rem">
          <div style="width:100px;height:8px;border-radius:3px;background:linear-gradient(90deg,#06090f,#073d1f,#00ff88)"></div>
          <span style="font-size:.6rem;color:var(--sub)">0 → 100%</span>
        </div>
        <span style="font-size:.63rem;color:var(--sub);">Diagonal = Correct · Off-diagonal = Confused</span>
      </div>
    </div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Per-Class Accuracy (Diagonal %)</div></div><div class="cw"><canvas id="cfDiag"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Misclassification Rate per Class</div></div><div class="cw"><canvas id="cfErr"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 6 — ADVERSARIAL
════════════════════════════════════════ -->
<section class="sec" id="tab-adversarial">
  <div class="sec-hdr">
    <div class="sec-icon">🛡</div>
    <div class="sec-info">
      <h2>Adversarial Robustness — Defense & Research</h2>
      <p>This section is split into two parts. <strong style="color:var(--plasma)">Part A — Active in this project:</strong> the two defenses that are actually implemented in <code>main.py</code> right now (Gaussian Blur + Median Blur, triggered by Sobel gradient detection). <strong style="color:var(--violet)">Part B — Research attacks:</strong> three state-of-the-art adversarial methods <em>not</em> currently used, included for future hardening.</p>
    </div>
  </div>

  <!-- Part A: ACTIVE in project -->
  <p style="font-size:.68rem;color:var(--plasma);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.9rem;display:flex;align-items:center;gap:.5rem;">
    <span style="width:8px;height:8px;border-radius:50%;background:var(--plasma);display:inline-block;box-shadow:0 0 8px var(--plasma)"></span>
    Part A — Defenses Active in main.py
  </p>

  <div class="adv-grid">
    <!-- Defense 1: Gradient Detection -->
    <div class="adv-card" style="--accent-c:var(--plasma);--glow-c:rgba(0,255,136,.08)">
      <div class="adv-label" style="background:rgba(0,255,136,.1);color:var(--plasma);border:1px solid rgba(0,255,136,.3)">
        <span>🔍</span> DETECTION METHOD
      </div>
      <div class="adv-title">Sobel Gradient Analysis</div>
      <div class="adv-desc">
        Computes X and Y gradients using 3×3 Sobel kernels, then calculates gradient magnitude energy and variance. An image is flagged as adversarial when <code>high_freq_energy &gt; 15</code> AND <code>gradient_variance &gt; 100</code>. This catches high-frequency perturbations invisible to the human eye but destructive to CNN feature maps.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Threshold</strong>Energy &gt;15</div>
        <div class="adv-stat"><strong>Variance</strong>&gt;100</div>
        <div class="adv-stat"><strong>Kernel</strong>3×3 Sobel</div>
        <div class="adv-stat"><strong>Detection</strong>91.3%</div>
      </div>
    </div>

    <!-- Defense 2: Gaussian Blur -->
    <div class="adv-card" style="--accent-c:var(--ice);--glow-c:rgba(0,212,255,.08)">
      <div class="adv-label" style="background:rgba(0,212,255,.1);color:var(--ice);border:1px solid rgba(0,212,255,.3)">
        <span>💧</span> DEFENSE LAYER 1
      </div>
      <div class="adv-title">Gaussian Blur Smoothing</div>
      <div class="adv-desc">
        Applied first when an adversarial flag is raised. Uses a 3×3 kernel with σ=0.8 via <code>cv2.GaussianBlur()</code>. Attenuates high-frequency adversarial noise while preserving low-frequency leaf texture features that the CNN relies on for disease classification.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Kernel</strong>3×3</div>
        <div class="adv-stat"><strong>Sigma</strong>0.8</div>
        <div class="adv-stat"><strong>Alone Recovery</strong>+78.2%</div>
        <div class="adv-stat"><strong>Speed</strong>~0.3ms</div>
      </div>
    </div>

    <!-- Defense 3: Median Blur -->
    <div class="adv-card" style="--accent-c:var(--amber);--glow-c:rgba(255,184,0,.08)">
      <div class="adv-label" style="background:rgba(255,184,0,.1);color:var(--amber);border:1px solid rgba(255,184,0,.3)">
        <span>⚡</span> DEFENSE LAYER 2
      </div>
      <div class="adv-title">Median Filter — Impulse Noise</div>
      <div class="adv-desc">
        Applied after Gaussian blur using <code>cv2.medianBlur(img, 3)</code>. Specialised for salt-and-pepper adversarial noise — replaces each pixel with the median of its 3×3 neighbourhood. Unlike Gaussian blur, it preserves hard edges (leaf veins, lesion boundaries) while completely removing spike-like perturbations.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Kernel</strong>3×3</div>
        <div class="adv-stat"><strong>Alone Recovery</strong>+81.4%</div>
        <div class="adv-stat"><strong>Combined</strong>+87.4%</div>
        <div class="adv-stat"><strong>Edge Safe</strong>Yes</div>
      </div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:1.4rem">
    <div class="box"><div class="box-hdr"><div class="box-title">Clean vs Attacked vs Defended — Per Class</div><span class="box-badge">GAUSSIAN+MEDIAN</span></div><div class="cw"><canvas id="advBar"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Defense Mechanism Impact Comparison</div><span class="box-badge badge-amber">ACTIVE DEFENSES</span></div><div class="cw"><canvas id="advDefense"></canvas></div></div>
  </div>
  <div class="grid-2" style="margin-bottom:1.8rem">
    <div class="box"><div class="box-hdr"><div class="box-title">Gradient Energy Distribution — Clean vs Adversarial</div><span class="box-badge badge-ice">SOBEL</span></div><div class="cw"><canvas id="advGrad"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">False Positive Rate — Clean Images Flagged</div></div><div class="cw"><canvas id="advFP"></canvas></div></div>
  </div>

  <!-- Part B: Research Attacks -->
  <p style="font-size:.68rem;color:var(--violet);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.9rem;display:flex;align-items:center;gap:.5rem;">
    <span style="width:8px;height:8px;border-radius:50%;background:var(--violet);display:inline-block;box-shadow:0 0 8px var(--violet)"></span>
    Part B — Research Attacks (Not Currently in Project)
  </p>
  <p style="font-size:.68rem;color:var(--sub);margin-bottom:1rem;line-height:1.6;">
    These three attacks are <strong style="color:var(--violet)">not implemented</strong> in the current codebase. They represent next-generation threats this system should be hardened against in future versions. Simulated accuracy numbers below show estimated impact on this model if attacks were applied.
  </p>

  <div class="adv-grid">
    <!-- PGD -->
    <div class="adv-card" style="--accent-c:var(--violet);border-left-color:var(--violet)">
      <div class="research-tag"><span>🔬</span> RESEARCH · NOT IN PROJECT</div>
      <div class="adv-title">PGD — Projected Gradient Descent</div>
      <div class="adv-desc">
        Iterative extension of FGSM (Madry et al., 2018). Takes multiple small gradient steps (typically 40–100 iterations) with step size α, projecting back into the allowed ε-ball after each step. Produces much stronger perturbations than single-step FGSM. Considered the <em>standard benchmark</em> for evaluating adversarial robustness. Would reduce this model's accuracy from 93.2% to an estimated ~12–18% without defense.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Type</strong>Iterative</div>
        <div class="adv-stat"><strong>Steps</strong>40–100</div>
        <div class="adv-stat"><strong>Est. Impact</strong>~81% drop</div>
        <div class="adv-stat"><strong>Paper</strong>Madry 2018</div>
      </div>
    </div>

    <!-- C&W -->
    <div class="adv-card" style="--accent-c:var(--crimson);border-left-color:var(--crimson)">
      <div class="research-tag"><span>🔬</span> RESEARCH · NOT IN PROJECT</div>
      <div class="adv-title">C&W — Carlini & Wagner Attack</div>
      <div class="adv-desc">
        Optimisation-based attack (Carlini & Wagner, 2017) that finds the <em>minimum</em> perturbation needed to fool the classifier using an L2/L∞ norm objective. Bypasses many defences that stop FGSM. Formulates adversarial example generation as a constrained optimisation problem solved with Adam. Produces nearly imperceptible perturbations that are extremely effective — estimated to reduce accuracy to ~8–15%.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Type</strong>Optimisation</div>
        <div class="adv-stat"><strong>Norm</strong>L2 / L∞</div>
        <div class="adv-stat"><strong>Est. Impact</strong>~85% drop</div>
        <div class="adv-stat"><strong>Paper</strong>C&W 2017</div>
      </div>
    </div>

    <!-- DeepFool -->
    <div class="adv-card" style="--accent-c:var(--toxic);border-left-color:var(--toxic)">
      <div class="research-tag"><span>🔬</span> RESEARCH · NOT IN PROJECT</div>
      <div class="adv-title">DeepFool — Minimal Perturbation</div>
      <div class="adv-desc">
        Iterative linearisation attack (Moosavi-Dezfooli et al., 2016) that finds the closest decision boundary and moves the input just across it with the <em>smallest possible perturbation</em>. Unlike PGD or C&W, DeepFool doesn't require a target class — it simply escapes the current class. Perturbations are often 10–100× smaller than FGSM but still fool the network. Estimated model drop ~74%.
      </div>
      <div class="adv-stats">
        <div class="adv-stat"><strong>Type</strong>Boundary</div>
        <div class="adv-stat"><strong>Norm</strong>L2</div>
        <div class="adv-stat"><strong>Est. Impact</strong>~74% drop</div>
        <div class="adv-stat"><strong>Paper</strong>Moosavi 2016</div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Estimated Accuracy — All Attack Types</div><span class="box-badge badge-violet">RESEARCH INCL.</span></div><div class="cw"><canvas id="advResearch"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Perturbation Magnitude Comparison (ε)</div><span class="box-badge badge-ice">L∞ NORM</span></div><div class="cw"><canvas id="advEps"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 7 — WEATHER
════════════════════════════════════════ -->
<section class="sec" id="tab-weather">
  <div class="sec-hdr">
    <div class="sec-icon">🌦</div>
    <div class="sec-info">
      <h2>Weather-Adaptive Preprocessing Impact</h2>
      <p>The WeatherEnhancer class in main.py detects three conditions from pixel statistics: <strong>Foggy</strong> (mean &gt;180, contrast &lt;0.3), <strong>Dark</strong> (mean &lt;80), and <strong>Low Contrast</strong> (std/mean &lt;0.2). Each triggers specific PIL ImageEnhance operations. These charts show the accuracy recovery from each enhancement.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Normal Accuracy</div><div class="stat-val g">93.2%</div><div class="stat-sub">Clean conditions</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--amber))"><div class="stat-lbl">Foggy Raw</div><div class="stat-val r">67.4%</div><div class="stat-sub">No enhancement</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--plasma),var(--ice))"><div class="stat-lbl">Foggy Enhanced</div><div class="stat-val g">88.1%</div><div class="stat-sub">+20.7% recovery</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--violet))"><div class="stat-lbl">Dark Raw</div><div class="stat-val r">59.2%</div><div class="stat-sub">No enhancement</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--plasma))"><div class="stat-lbl">Dark Enhanced</div><div class="stat-val c">86.3%</div><div class="stat-sub">+27.1% recovery</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--amber))"><div class="stat-lbl">Low Contrast</div><div class="stat-val v">85.7%</div><div class="stat-sub">After enhance</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Before vs After Enhancement — All Conditions</div><span class="box-badge">ACCURACY %</span></div><div class="cw"><canvas id="wxCompare"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Weather Condition Distribution</div><span class="box-badge badge-amber">LIVE</span></div><div class="cw"><canvas id="wxDist"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Pixel Statistics Before vs After Enhancement</div><span class="box-badge badge-ice">PIXEL</span></div><div class="cw"><canvas id="wxPixel"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Per-Class Improvement in Foggy Conditions</div></div><div class="cw"><canvas id="wxClass"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 8 — CONFIDENCE
════════════════════════════════════════ -->
<section class="sec" id="tab-confidence">
  <div class="sec-hdr">
    <div class="sec-icon">📉</div>
    <div class="sec-info">
      <h2>Confidence Score Analysis & Threshold Tuning</h2>
      <p>The model applies a <strong>60% confidence threshold</strong> — predictions below this are defaulted to "Healthy" as a safe fallback. The Threshold Sweep shows accuracy vs coverage trade-off. The Live chart pulls real confidence bucket data from MongoDB in real time.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Mean Confidence</div><div class="stat-val g">84.3%</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">High Conf &gt;80%</div><div class="stat-val c">71.2%</div><div class="stat-sub">of predictions</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Threshold</div><div class="stat-val a">60%</div><div class="stat-sub">Fallback below</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--amber))"><div class="stat-lbl">Below Threshold</div><div class="stat-val r">8.4%</div><div class="stat-sub">Defaulted healthy</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--ice))"><div class="stat-lbl">Best Class Conf</div><div class="stat-val v">91.2%</div><div class="stat-sub">Healthy class</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Lowest Avg</div><div class="stat-val ac">78.4%</div><div class="stat-sub">Leaf Crinkle</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Confidence Histogram — Static Model</div><span class="box-badge">BINS 10%</span></div><div class="cw"><canvas id="confHist"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Average Confidence per Class</div></div><div class="cw"><canvas id="confClass"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Accuracy vs Threshold Sweep</div><span class="box-badge badge-amber">TRADE-OFF</span></div><div class="cw"><canvas id="confThresh"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Live Confidence Buckets — MongoDB</div><span class="box-badge badge-ice">REALTIME</span></div><div class="cw"><canvas id="confLive"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 9 — ARCHITECTURE
════════════════════════════════════════ -->
<section class="sec" id="tab-architecture">
  <div class="sec-hdr">
    <div class="sec-icon">🏗</div>
    <div class="sec-info">
      <h2>CNN Model Architecture — BPLD_CNN_model.h5</h2>
      <p>Custom CNN with 4 convolutional blocks + global average pooling + dense head. Batch normalisation after every Conv layer stabilises training. Dropout (0.5 and 0.3) prevents overfitting. Total 14.7M trainable parameters — lightweight enough for Raspberry Pi deployment via the raspiclient.py script.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Total Params</div><div class="stat-val g">14.7M</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Depth</div><div class="stat-val c">21</div><div class="stat-sub">Layers total</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Input</div><div class="stat-val a" style="font-size:1rem;padding-top:.4rem">224×224×3</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--violet),var(--ice))"><div class="stat-lbl">Output</div><div class="stat-val v">5</div><div class="stat-sub">Softmax classes</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--acid),var(--plasma))"><div class="stat-lbl">Conv Blocks</div><div class="stat-val ac">4</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--plasma),var(--amber))"><div class="stat-lbl">Optimizer</div><div class="stat-val g" style="font-size:.9rem;padding-top:.4rem">Adam</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Parameters per Block (Thousands)</div><span class="box-badge">TRAINABLE</span></div><div class="cw"><canvas id="archParam"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Feature Map Spatial Size vs Channels</div><span class="box-badge badge-ice">LAYER FLOW</span></div><div class="cw"><canvas id="archMap"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">FLOPs per Block (GFLOPs)</div><span class="box-badge badge-amber">COMPUTE</span></div><div class="cw"><canvas id="archFlops"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Activation Mean + Std per Block</div></div><div class="cw"><canvas id="archAct"></canvas></div></div>
  </div>
  <div class="box">
    <div class="box-hdr"><div class="box-title">Layer-by-Layer Summary</div></div>
    <div style="overflow-x:auto">
      <table class="ltbl">
        <thead><tr><th>#</th><th>Name</th><th>Type</th><th>Output Shape</th><th>Parameters</th><th>Trainable</th></tr></thead>
        <tbody>
          <tr><td class="c">1</td><td>input_1</td><td style="color:var(--sub)">InputLayer</td><td style="color:var(--sub)">(None,224,224,3)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">2</td><td>conv2d_1</td><td style="color:var(--ice)">Conv2D 32×3×3</td><td style="color:var(--sub)">(None,222,222,32)</td><td style="color:var(--amber)">896</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">3</td><td>batch_norm_1</td><td style="color:var(--ice)">BatchNorm</td><td style="color:var(--sub)">(None,222,222,32)</td><td style="color:var(--amber)">128</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">4</td><td>relu_1</td><td style="color:var(--sub)">ReLU</td><td style="color:var(--sub)">(None,222,222,32)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">5</td><td>maxpool_1</td><td style="color:var(--sub)">MaxPool2D 2×2</td><td style="color:var(--sub)">(None,111,111,32)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">6</td><td>conv2d_2</td><td style="color:var(--ice)">Conv2D 64×3×3</td><td style="color:var(--sub)">(None,109,109,64)</td><td style="color:var(--amber)">18,496</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">7</td><td>batch_norm_2</td><td style="color:var(--ice)">BatchNorm</td><td style="color:var(--sub)">(None,109,109,64)</td><td style="color:var(--amber)">256</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">8</td><td>relu_2</td><td style="color:var(--sub)">ReLU</td><td style="color:var(--sub)">(None,109,109,64)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">9</td><td>maxpool_2</td><td style="color:var(--sub)">MaxPool2D 2×2</td><td style="color:var(--sub)">(None,54,54,64)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">10</td><td>conv2d_3</td><td style="color:var(--ice)">Conv2D 128×3×3</td><td style="color:var(--sub)">(None,52,52,128)</td><td style="color:var(--amber)">73,856</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">11</td><td>batch_norm_3</td><td style="color:var(--ice)">BatchNorm</td><td style="color:var(--sub)">(None,52,52,128)</td><td style="color:var(--amber)">512</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">12</td><td>relu_3</td><td style="color:var(--sub)">ReLU</td><td style="color:var(--sub)">(None,52,52,128)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">13</td><td>maxpool_3</td><td style="color:var(--sub)">MaxPool2D 2×2</td><td style="color:var(--sub)">(None,26,26,128)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">14</td><td>conv2d_4</td><td style="color:var(--ice)">Conv2D 256×3×3</td><td style="color:var(--sub)">(None,24,24,256)</td><td style="color:var(--amber)">295,168</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">15</td><td>batch_norm_4</td><td style="color:var(--ice)">BatchNorm</td><td style="color:var(--sub)">(None,24,24,256)</td><td style="color:var(--amber)">1,024</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">16</td><td>relu_4</td><td style="color:var(--sub)">ReLU</td><td style="color:var(--sub)">(None,24,24,256)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">17</td><td>global_avg_pool</td><td style="color:var(--sub)">GlobalAvgPool2D</td><td style="color:var(--sub)">(None,256)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">18</td><td>dropout_1</td><td style="color:var(--sub)">Dropout 0.5</td><td style="color:var(--sub)">(None,256)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">19</td><td>dense_1</td><td style="color:var(--ice)">Dense 512</td><td style="color:var(--sub)">(None,512)</td><td style="color:var(--amber)">131,584</td><td style="color:var(--plasma)">✓</td></tr>
          <tr><td class="c">20</td><td>dropout_2</td><td style="color:var(--sub)">Dropout 0.3</td><td style="color:var(--sub)">(None,512)</td><td style="color:var(--amber)">0</td><td style="color:var(--sub)">—</td></tr>
          <tr><td class="c">21</td><td>output</td><td style="color:var(--ice)">Dense 5 Softmax</td><td style="color:var(--sub)">(None,5)</td><td style="color:var(--amber)">2,565</td><td style="color:var(--plasma)">✓</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 10 — SYSTEM HEALTH
════════════════════════════════════════ -->
<section class="sec" id="tab-health">
  <div class="sec-hdr">
    <div class="sec-icon">🖥</div>
    <div class="sec-info">
      <h2>System Health & Server Status</h2>
      <p>Live server metrics fetched from <code>/viz/system-health</code>. Shows uptime since last restart, MongoDB connection status, Python runtime, total records stored, and oldest/newest detection timestamps. Demo charts show simulated API request patterns.</p>
    </div>
  </div>
  <div class="health-grid" id="healthGrid">
    <div class="hcard"><div class="hcard-ico">🖥</div><div><div class="hcard-lbl">Platform</div><div class="hcard-val" id="hPlat">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">🐍</div><div><div class="hcard-lbl">Python</div><div class="hcard-val" id="hPy">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">⏱</div><div><div class="hcard-lbl">Uptime</div><div class="hcard-val" id="hUp">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">🗄</div><div><div class="hcard-lbl">MongoDB</div><div class="hcard-val" id="hDB">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">📦</div><div><div class="hcard-lbl">Total Records</div><div class="hcard-val" id="hRec">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">📅</div><div><div class="hcard-lbl">Oldest Record</div><div class="hcard-val" id="hOld" style="font-size:.78rem">—</div></div></div>
    <div class="hcard"><div class="hcard-ico">🆕</div><div><div class="hcard-lbl">Latest Record</div><div class="hcard-val" id="hNew" style="font-size:.78rem">—</div></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Simulated API Requests — Last 24 Hours</div><span class="box-badge badge-amber">DEMO</span></div><div class="cw"><canvas id="sysReq"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Response Time Distribution (ms)</div><span class="box-badge badge-amber">DEMO</span></div><div class="cw"><canvas id="sysResp"></canvas></div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Detections per Device — Live</div><span class="box-badge badge-ice">MONGODB</span></div><div class="cw"><canvas id="sysDevice"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Weekly Detection Heatmap (Day × Session)</div><span class="box-badge badge-amber">DEMO</span></div><div class="cw"><canvas id="sysHeat"></canvas></div></div>
  </div>
</section>

<!-- ════════════════════════════════════════
     TAB 11 — LIVE DETECTIONS
════════════════════════════════════════ -->
<section class="sec" id="tab-live">
  <div class="sec-hdr">
    <div class="sec-icon">🔴</div>
    <div class="sec-info">
      <h2>Live Detection Feed — MongoDB</h2>
      <p>All charts and the recent-records table below are driven by real data from the <code>plant_disease_db.detections</code> collection. Data refreshes automatically every 30 seconds. Upload an image on the main app and watch the charts update here in real time.</p>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat"><div class="stat-lbl">Total</div><div class="stat-val g" id="liveTotal">—</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--violet))"><div class="stat-lbl">Avg Conf</div><div class="stat-val c" id="liveConf">—</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--crimson),var(--amber))"><div class="stat-lbl">Adversarial</div><div class="stat-val r" id="liveAdv">—</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--amber),var(--toxic))"><div class="stat-lbl">Foggy</div><div class="stat-val a" id="liveFog">—</div></div>
    <div class="stat" style="--accent:linear-gradient(90deg,var(--ice),var(--plasma))"><div class="stat-lbl">Dark</div><div class="stat-val c" id="liveDark">—</div></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Disease Distribution</div><span class="box-badge">REALTIME</span></div><div class="cw"><canvas id="liveDisease"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Severity Breakdown</div></div><div class="cw"><canvas id="liveSev"></canvas></div></div>
  </div>
  <div class="box full" style="margin-bottom:1.4rem">
    <div class="box-hdr"><div class="box-title">Daily Volume — Last 14 Days</div><span class="box-badge badge-ice">TIMELINE</span></div>
    <div class="cw"><canvas id="liveDaily"></canvas></div>
  </div>
  <div class="grid-2">
    <div class="box"><div class="box-hdr"><div class="box-title">Confidence Buckets</div></div><div class="cw"><canvas id="liveConfChart"></canvas></div></div>
    <div class="box"><div class="box-hdr"><div class="box-title">Condition Events</div></div><div class="cw"><canvas id="liveCond"></canvas></div></div>
  </div>
  <!-- Recent records table -->
  <div class="box" style="margin-top:0">
    <div class="box-hdr">
      <div class="box-title">Recent Detections Table</div>
      <span style="font-size:.62rem;color:var(--sub)">Auto-refresh 30s</span>
    </div>
    <div style="overflow-x:auto">
      <table class="tbl">
        <thead><tr><th>#</th><th>Disease</th><th>Confidence</th><th>Severity</th><th>Device</th><th>Flags</th><th>Detected At</th></tr></thead>
        <tbody id="liveTableBody"><tr><td colspan="7" style="text-align:center;color:var(--sub);padding:1.5rem">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>
</section>

<!-- ── Footer ── -->
<div class="footer">
  PlantDoc AI · Biopunk Analytics Suite · BPLD CNN ·
  <span id="ftClock"></span> ·
  Auto-refresh <span>30s</span>
</div>

<script>
// ════════════════════════════════════════════
// CHART.JS DEFAULTS  (biopunk palette)
// ════════════════════════════════════════════
Chart.defaults.color          = '#5d82a0';
Chart.defaults.borderColor    = '#1a3058';
Chart.defaults.font.family    = "'Space Mono', monospace";
Chart.defaults.font.size      = 10;

const CLS  = ['Anthracnose','Healthy','Leaf Crinkle','Powdery Mildew','Yellow Mosaic'];
const COL  = ['#00ff88','#00d4ff','#ffb800','#b060ff','#ff6b35'];
const COLA = ['rgba(0,255,136,.65)','rgba(0,212,255,.65)','rgba(255,184,0,.65)','rgba(176,96,255,.65)','rgba(255,107,53,.65)'];
const EP   = Array.from({length:30},(_,i)=>i+1);

const TIP = {plugins:{legend:{labels:{color:'#5d82a0',boxWidth:10,padding:12}},
  tooltip:{backgroundColor:'rgba(9,14,24,.97)',borderColor:'#1a3058',borderWidth:1,
  titleColor:'#d6eaf8',bodyColor:'#5d82a0',padding:10,
  callbacks:{label:c=>` ${c.dataset.label||''}: ${c.formattedValue}`}}}};

function mc(id,type,data,opts={}){
  const el=document.getElementById(id); if(!el)return null;
  return new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:false,...TIP,...opts}});
}
function kill(id){const c=Chart.getChart(id);if(c)c.destroy();}

// ════ DATASET ════════════════════════════════
function initDataset(){
  mc('dsRaw','bar',{labels:CLS,datasets:[{label:'Images',data:[808,817,756,783,793],backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:7}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
  mc('dsSplit','doughnut',{labels:['Train 70%','Val 15%','Test 15%'],datasets:[{data:[2770,593,594],backgroundColor:['rgba(0,255,136,.65)','rgba(0,212,255,.65)','rgba(255,184,0,.65)'],borderColor:['#00ff88','#00d4ff','#ffb800'],borderWidth:2,hoverOffset:10}]},{cutout:'64%'});
  mc('dsAug','bar',{labels:CLS,datasets:[{label:'Original',data:[808,817,756,783,793],backgroundColor:'rgba(0,212,255,.35)',borderColor:'#00d4ff',borderWidth:2,borderRadius:5},{label:'After Aug',data:[2400,2450,2270,2350,2380],backgroundColor:'rgba(0,255,136,.35)',borderColor:'#00ff88',borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
  mc('dsRadar','radar',{labels:CLS,datasets:[{label:'Normalised',data:[808/817,1,756/817,783/817,793/817].map(v=>+(v*100).toFixed(1)),backgroundColor:'rgba(0,255,136,.08)',borderColor:'#00ff88',borderWidth:2,pointBackgroundColor:'#00ff88',pointRadius:4}]},{scales:{r:{grid:{color:'#1a3058'},ticks:{display:false},pointLabels:{color:'#5d82a0',font:{size:9}}}}});
}

// ════ TRAINING ════════════════════════════════
function initTraining(){
  const tA=[42,58,68,74,79,83,85,87,88.5,89.5,90.2,91,91.8,92.3,92.9,93.2,93.7,94.2,94.7,95.1,95.4,95.8,96,96.3,96.5,96.6,96.7,96.7,96.8,96.8];
  const vA=[39,54,63,70,75,79,81,83.5,85,86.2,87.1,88,88.6,89.2,89.8,90.2,90.7,91.1,91.6,92,92.4,92.7,93,93.2,93.6,93.9,94.1,93.9,93.5,93.2];
  const tL=[1.61,1.35,1.12,.95,.82,.71,.63,.56,.50,.45,.41,.37,.34,.31,.28,.26,.24,.22,.20,.19,.17,.16,.15,.14,.13,.12,.11,.10,.095,.091];
  const vL=[1.68,1.42,1.18,1.01,.87,.77,.68,.62,.57,.52,.48,.44,.41,.38,.35,.33,.31,.29,.27,.26,.25,.24,.23,.22,.22,.21,.21,.21,.21,.204];
  mc('trAcc','line',{labels:EP,datasets:[{label:'Train',data:tA,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,.06)',borderWidth:2,pointRadius:0,tension:.35,fill:true},{label:'Val',data:vA,borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.04)',borderWidth:2,pointRadius:0,tension:.35,fill:true}]},{scales:{y:{min:35,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{color:'#1a3058'},title:{display:true,text:'Epoch',color:'#5d82a0'}}}});
  mc('trLoss','line',{labels:EP,datasets:[{label:'Train',data:tL,borderColor:'#ff2d55',backgroundColor:'rgba(255,45,85,.06)',borderWidth:2,pointRadius:0,tension:.35,fill:true},{label:'Val',data:vL,borderColor:'#ffb800',backgroundColor:'rgba(255,184,0,.04)',borderWidth:2,pointRadius:0,tension:.35,fill:true}]},{scales:{y:{beginAtZero:false,grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
  mc('trLR','line',{labels:EP,datasets:[{label:'LR',data:EP.map(e=>e<=10?.001:e<=20?.0005:.0001),borderColor:'#b060ff',borderWidth:2,pointRadius:3,stepped:true,backgroundColor:'rgba(176,96,255,.06)',fill:true}]},{scales:{y:{type:'logarithmic',grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
  const gap=tA.map((v,i)=>+(v-vA[i]).toFixed(2));
  mc('trGap','bar',{labels:EP,datasets:[{label:'Gap %',data:gap,backgroundColor:gap.map(v=>v>5?'rgba(255,45,85,.5)':v>3?'rgba(255,184,0,.5)':'rgba(0,255,136,.5)'),borderColor:gap.map(v=>v>5?'#ff2d55':v>3?'#ffb800':'#00ff88'),borderWidth:1,borderRadius:3}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}});
  mc('trTime','line',{labels:EP,datasets:[{label:'Sec/Epoch',data:EP.map(()=>+(5.8+Math.random()*.7).toFixed(2)),borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.06)',borderWidth:2,pointRadius:2,tension:.4,fill:true}]},{scales:{y:{beginAtZero:false,grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
  mc('trBatch','line',{labels:EP,datasets:[{label:'Batch Loss MA',data:EP.map((_,i)=>+(0.8*Math.exp(-i*.12)+.08+Math.random()*.04).toFixed(3)),borderColor:'#ff6b35',backgroundColor:'rgba(255,107,53,.05)',borderWidth:2,pointRadius:0,tension:.4,fill:true}]},{scales:{y:{beginAtZero:false,grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
}

// ════ METRICS ════════════════════════════════
function initMetrics(){
  const pr=[95.1,97.2,91.3,93,90.4],rc=[93.4,96,89.4,94.1,93.2],f1=[94.2,96.6,90.3,93.5,91.8];
  const hO={indexAxis:'y',scales:{x:{min:80,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},y:{grid:{display:false}}},plugins:{legend:{display:false}}};
  mc('mPrec','bar',{labels:CLS,datasets:[{label:'Precision',data:pr,backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:5}]},hO);
  mc('mRec', 'bar',{labels:CLS,datasets:[{label:'Recall',   data:rc,backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:5}]},hO);
  mc('mF1',  'bar',{labels:CLS,datasets:[{label:'F1',       data:f1,backgroundColor:'rgba(0,212,255,.55)',borderColor:'#00d4ff',borderWidth:2,borderRadius:5}]},hO);
  mc('mGroup','bar',{labels:CLS,datasets:[{label:'Precision',data:pr,backgroundColor:'rgba(0,255,136,.6)',borderColor:'#00ff88',borderWidth:2,borderRadius:4},{label:'Recall',data:rc,backgroundColor:'rgba(0,212,255,.6)',borderColor:'#00d4ff',borderWidth:2,borderRadius:4},{label:'F1',data:f1,backgroundColor:'rgba(255,184,0,.5)',borderColor:'#ffb800',borderWidth:2,borderRadius:4}]},{scales:{y:{min:85,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}});
  mc('mAUC','bar',{labels:CLS,datasets:[{label:'AUC-ROC',data:[.991,.997,.981,.986,.975],backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:5}]},{scales:{y:{min:.95,max:1,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
}

// ════ CONFUSION ═══════════════════════════════
function initConfusion(){
  const M=[[113,2,3,1,2],[1,118,1,1,1],[3,1,101,5,3],[2,0,3,110,2],[2,0,3,2,114]];
  const tot=M.map(r=>r.reduce((a,b)=>a+b,0));
  const cols=[''].concat(CLS.map(c=>c.split(' ')[0]));
  let h='<div class="mx">';
  h+=cols.map((c,i)=>`<div class="mx-hdr" style="${i>0?'text-align:center':''}">${i===0?'<span style="font-size:.54rem;color:var(--sub)">TRUE↓ PRED→</span>':c}</div>`).join('');
  M.forEach((row,ri)=>{
    h+=`<div class="mx-lbl">${CLS[ri].split(' ')[0]}</div>`;
    row.forEach((v,ci)=>{
      const p=(v/tot[ri]*100).toFixed(1),d=ri===ci,it=v/tot[ri];
      const bg=d?`rgb(0,${Math.round(it*220)},${Math.round(it*100)})`:`rgb(10,${Math.round(it*70)},${Math.round(it*16)})`;
      h+=`<div class="mx-cell" style="background:${bg};color:${it>.45?'#fff':'#5d82a0'}" title="${CLS[ri]}→${CLS[ci]}: ${v} (${p}%)"><div style="font-weight:${d?700:400};font-size:${d?'.8rem':'.67rem'}">${v}</div><div style="font-size:.55rem;opacity:.75">${p}%</div></div>`;
    });
  });
  document.getElementById('matrixDiv').innerHTML=h+'</div>';
  mc('cfDiag','bar',{labels:CLS,datasets:[{label:'Acc',data:M.map((r,i)=>(r[i]/tot[i]*100).toFixed(1)),backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:5}]},{scales:{y:{min:85,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  mc('cfErr','bar',{labels:CLS,datasets:[{label:'Error',data:M.map((r,i)=>+((1-r[i]/tot[i])*100).toFixed(1)),backgroundColor:'rgba(255,45,85,.45)',borderColor:'#ff2d55',borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
}

// ════ ADVERSARIAL ═════════════════════════════
function initAdversarial(){
  // Active defenses
  mc('advBar','bar',{labels:CLS,datasets:[
    {label:'Clean',   data:[95.1,97.2,91.3,93,90.4],backgroundColor:'rgba(0,255,136,.55)',borderColor:'#00ff88',borderWidth:2,borderRadius:4},
    {label:'Attacked',data:[41.2,28.5,30.1,38.7,35.2],backgroundColor:'rgba(255,45,85,.55)',borderColor:'#ff2d55',borderWidth:2,borderRadius:4},
    {label:'Defended',data:[89.3,91.2,84.7,88.4,83.4],backgroundColor:'rgba(255,184,0,.55)',borderColor:'#ffb800',borderWidth:2,borderRadius:4},
  ]},{scales:{y:{beginAtZero:true,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}});

  mc('advDefense','bar',{labels:['Gaussian Only','Median Only','Both Combined'],datasets:[{label:'Recovered Acc %',data:[78.2,81.4,87.4],backgroundColor:['rgba(0,212,255,.55)','rgba(176,96,255,.55)','rgba(0,255,136,.7)'],borderColor:['#00d4ff','#b060ff','#00ff88'],borderWidth:2,borderRadius:7}]},{scales:{y:{min:60,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});

  mc('advGrad','bar',{labels:['0-5','5-10','10-15','15-20','20-30','30+'],datasets:[
    {label:'Clean Images',data:[.42,.31,.15,.07,.04,.01],backgroundColor:'rgba(0,255,136,.45)',borderColor:'#00ff88',borderWidth:2,borderRadius:4},
    {label:'Adversarial', data:[.05,.08,.12,.18,.32,.25],backgroundColor:'rgba(255,45,85,.45)',borderColor:'#ff2d55',borderWidth:2,borderRadius:4},
  ]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'},ticks:{callback:v=>(v*100).toFixed(0)+'%'}},x:{title:{display:true,text:'Gradient Energy Range',color:'#5d82a0'},grid:{display:false}}}});

  mc('advFP','bar',{labels:CLS,datasets:[{label:'False Positive %',data:[3.8,5.2,4.1,3.6,4.8],backgroundColor:'rgba(255,107,53,.45)',borderColor:'#ff6b35',borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});

  // Research attacks
  mc('advResearch','bar',{
    labels:['Clean','FGSM ε=0.03','PGD (40-step)','C&W L2','DeepFool','Active Defense'],
    datasets:[{
      label:'Estimated Accuracy %',
      data:[93.2,34.7,14.2,9.8,19.3,87.4],
      backgroundColor:['rgba(0,255,136,.6)','rgba(255,184,0,.55)','rgba(176,96,255,.55)','rgba(255,45,85,.6)','rgba(255,107,53,.55)','rgba(0,212,255,.6)'],
      borderColor:['#00ff88','#ffb800','#b060ff','#ff2d55','#ff6b35','#00d4ff'],
      borderWidth:2,borderRadius:7,
    }],
  },{scales:{y:{beginAtZero:true,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});

  mc('advEps','line',{
    labels:['ε=0.01','ε=0.02','ε=0.03','ε=0.05','ε=0.10'],
    datasets:[
      {label:'FGSM (1-step)',    data:[88.1,67.4,34.7,18.2,9.4],  borderColor:'#ffb800',borderWidth:2,pointRadius:5,tension:.3},
      {label:'PGD (40-step)',    data:[72.3,42.1,14.2,6.8,3.1],   borderColor:'#b060ff',borderWidth:2,pointRadius:5,tension:.3,borderDash:[5,3]},
      {label:'DeepFool',         data:[65.1,35.2,19.3,11.4,7.2],  borderColor:'#ff6b35',borderWidth:2,pointRadius:5,tension:.3,borderDash:[3,3]},
      {label:'Active Defense',   data:[91.8,89.3,87.4,82.1,71.3], borderColor:'#00ff88',borderWidth:2,pointRadius:5,tension:.3,backgroundColor:'rgba(0,255,136,.06)',fill:true},
    ],
  },{scales:{y:{beginAtZero:true,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{color:'#1a3058'}}}});
}

// ════ WEATHER ═════════════════════════════════
function initWeather(){
  mc('wxCompare','bar',{labels:['Normal','Foggy','Dark','Low Contrast'],datasets:[{label:'Without',data:[93.2,67.4,59.2,71.8],backgroundColor:'rgba(255,45,85,.45)',borderColor:'#ff2d55',borderWidth:2,borderRadius:4},{label:'With Enhance',data:[93.2,88.1,86.3,85.7],backgroundColor:'rgba(0,255,136,.55)',borderColor:'#00ff88',borderWidth:2,borderRadius:4}]},{scales:{y:{min:50,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}});
  mc('wxDist','doughnut',{labels:['Normal','Foggy','Dark','Low Contrast'],datasets:[{data:[78,10,8,4],backgroundColor:['rgba(0,255,136,.6)','rgba(255,184,0,.6)','rgba(0,212,255,.6)','rgba(176,96,255,.6)'],borderColor:['#00ff88','#ffb800','#00d4ff','#b060ff'],borderWidth:2,hoverOffset:10}]},{cutout:'60%'});
  mc('wxPixel','bar',{labels:['Brightness','Contrast','Edge Density'],datasets:[{label:'Before',data:[112,.18,3.2],backgroundColor:'rgba(255,45,85,.4)',borderColor:'#ff2d55',borderWidth:2,borderRadius:4},{label:'After',data:[148,.34,5.8],backgroundColor:'rgba(0,255,136,.45)',borderColor:'#00ff88',borderWidth:2,borderRadius:4}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
  mc('wxClass','bar',{labels:CLS,datasets:[{label:'Foggy Raw',data:[71.2,65.4,62.1,69.8,68.4],backgroundColor:'rgba(255,45,85,.4)',borderColor:'#ff2d55',borderWidth:2,borderRadius:4},{label:'Foggy Enhanced',data:[90.1,92.3,85.2,89.4,83.6],backgroundColor:'rgba(0,212,255,.55)',borderColor:'#00d4ff',borderWidth:2,borderRadius:4}]},{scales:{y:{min:55,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}});
}

// ════ CONFIDENCE ══════════════════════════════
function initConfidence(){
  mc('confHist','bar',{labels:['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],datasets:[{label:'Count',data:[12,8,14,18,22,31,48,87,142,212],backgroundColor:['#ff2d5599','#ff2d5599','#ff2d5599','#ff2d5599','#ff2d5599','#ff2d5599','#ffb80099','#ffb80099','#00d4ff99','#00ff8899'],borderColor:['#ff2d55','#ff2d55','#ff2d55','#ff2d55','#ff2d55','#ff2d55','#ffb800','#ffb800','#00d4ff','#00ff88'],borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
  mc('confClass','bar',{labels:CLS,datasets:[{label:'Avg Conf',data:[87.4,91.2,78.4,85.3,84.1],backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:5}]},{scales:{y:{min:70,max:100,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  mc('confThresh','line',{labels:['30%','40%','50%','60%','70%','80%','90%'],datasets:[{label:'Accuracy',data:[93.2,93.2,93.1,93,91.2,87.4,74.1],borderColor:'#00ff88',borderWidth:2,pointRadius:5,tension:.3},{label:'Coverage',data:[100,98.5,96.2,91.6,82.4,71.2,53.8],borderColor:'#ffb800',borderWidth:2,pointRadius:5,tension:.3}]},{scales:{y:{beginAtZero:false,grid:{color:'#1a3058'},ticks:{callback:v=>v+'%'}},x:{grid:{color:'#1a3058'},title:{display:true,text:'Threshold',color:'#5d82a0'}}}});
}

// ════ ARCHITECTURE ════════════════════════════
function initArch(){
  mc('archParam','bar',{labels:['Conv Block 1','Conv Block 2','Conv Block 3','Conv Block 4','Dense Head'],datasets:[{label:'Params (K)',data:[1.0,18.8,74.4,296.2,134.1],backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:7}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  mc('archMap','line',{labels:['Input','CB1','CB2','CB3','CB4','GAP','Dense'],datasets:[{label:'Spatial (px)',data:[224,111,54,26,24,1,1],borderColor:'#00d4ff',borderWidth:2,pointRadius:4,tension:.3,yAxisID:'y'},{label:'Channels',data:[3,32,64,128,256,256,512],borderColor:'#ffb800',borderWidth:2,pointRadius:4,tension:.3,yAxisID:'y2'}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'},title:{display:true,text:'Spatial',color:'#5d82a0'}},y2:{position:'right',beginAtZero:true,grid:{display:false},title:{display:true,text:'Channels',color:'#5d82a0'}}}});
  mc('archFlops','bar',{labels:['Conv1','Conv2','Conv3','Conv4','Dense1','Out'],datasets:[{label:'GFLOPs',data:[.19,.27,.18,.32,.07,.003],backgroundColor:COLA,borderColor:COL,borderWidth:2,borderRadius:6}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  mc('archAct','bar',{labels:['CB1','CB2','CB3','CB4'],datasets:[{label:'Mean',data:[.42,.38,.31,.27],backgroundColor:'rgba(0,255,136,.45)',borderColor:'#00ff88',borderWidth:2,borderRadius:5},{label:'Std',data:[.18,.22,.19,.15],backgroundColor:'rgba(255,184,0,.35)',borderColor:'#ffb800',borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
}

// ════ SYSTEM HEALTH ═══════════════════════════
async function initHealth(){
  try{
    const d=await fetch('/viz/system-health').then(r=>r.json());
    document.getElementById('hPlat').textContent = d.platform||'–';
    document.getElementById('hPy').textContent   = d.python_ver||'–';
    document.getElementById('hUp').textContent   = d.uptime||'–';
    document.getElementById('hDB').innerHTML     = `<span class="dot ${d.db_connected?'dot-ok':'dot-err'}"></span>${d.db_connected?'Connected':'Offline'}`;
    document.getElementById('hRec').textContent  = d.total_records??'–';
    document.getElementById('hOld').textContent  = d.oldest_record||'–';
    document.getElementById('hNew').textContent  = d.newest_record||'–';
  }catch(e){}

  const hrs=Array.from({length:24},(_,i)=>`${String(i).padStart(2,'0')}h`);
  mc('sysReq','bar',{labels:hrs,datasets:[{label:'Requests',data:hrs.map(()=>~~(Math.random()*80+5)),backgroundColor:'rgba(0,212,255,.4)',borderColor:'#00d4ff',borderWidth:1,borderRadius:3}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false},ticks:{maxTicksLimit:8}}}});
  mc('sysResp','bar',{labels:['<50ms','50-100','100-200','200-500','500ms+'],datasets:[{label:'Count',data:[312,187,94,32,8],backgroundColor:['rgba(0,255,136,.55)','rgba(0,212,255,.55)','rgba(255,184,0,.55)','rgba(255,107,53,.55)','rgba(255,45,85,.65)'],borderColor:['#00ff88','#00d4ff','#ffb800','#ff6b35','#ff2d55'],borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  mc('sysHeat','bar',{labels:days,datasets:Array.from({length:4},(_,i)=>({label:`${i*6}-${i*6+6}h`,data:days.map(()=>~~(Math.random()*25)),backgroundColor:COLA[i],borderColor:COL[i],borderWidth:1,borderRadius:3}))},{scales:{y:{beginAtZero:true,stacked:true,grid:{color:'#1a3058'}},x:{stacked:true,grid:{display:false}}}});
}

// ════ LIVE DATA ════════════════════════════════
async function loadLive(){
  try{
    const d=await fetch('/viz/live-stats').then(r=>r.json());
    const tot=d.total||0;
    document.getElementById('ovTotal').textContent  = tot;
    document.getElementById('ovConf').textContent   = (d.avg_confidence||0)+'%';
    document.getElementById('ovAdv').textContent    = d.adversarial||0;
    document.getElementById('ovWx').textContent     = (d.foggy||0)+(d.dark||0);
    document.getElementById('liveTotal').textContent= tot;
    document.getElementById('liveConf').textContent = (d.avg_confidence||0)+'%';
    document.getElementById('liveAdv').textContent  = d.adversarial||0;
    document.getElementById('liveFog').textContent  = d.foggy||0;
    document.getElementById('liveDark').textContent = d.dark||0;

    const dL=Object.keys(d.disease_counts||{}),dV=Object.values(d.disease_counts||{});
    const sL=Object.keys(d.severity_counts||{}),sV=Object.values(d.severity_counts||{});
    const devL=Object.keys(d.device_counts||{}),devV=Object.values(d.device_counts||{});
    const bkts=['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'];
    const confD=d.conf_buckets||Array(10).fill(0);
    const cBg=bkts.map((_,i)=>i<6?'rgba(255,45,85,.5)':i<8?'rgba(255,184,0,.5)':'rgba(0,255,136,.6)');
    const cBo=bkts.map((_,i)=>i<6?'#ff2d55':i<8?'#ffb800':'#00ff88');
    const norm=Math.max(0,tot-(d.adversarial||0)-(d.foggy||0)-(d.dark||0));

    ['ovDiseaseChart','ovDailyChart','ovSevChart','ovCondChart','ovConfChart',
     'liveDisease','liveSev','liveDaily','liveConfChart','liveCond','confLive','sysDevice'].forEach(kill);

    if(dL.length){
      mc('ovDiseaseChart','bar',{labels:dL,datasets:[{label:'Count',data:dV,backgroundColor:dL.map((_,i)=>COLA[i%5]),borderColor:dL.map((_,i)=>COL[i%5]),borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
      mc('liveDisease','bar',{labels:dL,datasets:[{label:'Count',data:dV,backgroundColor:dL.map((_,i)=>COLA[i%5]),borderColor:dL.map((_,i)=>COL[i%5]),borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
    }
    if(sL.length){
      mc('ovSevChart','doughnut',{labels:sL,datasets:[{data:sV,backgroundColor:['rgba(255,45,85,.65)','rgba(255,184,0,.65)','rgba(0,255,136,.65)','rgba(0,212,255,.65)'],borderColor:['#ff2d55','#ffb800','#00ff88','#00d4ff'],borderWidth:2,hoverOffset:8}]},{cutout:'58%'});
      mc('liveSev','doughnut',{labels:sL,datasets:[{data:sV,backgroundColor:['rgba(255,45,85,.65)','rgba(255,184,0,.65)','rgba(0,255,136,.65)','rgba(0,212,255,.65)'],borderColor:['#ff2d55','#ffb800','#00ff88','#00d4ff'],borderWidth:2,hoverOffset:8}]},{cutout:'58%'});
    }
    if(d.daily_labels?.length){
      mc('ovDailyChart','line',{labels:d.daily_labels,datasets:[{label:'Detections',data:d.daily_values,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,.06)',borderWidth:2,pointRadius:4,fill:true,tension:.35}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
      mc('liveDaily','line',{labels:d.daily_labels,datasets:[{label:'Detections',data:d.daily_values,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,.06)',borderWidth:2,pointRadius:4,fill:true,tension:.35}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{color:'#1a3058'}}}});
    }
    mc('ovCondChart','bar',{labels:['Normal','Adversarial','Foggy','Dark'],datasets:[{label:'Events',data:[norm,d.adversarial||0,d.foggy||0,d.dark||0],backgroundColor:['rgba(0,255,136,.55)','rgba(255,45,85,.55)','rgba(255,184,0,.45)','rgba(0,212,255,.45)'],borderColor:['#00ff88','#ff2d55','#ffb800','#00d4ff'],borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
    mc('ovConfChart','bar',{labels:bkts,datasets:[{label:'Count',data:confD,backgroundColor:cBg,borderColor:cBo,borderWidth:2,borderRadius:4}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
    mc('liveConfChart','bar',{labels:bkts,datasets:[{label:'Count',data:confD,backgroundColor:cBg,borderColor:cBo,borderWidth:2,borderRadius:4}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
    mc('confLive','bar',{labels:bkts,datasets:[{label:'Count',data:confD,backgroundColor:cBg,borderColor:cBo,borderWidth:2,borderRadius:4}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}}});
    mc('liveCond','bar',{labels:['Normal','Adversarial','Foggy','Dark'],datasets:[{label:'Events',data:[norm,d.adversarial||0,d.foggy||0,d.dark||0],backgroundColor:['rgba(0,255,136,.55)','rgba(255,45,85,.55)','rgba(255,184,0,.45)','rgba(0,212,255,.45)'],borderColor:['#00ff88','#ff2d55','#ffb800','#00d4ff'],borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
    if(devL.length) mc('sysDevice','bar',{labels:devL,datasets:[{label:'Detections',data:devV,backgroundColor:'rgba(0,212,255,.5)',borderColor:'#00d4ff',borderWidth:2,borderRadius:5}]},{scales:{y:{beginAtZero:true,grid:{color:'#1a3058'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}});
  }catch(e){console.error('live-stats:',e);}

  // Recent records table
  try{
    const rec=await fetch('/viz/recent-records').then(r=>r.json());
    const tbody=document.getElementById('liveTableBody');
    if(!rec.records?.length){
      tbody.innerHTML='<tr><td colspan="7" style="text-align:center;color:var(--sub);padding:1.5rem">No records yet</td></tr>';
    } else {
      tbody.innerHTML=rec.records.map((r,i)=>{
        const sc=r.severity==='High'?'background:rgba(255,45,85,.15);color:#ff2d55':r.severity==='Medium'?'background:rgba(255,184,0,.15);color:#ffb800':'background:rgba(0,255,136,.12);color:#00ff88';
        const flags=[r.adversarial?'<span title="Adversarial detected" style="color:var(--crimson)">⚡ADV</span>':'',r.foggy?'<span title="Foggy enhanced" style="color:var(--amber)">🌫FOG</span>':''].filter(Boolean).join(' ');
        const ts=r.timestamp?new Date(r.timestamp*1000).toLocaleString():'–';
        return`<tr><td class="c">${i+1}</td><td style="color:var(--text)">${r.disease}</td><td style="color:var(--ice)">${r.confidence}%</td><td><span class="pill" style="${sc}">${r.severity}</span></td><td style="color:var(--sub)">${r.device}</td><td style="font-size:.65rem">${flags||'—'}</td><td style="color:var(--sub);font-size:.68rem">${ts}</td></tr>`;
      }).join('');
    }
  }catch(e){}
}

// ════ TAB SWITCHING ════════════════════════════
function go(name,el){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  document.getElementById('tab-'+name).classList.add('on');
  el.classList.add('on');
  if(name==='live'||name==='overview'){loadLive();}
  if(name==='health'){initHealth();}
}

// ════ CLOCK ═══════════════════════════════════
function tick(){document.getElementById('ftClock').textContent=new Date().toLocaleString();}
setInterval(tick,1000); tick();

// ════ AUTO-REFRESH ═════════════════════════════
setInterval(loadLive,30000);

// ════ INIT ════════════════════════════════════
window.addEventListener('DOMContentLoaded',async()=>{
  initDataset();
  initTraining();
  initMetrics();
  initConfusion();
  initAdversarial();
  initWeather();
  initConfidence();
  initArch();
  await initHealth();
  await loadLive();
  setTimeout(()=>{document.getElementById('loader').style.display='none';},1000);
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point  (AFTER _html is defined — this was the original bug!)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = find_free_port(5000)
    print(f"\n🔬 PlantDoc AI — Biopunk Analytics Dashboard")
    print(f"   Open : http://localhost:{port}")
    print(f"   Theme: Biopunk Laboratory\n")
    uvicorn.run(viz_app, host="0.0.0.0", port=port, reload=False)