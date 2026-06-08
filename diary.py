"""
diary.py — Farm Diary Module with Unified PlantDoc AI Design
"""

from fastapi import APIRouter, Request, Depends
from auth import get_current_user
from fastapi.responses import JSONResponse, HTMLResponse
from pymongo import MongoClient
from bson import ObjectId
import os, logging
from datetime import datetime

logger = logging.getLogger(__name__)
diary_router = APIRouter()

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_client = MongoClient(MONGO_URI)
_db = _client.get_database("plant_disease_db")
diary_col = _db.get_collection("diary")


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@diary_router.post("/diary/save")
async def save_diary_entry(request: Request,user=Depends(get_current_user)):
    try:
        body = await request.json()
        doc = {
            "user_id": user["user_id"],
            "type":      body.get("type", "chat"),
            "title":     body.get("title", "Untitled Entry"),
            "content":   body.get("content", {}),
            "tags":      body.get("tags", []),
            "crop":      body.get("crop", ""),
            "language":  body.get("language", "english"),
            "timestamp": int(datetime.now().timestamp()),
            "date_str":  datetime.now().strftime("%d %B %Y, %I:%M %p"),
        }
        result = diary_col.insert_one(doc)
        return JSONResponse({"status": "success", "id": str(result.inserted_id)})
    except Exception as e:
        logger.error(f"Diary save error: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@diary_router.get("/diary/entries")
async def get_diary_entries(limit: int = 100, entry_type: str = "",user=Depends(get_current_user)):
    try:
        query = {"user_id": user["user_id"]}
        if entry_type in ("chat", "detection"):
            query["type"] = entry_type
        docs = list(diary_col.find(query).sort("timestamp", -1).limit(limit))
        for d in docs:
            d["_id"] = str(d["_id"])
        return JSONResponse({"status": "success", "entries": docs})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@diary_router.delete("/diary/entries/{entry_id}")
async def delete_diary_entry(entry_id: str,user=Depends(get_current_user)):
    try:
        if not ObjectId.is_valid(entry_id):
            return JSONResponse({"status": "error", "detail": "Invalid ID"}, status_code=400)
        result = diary_col.delete_one({"_id": ObjectId(entry_id),"user_id": user["user_id"]})
        if result.deleted_count:
            return JSONResponse({"status": "success"})
        return JSONResponse({"status": "error", "detail": "Entry not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@diary_router.get("/diary/stats")
async def get_diary_stats(user=Depends(get_current_user)):
    try:
        uid = user["user_id"]
        total = diary_col.count_documents({"user_id": uid})
        chats = diary_col.count_documents({"user_id": uid,"type": "chat"})
        detections = diary_col.count_documents({"user_id": uid,"type": "detection"})
        pipeline = [
            {"$match": {"user_id": uid,"type": "detection"}},
            {"$group": {"_id": "$content.disease_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}, {"$limit": 1}
        ]
        top = list(diary_col.aggregate(pipeline))
        top_disease = top[0]["_id"] if top else "—"
        return JSONResponse({
            "status": "success", "total": total, "chats": chats,
            "detections": detections, "top_disease": top_disease
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@diary_router.get("/diary")
async def diary_page():
    return HTMLResponse(content=_diary_html())


@diary_router.get("/agri-shops")
async def agri_shops_page():
    return HTMLResponse(content=_shops_html())


# ── DIARY HTML ────────────────────────────────────────────────────────────────

def _diary_html() -> str:
    from shared_ui import shared_head, build_nav, TOAST_SCRIPT
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Farm Diary")}<style>
.diary-layout{{display:grid;grid-template-columns:220px 1fr;gap:1.5rem;margin-top:1.5rem;}}
@media(max-width:800px){{.diary-layout{{grid-template-columns:1fr;}}}}

.filter-panel{{display:flex;flex-direction:column;gap:1rem;}}
.filter-btn{{display:flex;align-items:center;gap:.5rem;width:100%;text-align:left;
  background:transparent;border:1.5px solid var(--bd);padding:.55rem .75rem;border-radius:8px;
  font-size:.83rem;color:var(--tx2);cursor:pointer;transition:all .18s;font-family:inherit;font-weight:600;}}
.filter-btn:hover{{background:var(--bg2);border-color:var(--p3);color:var(--p1);}}
.filter-btn.active{{background:var(--p1);color:#fff;border-color:var(--p1);}}
.filter-btn .cnt{{margin-left:auto;background:rgba(255,255,255,.25);border-radius:10px;
  padding:.05rem .4rem;font-size:.72rem;}}
.filter-btn:not(.active) .cnt{{background:var(--bg2);color:var(--tx3);}}

/* Timeline */
.timeline{{position:relative;}}
.timeline::before{{content:'';position:absolute;left:20px;top:0;bottom:0;width:2px;
  background:linear-gradient(to bottom,var(--ac),var(--p2));border-radius:2px;}}

.t-entry{{display:flex;gap:1rem;margin-bottom:1.25rem;animation:slideIn .3s ease both;}}
@keyframes slideIn{{from{{opacity:0;transform:translateX(-8px)}}to{{opacity:1;transform:none}}}}

.t-dot{{width:40px;height:40px;border-radius:50%;flex-shrink:0;display:flex;
  align-items:center;justify-content:center;font-size:1.1rem;z-index:1;
  border:3px solid #fff;box-shadow:var(--sh);}}
.t-dot.chat{{background:linear-gradient(135deg,var(--p1),var(--p2));}}
.t-dot.detection{{background:linear-gradient(135deg,var(--ac),var(--ac2));}}

.t-card{{flex:1;background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1rem 1.1rem;box-shadow:var(--sh);transition:box-shadow .18s;}}
.t-card:hover{{box-shadow:var(--shm);}}

.t-card-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:.5rem;margin-bottom:.6rem;}}
.t-title{{font-size:.92rem;font-weight:700;color:var(--tx);}}
.t-meta{{font-size:.72rem;color:var(--tx3);white-space:nowrap;flex-shrink:0;}}

.t-preview{{border-top:1px solid var(--bd2);padding-top:.6rem;margin-top:.5rem;}}
.preview-msg{{display:flex;gap:.4rem;font-size:.82rem;margin-bottom:.3rem;}}
.pm-role{{font-weight:700;font-size:.73rem;flex-shrink:0;padding-top:.05rem;}}
.pm-role.user{{color:var(--p1);}}
.pm-role.assistant{{color:var(--ac2);}}
.pm-text{{color:var(--tx2);overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}}

.t-tags{{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.6rem;}}

.t-footer{{display:flex;align-items:center;justify-content:space-between;
  margin-top:.75rem;padding-top:.65rem;border-top:1px solid var(--bd2);}}
.t-msg-count{{font-size:.73rem;color:var(--tx3);}}

/* Detail modal */
.modal-overlay{{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;
  display:none;align-items:center;justify-content:center;padding:1rem;}}
.modal-overlay.open{{display:flex;}}
.modal-box{{background:var(--card);border-radius:var(--rl);max-width:680px;width:100%;
  max-height:85vh;overflow-y:auto;box-shadow:var(--shl);}}
.modal-header{{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--bd2);
  padding:1rem 1.25rem;display:flex;align-items:center;justify-content:space-between;
  border-radius:var(--rl) var(--rl) 0 0;z-index:1;}}
.modal-title{{font-size:1rem;font-weight:700;}}
.modal-body{{padding:1.25rem;}}
.modal-msg{{display:flex;gap:.5rem;margin-bottom:.75rem;}}
.modal-msg .role-label{{width:75px;font-size:.75rem;font-weight:700;flex-shrink:0;
  padding:.2rem 0;text-align:right;}}
.modal-msg.user .role-label{{color:var(--p1);}}
.modal-msg.bot .role-label{{color:var(--ac2);}}
.modal-bubble{{background:var(--bg2);border-radius:8px;padding:.6rem .8rem;
  font-size:.84rem;color:var(--tx);line-height:1.55;flex:1;}}
.modal-msg.user .modal-bubble{{background:#E8F5EE;border:1px solid var(--p4);}}

.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}}
@media(max-width:700px){{.stats-row{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
{build_nav("/diary")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>📔 Farm Diary</h1>
    <p>Your complete record of plant disease detections and expert consultations</p>
    <div class="hero-badges">
      <span class="hero-badge">🔍 Filterable</span>
      <span class="hero-badge">📊 Stats</span>
      <span class="hero-badge">📤 Exportable</span>
    </div>
  </div>
</div>

<div class="pda-container">

  <!-- Stats -->
  <div class="stats-row" id="statsRow">
    <div class="stat-card"><div class="stat-icon">📋</div><div class="stat-num" id="sTotal">—</div><div class="stat-lbl">Total Entries</div></div>
    <div class="stat-card"><div class="stat-icon">💬</div><div class="stat-num" id="sChats">—</div><div class="stat-lbl">Consultations</div></div>
    <div class="stat-card"><div class="stat-icon">🔬</div><div class="stat-num" id="sDet">—</div><div class="stat-lbl">Detections</div></div>
    <div class="stat-card"><div class="stat-icon">🦠</div><div class="stat-num" id="sTop">—</div><div class="stat-lbl">Top Disease</div></div>
  </div>

  <div class="diary-layout">

    <!-- Filter panel -->
    <div class="filter-panel">
      <div class="pda-card">
        <div class="pda-card-header">🗂️ Filter By</div>
        <div class="pda-card-body" style="display:flex;flex-direction:column;gap:.4rem;">
          <button class="filter-btn active" onclick="setFilter('all', this)">📋 All Entries <span class="cnt" id="cntAll">0</span></button>
          <button class="filter-btn" onclick="setFilter('chat', this)">💬 Consultations <span class="cnt" id="cntChat">0</span></button>
          <button class="filter-btn" onclick="setFilter('detection', this)">🔬 Detections <span class="cnt" id="cntDet">0</span></button>
        </div>
      </div>

      <div class="pda-card">
        <div class="pda-card-header">🔍 Search</div>
        <div class="pda-card-body">
          <input type="text" id="searchInput" class="form-control" placeholder="Search entries..."
            oninput="applyFilters()">
        </div>
      </div>

      <div class="pda-card">
        <div class="pda-card-header">⚙️ Actions</div>
        <div class="pda-card-body" style="display:flex;flex-direction:column;gap:.5rem;">
          <a href="/predict" class="btn btn-primary btn-block btn-sm">🔬 New Detection</a>
          <a href="/consultation" class="btn btn-outline btn-block btn-sm">💬 Ask Expert</a>
        </div>
      </div>
    </div>

    <!-- Timeline -->
    <div>
      <div id="timelineContainer">
        <div class="empty-state">
          <div class="es-icon"><span class="spinner" style="width:40px;height:40px;border-width:4px;"></span></div>
          <h3>Loading diary...</h3>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="detailModal">
  <div class="modal-box">
    <div class="modal-header">
      <div class="modal-title" id="modalTitle">Chat History</div>
      <button class="btn btn-ghost btn-sm" onclick="closeModal()">✕ Close</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
let allEntries = [], currentFilter = 'all', entryMap = {{}};

async function loadDiary() {{
  try {{
    const [entriesRes, statsRes] = await Promise.all([
      authFetch('/diary/entries?limit=100'), authFetch('/diary/stats')
    ]);
    const entries = await entriesRes.json();
    const stats = await statsRes.json();

    if(entries.status === 'success') {{
      allEntries = entries.entries;
      entries.entries.forEach(e => entryMap[e._id] = e);
      updateCounts(allEntries);
      applyFilters();
    }}
    if(stats.status === 'success') {{
      document.getElementById('sTotal').textContent = stats.total;
      document.getElementById('sChats').textContent = stats.chats;
      document.getElementById('sDet').textContent = stats.detections;
      document.getElementById('sTop').textContent = (stats.top_disease || '—').replace('_',' ');
    }}
  }} catch(e) {{
    document.getElementById('timelineContainer').innerHTML =
      '<div class="alert alert-err" style="margin-top:2rem;">❌ Failed to load diary.</div>';
  }}
}}

function updateCounts(entries) {{
  document.getElementById('cntAll').textContent = entries.length;
  document.getElementById('cntChat').textContent = entries.filter(e=>e.type==='chat').length;
  document.getElementById('cntDet').textContent = entries.filter(e=>e.type==='detection').length;
}}

function setFilter(f, btn) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  let filtered = allEntries;
  if(currentFilter !== 'all') filtered = filtered.filter(e => e.type === currentFilter);
  if(q) filtered = filtered.filter(e =>
    (e.title||'').toLowerCase().includes(q) ||
    (e.tags||[]).join(' ').toLowerCase().includes(q)
  );
  renderTimeline(filtered);
}}

function renderTimeline(entries) {{
  const container = document.getElementById('timelineContainer');
  if(!entries.length) {{
    container.innerHTML = `<div class="empty-state">
      <div class="es-icon">📭</div>
      <h3>No Entries Found</h3>
      <p>Try changing filters or <a href="/predict" style="color:var(--p1);">detect a disease</a> to create your first entry.</p>
    </div>`;
    return;
  }}

  container.innerHTML = `<div class="timeline">${{entries.map(entry => {{
    const isChat = entry.type === 'chat';
    const icon = isChat ? '💬' : '🔬';
    const dotClass = isChat ? 'chat' : 'detection';
    const typeLabel = isChat ? 'Consultation' : 'Detection';
    const typeBadge = isChat ? 'badge-ok' : 'badge-warn';
    const tags = (entry.tags || []).map(t => `<span class="tag">${{t}}</span>`).join('');

    let preview = '';
    if(isChat && entry.content?.messages?.length) {{
      const msgs = entry.content.messages.slice(0, 2);
      preview = `<div class="t-preview">${{msgs.map(m =>
        `<div class="preview-msg">
          <span class="pm-role ${{m.role === 'user' ? 'user' : 'assistant'}}">${{m.role === 'user' ? 'You' : 'AgriDoc'}}</span>
          <span class="pm-text">${{(typeof m.content === 'string' ? m.content : '').substring(0,100)}}</span>
        </div>`
      ).join('')}}</div>`;
    }} else if(!isChat && entry.content) {{
      const c = entry.content;
      preview = `<div class="t-preview" style="font-size:.82rem;color:var(--tx2);">
        <strong>${{c.disease_name||'Unknown'}}</strong>
        ${{c.confidence ? ` — ${{(c.confidence*100).toFixed(1)}}% confidence` : ''}}
        ${{c.severity ? ` · ${{c.severity}} severity` : ''}}
      </div>`;
    }}

    const msgCount = isChat ? (entry.content?.messages?.length || 0) : 0;

    return `<div class="t-entry">
      <div class="t-dot ${{dotClass}}">${{icon}}</div>
      <div class="t-card">
        <div class="t-card-top">
          <div>
            <div class="t-title">${{entry.title || 'Untitled Entry'}}</div>
            <div style="margin-top:.2rem;">
              <span class="badge ${{typeBadge}}">${{typeLabel}}</span>
              ${{entry.language && entry.language !== 'english' ? `<span class="badge badge-gray" style="margin-left:.3rem;">${{entry.language}}</span>` : ''}}
            </div>
          </div>
          <div class="t-meta">${{entry.date_str || ''}}</div>
        </div>
        ${{preview}}
        ${{tags ? `<div class="t-tags">${{tags}}</div>` : ''}}
        <div class="t-footer">
          <span class="t-msg-count">${{isChat && msgCount ? `${{msgCount}} messages` : ''}}</span>
          <div style="display:flex;gap:.4rem;">
            ${{isChat ? `<button class="btn btn-ghost btn-xs" onclick="openModal('${{entry._id}}')">👁 View Chat</button>` : ''}}
            <button class="btn btn-danger btn-xs" onclick="deleteEntry('${{entry._id}}')">🗑️</button>
          </div>
        </div>
      </div>
    </div>`;
  }}).join('')}}</div>`;
}}
function formatDiaryText(text) {{
  return text
    .replace(/###\\s?(.*)/g, '<br><b>$1</b><br>')  // ✅ fix ###
    .replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>')
    .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
    .replace(/\\n- /g, '<br>• ')
    .replace(/\\n/g, '<br>');
}}

function openModal(id) {{
  const entry = entryMap[id];
  if(!entry || !entry.content?.messages) return;
  document.getElementById('modalTitle').textContent = entry.title || 'Chat History';
  document.getElementById('modalBody').innerHTML = entry.content.messages.map(m => {{
    const role = m.role === 'user' ? 'user' : 'bot';
    const label = m.role === 'user' ? 'You' : 'AgriDoc';
    const text = typeof m.content === 'string' ? m.content : '';
    return `<div class="modal-msg ${{role}}">
      <div class="role-label">${{label}}</div>
      <div class="modal-bubble">${{formatDiaryText(text)}}</div>
    </div>`;
  }}).join('');
  document.getElementById('detailModal').classList.add('open');
}}

function closeModal() {{
  document.getElementById('detailModal').classList.remove('open');
}}
document.getElementById('detailModal').addEventListener('click', e => {{
  if(e.target === document.getElementById('detailModal')) closeModal();
}});

async function deleteEntry(id) {{
  if(!confirm('Delete this diary entry permanently?')) return;
  try {{
    const res = await authFetch(`/diary/entries/${{id}}`, {{method:'DELETE'}});
    const d = await res.json();
    if(d.status === 'success') {{ showToast('✅ Entry deleted'); loadDiary(); }}
    else showToast('⚠️ ' + d.detail);
  }} catch(e) {{ showToast('❌ Network error'); }}
}}

loadDiary();
</script>
</body>
</html>"""


# ── AGRI SHOPS HTML ───────────────────────────────────────────────────────────

def _shops_html() -> str:
    from shared_ui import shared_head, build_nav, TOAST_SCRIPT
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Agri Shops")}<style>
.shops-intro{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1.5rem;margin-bottom:1.5rem;box-shadow:var(--sh);text-align:center;}}
.shops-intro p{{font-size:.9rem;color:var(--tx2);margin-bottom:1rem;}}
.map-frame{{width:100%;height:520px;border-radius:var(--r);border:2px solid var(--bd);
  box-shadow:var(--shm);}}
.quick-links{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin-top:1.5rem;}}
.quick-link-card{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1.25rem;box-shadow:var(--sh);text-align:center;transition:all .2s;}}
.quick-link-card:hover{{transform:translateY(-3px);box-shadow:var(--shm);}}
.qlc-icon{{font-size:2rem;margin-bottom:.6rem;}}
.qlc-title{{font-size:.9rem;font-weight:700;color:var(--tx);margin-bottom:.3rem;}}
.qlc-desc{{font-size:.78rem;color:var(--tx2);}}
/* 🔍 Search Panel Styling */
.search-panel{{
  background: white;
  padding: 20px;
  border-radius: 14px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  margin: 20px;
}}
.form-row {{
  margin-bottom: 15px;
}}

label {{
  display: block;
  font-weight: 600;
  margin-bottom: 5px;
}}

select {{
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ddd;
}}

button {{
  background: #2d6a4f;
  color: white;
  border: none;
  padding: 12px;
  border-radius: 10px;
  width: 100%;
  cursor: pointer;
  font-weight: bold;
}}

button:hover {{
  background: #1b4332;
}}
</style>
</head>
<body>
{build_nav("/agri-shops")}


<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>🏪 Nearby Agri Shops</h1>
    <p>Find agricultural supply stores, pesticide dealers, and farming equipment near you</p>
    <div class="hero-badges">
      <span class="hero-badge">📍 Live Location</span>
      <span class="hero-badge">🗺️ Google Maps</span>
      <span class="hero-badge">📞 Direct Contact</span>
    </div>
  </div>
</div>
<div class="search-panel">
  <h2>🔍 Find Nearby Agri Shops</h2>

  <div class="form-row">
    <label>📍 Search Radius (km)</label>
    <select id="radius">
      <option value="2">2 km</option>
      <option value="5" selected>5 km</option>
      <option value="10">10 km</option>
      <option value="20">20 km</option>
    </select>
  </div>

  <div class="form-row">
    <label>🧾 What are you looking for?</label>
    <select id="category">
      <option value="agriculture shop">🌾 General Agri Shops</option>
      <option value="pesticide shop">🧪 Pesticides</option>
      <option value="fertilizer shop">🌱 Fertilizers</option>
      <option value="seed store">🌿 Seeds</option>
      <option value="farm tools shop">🛠 Tools</option>
      <option value="plant medicine shop">💊 Medicines</option>
    </select>
  </div>

  <button onclick="searchShops()">🔎 Search Now</button>
</div>

<div id="mapContainer" style="padding:20px;">
  <iframe id="mapFrame"
    width="100%"
    height="450"
    style="border:0;border-radius:12px;"
    loading="lazy">
  </iframe>
</div>
<script>
    function searchShops() {{
        const radius = document.getElementById("radius").value;
        const category = document.getElementById("category").value;

        const query = category + " within " + radius + " km near me";

        const url = "https://www.google.com/maps?q=" + encodeURIComponent(query) + "&output=embed";

    document.getElementById("mapFrame").src = url;
  }}

// Load default automatically
window.onload = () => {{
    searchShops();
}};
</script>


<div class="pda-container">
  <div class="shops-intro">
    <p>The map below shows agricultural supply stores, pesticide shops, and farming centers near your current location. Click any marker for details, directions, and contact information.</p>
    <button class="btn btn-primary" onclick="loadNearbyShops()">📍 Find Shops Near Me</button>
    <button class="btn btn-ghost" style="margin-left:.5rem;" onclick="searchCustom()">🔍 Search Location</button>
  </div>

  <div class="quick-links">
    <div class="quick-link-card">
      <div class="qlc-icon">🌾</div>
      <div class="qlc-title">Kisan Helpline</div>
      <p class="qlc-desc">Government helpline for farmers — free advice and support</p>
      <a href="tel:1800-180-1551" class="btn btn-primary btn-sm" style="margin-top:.75rem;">📞 1800-180-1551</a>
    </div>
    <div class="quick-link-card">
      <div class="qlc-icon">🏛️</div>
      <div class="qlc-title">KVK Centers</div>
      <p class="qlc-desc">Krishi Vigyan Kendra — local agricultural extension centers</p>
      <a href="https://icar.org.in/en/krishi-vigyan-kendras-kvks" target="_blank" class="btn btn-outline btn-sm" style="margin-top:.75rem;">🔗 Find KVK</a>
    </div>
    <div class="quick-link-card">
      <div class="qlc-icon">🧪</div>
      <div class="qlc-title">Soil Testing Labs</div>
      <p class="qlc-desc">Get your soil tested for nutrient deficiencies and pH levels</p>
      <a href="https://www.google.com/maps/search/soil+testing+laboratory+near+me" target="_blank" class="btn btn-outline btn-sm" style="margin-top:.75rem;">📍 Find Labs</a>
    </div>
    <div class="quick-link-card">
      <div class="qlc-icon">💊</div>
      <div class="qlc-title">Pesticide Dealers</div>
      <p class="qlc-desc">Certified pesticide and fungicide dealers near you</p>
      <a href="https://www.google.com/maps/search/pesticide+dealer+near+me" target="_blank" class="btn btn-outline btn-sm" style="margin-top:.75rem;">📍 Find Dealers</a>
    </div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
function loadNearbyShops() {{
  if(!navigator.geolocation) {{
    showToast('⚠️ Geolocation not supported in your browser');
    return;
  }}
  showToast('📍 Getting your location...');
  navigator.geolocation.getCurrentPosition(pos => {{
    const lat = pos.coords.latitude, lng = pos.coords.longitude;
    const query = "agriculture shop near me";

    const src = "https://www.google.com/maps?q=" 
            + encodeURIComponent(query) 
            + "&output=embed";
    document.getElementById('mapFrame').src = src;
    showToast('✅ Map updated to your location!');
  }}, err => {{
    showToast('❌ Could not get location: ' + err.message);
  }});
}}

function searchCustom() {{
  const loc = prompt('Enter city or area name:');
  if(loc) {{
    const query = "agriculture shop " + loc;

const url = "https://www.google.com/maps?q=" 
            + encodeURIComponent(query) 
            + "&output=embed";

document.getElementById('mapFrame').src = url;
  }}
}}
</script>
</body>
</html>"""