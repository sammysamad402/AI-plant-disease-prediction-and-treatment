"""
shared_ui.py — Unified Design System for PlantDoc AI
Import this in main.py, consultation.py, diary.py, crop_calendar.py
"""

# ── NAV LINKS (used by every page) ────────────────────────────────────────────
NAV_ITEMS = [
    ("/",            "🏠",  "Home"),
    ("/predict",     "🔬",  "Detect"),
    ("/webcam",      "📹",  "Live Cam"),
    ("/records-page","📊",  "Records"),
    ("/consultation","💬",  "Consult"),
    ("/diary",       "📔",  "Diary"),
    ("/crop-calendar","🗓️","Calendar"),
    ("/agri-shops",  "🏪",  "Agri Shops"),
]

# ── SHARED CSS ─────────────────────────────────────────────────────────────────
SHARED_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --p0:#0D4A26; --p1:#1B6B3A; --p2:#2E8B57; --p3:#4CAF78; --p4:#A8D5B8;
  --ac:#F5A623; --ac2:#E08C10;
  --bg:#F0F7F2; --bg2:#E4F0E8;
  --card:#FFFFFF; --card2:#F8FCF9;
  --tx:#1A2E1A; --tx2:#3D5A3D; --tx3:#7A9A7A; --tx4:#A8C0A8;
  --bd:#C8DFC8; --bd2:#E0EEE0;
  --ok:#27AE60; --warn:#E67E22; --err:#E74C3C; --info:#2980B9;
  --sh:0 2px 8px rgba(27,107,58,.08);
  --shm:0 4px 20px rgba(27,107,58,.12);
  --shl:0 8px 40px rgba(27,107,58,.16);
  --r:12px; --rl:20px;
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--tx);
  min-height:100vh;line-height:1.6;font-size:15px;}

/* ── NAVBAR ── */
.pda-nav{background:#fff;border-bottom:1.5px solid var(--bd);position:sticky;top:0;z-index:999;
  box-shadow:var(--sh);}
.pda-nav-inner{max-width:1320px;margin:0 auto;display:flex;align-items:center;
  justify-content:space-between;padding:0 1.5rem;height:60px;gap:1rem;}
.pda-logo{display:flex;align-items:center;gap:.45rem;text-decoration:none;
  font-size:1.2rem;font-weight:800;color:var(--p1);white-space:nowrap;flex-shrink:0;}
.pda-logo span{font-size:1.5rem;}
.pda-links{display:flex;list-style:none;gap:.2rem;align-items:center;flex-wrap:wrap;}
.pda-links a{display:flex;align-items:center;gap:.3rem;text-decoration:none;
  color:var(--tx2);font-weight:600;font-size:.8rem;padding:.35rem .65rem;
  border-radius:8px;transition:all .18s;white-space:nowrap;}
.pda-links a:hover{background:var(--bg2);color:var(--p1);}
.pda-links a.active{background:var(--p1);color:#fff;}
.pda-links .nav-icon{font-size:.95rem;}

/* ── HERO ── */
.pda-hero{background:linear-gradient(135deg,var(--p0) 0%,var(--p1) 55%,var(--p2) 100%);
  color:#fff;padding:2.75rem 1.5rem 2.25rem;text-align:center;position:relative;overflow:hidden;}
.pda-hero::before{content:'';position:absolute;inset:0;
  background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");}
.pda-hero-inner{max-width:700px;margin:0 auto;position:relative;}
.pda-hero h1{font-size:2rem;font-weight:800;line-height:1.2;margin-bottom:.5rem;letter-spacing:-.5px;}
.pda-hero p{font-size:.98rem;opacity:.88;max-width:540px;margin:0 auto;}
.hero-badges{display:flex;gap:.5rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;}
.hero-badge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
  border-radius:20px;padding:.2rem .75rem;font-size:.75rem;font-weight:600;}

/* ── CONTAINER ── */
.pda-container{max-width:1320px;margin:0 auto;padding:2rem 1.5rem;}

/* ── CARD ── */
.pda-card{background:var(--card);border-radius:var(--r);box-shadow:var(--shm);
  border:1px solid var(--bd);}
.pda-card-body{padding:1.5rem;}
.pda-card-header{padding:1rem 1.5rem;border-bottom:1px solid var(--bd2);
  display:flex;align-items:center;gap:.5rem;font-weight:700;font-size:.92rem;color:var(--p1);}
.pda-card-footer{padding:.9rem 1.5rem;border-top:1px solid var(--bd2);background:var(--card2);
  border-radius:0 0 var(--r) var(--r);}

/* ── BUTTONS ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:.4rem;
  padding:.6rem 1.25rem;border:none;border-radius:8px;font-size:.875rem;font-weight:600;
  cursor:pointer;transition:all .18s;text-decoration:none;line-height:1;font-family:inherit;}
.btn:disabled{opacity:.45;cursor:not-allowed !important;transform:none !important;box-shadow:none !important;}
.btn-primary{background:var(--p1);color:#fff;}
.btn-primary:hover{background:var(--p0);transform:translateY(-1px);box-shadow:0 4px 12px rgba(27,107,58,.3);}
.btn-accent{background:var(--ac);color:#fff;}
.btn-accent:hover{background:var(--ac2);transform:translateY(-1px);}
.btn-outline{background:transparent;color:var(--p1);border:1.5px solid var(--p1);}
.btn-outline:hover{background:var(--p1);color:#fff;}
.btn-ghost{background:var(--bg2);color:var(--tx2);border:1px solid var(--bd);}
.btn-ghost:hover{background:var(--bd);color:var(--tx);}
.btn-danger{background:var(--err);color:#fff;}
.btn-danger:hover{background:#c0392b;}
.btn-lg{padding:.8rem 1.75rem;font-size:1rem;}
.btn-sm{padding:.35rem .8rem;font-size:.78rem;}
.btn-xs{padding:.25rem .6rem;font-size:.73rem;}
.btn-block{width:100%;}

/* ── FORM ── */
.form-group{margin-bottom:1rem;}
.form-label{display:block;font-size:.8rem;font-weight:700;color:var(--tx2);
  margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.03em;}
.form-control{width:100%;padding:.65rem .9rem;border:1.5px solid var(--bd);border-radius:8px;
  font-size:.88rem;color:var(--tx);background:#fff;transition:border .18s,box-shadow .18s;
  font-family:inherit;outline:none;}
.form-control:focus{border-color:var(--p1);box-shadow:0 0 0 3px rgba(27,107,58,.1);}
select.form-control{cursor:pointer;}

/* ── BADGES ── */
.badge{display:inline-flex;align-items:center;padding:.18rem .6rem;border-radius:20px;
  font-size:.72rem;font-weight:700;letter-spacing:.02em;}
.badge-green{background:var(--p1);color:#fff;}
.badge-ok{background:#D4EDDA;color:#155724;}
.badge-warn{background:#FFF3CD;color:#856404;}
.badge-err{background:#F8D7DA;color:#721C24;}
.badge-info{background:#D1ECF1;color:#0C5460;}
.badge-gray{background:var(--bg2);color:var(--tx2);}

/* ── ALERTS ── */
.alert{padding:.8rem 1rem;border-radius:8px;font-size:.85rem;display:flex;
  align-items:flex-start;gap:.5rem;margin-bottom:.75rem;}
.alert-ok{background:#EAF7EE;border-left:4px solid var(--ok);color:#1A5C32;}
.alert-warn{background:#FEF8E8;border-left:4px solid var(--warn);color:#7D4E00;}
.alert-err{background:#FDEDEC;border-left:4px solid var(--err);color:#7D1A1A;}
.alert-info{background:#EBF5FB;border-left:4px solid var(--info);color:#1A4A7D;}

/* ── PROGRESS ── */
.pbar-wrap{height:8px;background:var(--bd2);border-radius:10px;overflow:hidden;margin:.3rem 0;}
.pbar-fill{height:100%;border-radius:10px;background:linear-gradient(90deg,var(--p2),var(--p1));transition:width .5s ease;}

/* ── GRID UTILS ── */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;}
@media(max-width:900px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr;}}

/* ── SECTION HEADER ── */
.sec-head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:1.25rem;gap:1rem;flex-wrap:wrap;}
.sec-title{font-size:1.1rem;font-weight:800;color:var(--tx);}
.sec-sub{font-size:.83rem;color:var(--tx3);margin-top:.15rem;}

/* ── STAT BOXES ── */
.stat-card{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1.25rem;box-shadow:var(--sh);text-align:center;}
.stat-num{font-size:2rem;font-weight:800;color:var(--p1);line-height:1;}
.stat-lbl{font-size:.75rem;font-weight:600;color:var(--tx3);margin-top:.3rem;
  text-transform:uppercase;letter-spacing:.04em;}
.stat-icon{font-size:1.5rem;margin-bottom:.4rem;}

/* ── DIVIDER ── */
.divider{height:1px;background:var(--bd2);margin:1.25rem 0;}

/* ── TAG ── */
.tag{display:inline-block;padding:.18rem .55rem;background:var(--bg2);
  border:1px solid var(--bd);border-radius:6px;font-size:.73rem;
  color:var(--tx2);font-weight:600;}

/* ── EMPTY STATE ── */
.empty-state{text-align:center;padding:4rem 2rem;color:var(--tx3);}
.empty-state .es-icon{font-size:3.5rem;margin-bottom:1rem;}
.empty-state h3{font-size:1.2rem;font-weight:700;color:var(--tx2);margin-bottom:.5rem;}
.empty-state p{font-size:.88rem;}

/* ── LOADING ── */
.spinner{display:inline-block;width:18px;height:18px;border:2.5px solid var(--bd);
  border-top-color:var(--p1);border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── TOAST ── */
#pda-toast{position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
  background:var(--p0);color:#fff;padding:.75rem 1.25rem;border-radius:10px;
  font-size:.85rem;font-weight:600;box-shadow:var(--shl);
  transform:translateY(100px);opacity:0;transition:all .3s ease;pointer-events:none;}
#pda-toast.show{transform:none;opacity:1;}
/* ── UPLOAD BOX FIX ── */
.upload-box{
  display:flex;
  flex-direction:column;
  justify-content:center;   /* vertical center */
  align-items:center;       /* horizontal center */
  text-align:center;
  padding:2.5rem 1.5rem;
  min-height:220px;
  border:2px dashed var(--bd);
  border-radius:12px;
  background:var(--card2);
  gap:.6rem;
}
.upload-zone {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  min-height: 220px;
}
/* ✨ QUICK POLISH */
.upload-zone:hover {
  background: var(--bg2);
  border-color: var(--p2);
  transition: 0.3s;
}

.upload-icon {
  font-size: 3rem;
  transition: 0.3s;
}

.upload-zone:hover .upload-icon {
  transform: scale(1.1);
}

.upload-zone {
  cursor: pointer;
}

.upload-box img{
  width:60px;
  opacity:.85;
}

.upload-box p{
  font-weight:600;
  color:var(--tx2);
}

.upload-box small{
  color:var(--tx3);
}
/* ── RESPONSIVE NAVBAR ── */
@media(max-width:900px){
  .pda-nav-inner{height:auto;flex-wrap:wrap;padding:.75rem 1rem;}
  .pda-links a{font-size:.73rem;padding:.28rem .5rem;}
}
</style>
"""

# ── NAV BUILDER ───────────────────────────────────────────────────────────────
def build_nav(active: str = "") -> str:
    links_html = "\n".join(
        f'<li><a href="{href}" class="{"active" if active == href else ""}">'
        f'<span class="nav-icon">{icon}</span>{label}</a></li>'
        for href, icon, label in NAV_ITEMS
    )
    return f"""
<nav class="pda-nav">
  <div class="pda-nav-inner">
    <a href="/" class="pda-logo"><span>🌿</span>PlantDoc AI</a>

    <ul class="pda-links">{links_html}</ul>

    <div style="display:flex;align-items:center;gap:.6rem;">
      <span id="navUserName"
            style="font-size:.82rem;color:var(--tx2);font-weight:600;">
      </span>

      <button
        onclick="logout()"
        class="btn btn-sm btn-outline"
        style="padding:.35rem .7rem;">
        Logout
      </button>
    </div>

  </div>
</nav>"""
# ── SHARED HEAD ───────────────────────────────────────────────────────────────
def shared_head(title: str) -> str:
    return f"""<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — PlantDoc AI</title>
{SHARED_CSS}"""

# ── TOAST SCRIPT ─────────────────────────────────────────────────────────────
TOAST_SCRIPT = """
<div id="pda-toast"></div>

<script>

function getToken() {
  return localStorage.getItem('plantdoc_token');
}

function getUserName() {
  return localStorage.getItem('plantdoc_name') || 'Farmer';
}

function logout() {
  localStorage.removeItem('plantdoc_token');
  localStorage.removeItem('plantdoc_name');
  localStorage.removeItem('plantdoc_email');
  window.location.href = '/login';
}

function requireAuth() {
  if (
      !window.location.pathname.startsWith('/login')
      &&
      !window.location.pathname.startsWith('/register')
  ) {
      if (!getToken()) {
          window.location.href = '/login';
      }
  }
}

async function authFetch(url, options = {}) {
  const token = getToken();

  if (!token) {
      window.location.href = '/login';
      return;
  }

  options.headers = options.headers || {};
  options.headers['Authorization'] = 'Bearer ' + token;

  const res = await fetch(url, options);

  if (res.status === 401) {
      logout();
      return;
  }

  return res;
}

function showNavUser() {
  const el = document.getElementById('navUserName');
  if (el) {
      el.textContent = getUserName();
  }
}

function showToast(msg, dur=3000){
  const t=document.getElementById('pda-toast');
  t.textContent=msg;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),dur);
}

document.addEventListener('DOMContentLoaded', () => {
  requireAuth();
  showNavUser();
});

</script>
"""