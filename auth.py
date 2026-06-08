"""
auth.py — JWT Authentication for PlantDoc AI
Provides user registration, login, and token verification.
All user data (uploads, detections, diary, consultations) is scoped by user_id.
"""

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os, logging, re

logger = logging.getLogger(__name__)
auth_router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

# ── DB ────────────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_client = MongoClient(MONGO_URI)
_db = _client.get_database("plant_disease_db")
users_col = _db.get_collection("users")
users_col.create_index("email", unique=True)

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    """Returns payload dict or raises HTTPException."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ── FastAPI dependency ────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """
    Dependency — use in any protected route:
        @app.get("/protected")
        async def route(user = Depends(get_current_user)):
            ...
    Returns {"user_id": str, "email": str}
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    return {"user_id": payload["sub"], "email": payload["email"]}

def optional_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict | None:
    """Like get_current_user but returns None instead of raising if no token."""
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        return {"user_id": payload["sub"], "email": payload["email"]}
    except HTTPException:
        return None

# ── Validation helpers ────────────────────────────────────────────────────────
def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@]+@[^@]+\.[^@]+$", email))

def _valid_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isdigit() for c in pw):
        return False, "Password must contain at least one number"
    return True, ""

# ── API endpoints ─────────────────────────────────────────────────────────────
@auth_router.post("/auth/register")
async def register(request: Request):
    try:
        body = await request.json()
        name     = body.get("name", "").strip()
        email    = body.get("email", "").strip().lower()
        password = body.get("password", "")

        if not name or not email or not password:
            return JSONResponse({"status": "error", "detail": "Name, email, and password are required"}, status_code=400)
        if not _valid_email(email):
            return JSONResponse({"status": "error", "detail": "Invalid email address"}, status_code=400)
        ok, msg = _valid_password(password)
        if not ok:
            return JSONResponse({"status": "error", "detail": msg}, status_code=400)

        if users_col.find_one({"email": email}):
            return JSONResponse({"status": "error", "detail": "An account with this email already exists"}, status_code=409)

        from bson import ObjectId
        user_id = str(ObjectId())
        users_col.insert_one({
            "_id": ObjectId(user_id),
            "name": name,
            "email": email,
            "password_hash": hash_password(password),
            "created_at": int(datetime.utcnow().timestamp()),
        })

        token = create_access_token(user_id, email)
        logger.info(f"New user registered: {email}")
        return JSONResponse({"status": "success", "token": token, "name": name, "email": email})

    except Exception as e:
        logger.error(f"Register error: {e}", exc_info=True)
        return JSONResponse({"status": "error", "detail": "Registration failed"}, status_code=500)


@auth_router.post("/auth/login")
async def login(request: Request):
    try:
        body = await request.json()
        email    = body.get("email", "").strip().lower()
        password = body.get("password", "")

        user = users_col.find_one({"email": email})
        if not user or not verify_password(password, user["password_hash"]):
            return JSONResponse({"status": "error", "detail": "Invalid email or password"}, status_code=401)

        token = create_access_token(str(user["_id"]), email)
        logger.info(f"User logged in: {email}")
        return JSONResponse({"status": "success", "token": token, "name": user.get("name", ""), "email": email})

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return JSONResponse({"status": "error", "detail": "Login failed"}, status_code=500)


@auth_router.get("/auth/me")
async def get_me(user=Depends(get_current_user)):
    doc = users_col.find_one({"email": user["email"]})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse({
        "status": "success",
        "user_id": user["user_id"],
        "email": user["email"],
        "name": doc.get("name", ""),
        "created_at": doc.get("created_at"),
    })


# ── Auth pages ────────────────────────────────────────────────────────────────
@auth_router.get("/login")
async def login_page():
    return HTMLResponse(_login_html())

@auth_router.get("/register")
async def register_page():
    return HTMLResponse(_register_html())


def _auth_styles() -> str:
    return """
<style>
:root {
  --p1:#27AE60;--p2:#2ECC71;--p3:#A8E6CF;--p4:#E8F8EF;
  --tx:#1A2332;--tx2:#4A5568;--tx3:#9CA3AF;
  --card:#fff;--bg:#F0F4F8;--bg2:#F7FAFC;--bd:#E2E8F0;--bd2:#F0F4F8;
  --r:12px;--sh:0 2px 8px rgba(0,0,0,.07);--shm:0 6px 20px rgba(0,0,0,.12);
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',Tahoma,sans-serif;background:linear-gradient(135deg,#1a6b3a 0%,#0d4a27 60%,#052e17 100%);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1rem;}
.auth-wrap{width:100%;max-width:420px;}
.auth-logo{text-align:center;color:#fff;margin-bottom:1.5rem;}
.auth-logo h1{font-size:1.6rem;font-weight:800;letter-spacing:-.5px;}
.auth-logo p{font-size:.85rem;opacity:.8;margin-top:.25rem;}
.auth-card{background:var(--card);border-radius:16px;padding:2rem;box-shadow:0 20px 60px rgba(0,0,0,.25);}
.auth-card h2{font-size:1.25rem;font-weight:700;color:var(--tx);margin-bottom:1.5rem;}
.field{margin-bottom:1rem;}
.field label{display:block;font-size:.8rem;font-weight:600;color:var(--tx2);margin-bottom:.35rem;}
.field input{width:100%;padding:.7rem .9rem;border:1.5px solid var(--bd);border-radius:8px;font-size:.9rem;font-family:inherit;outline:none;transition:border .18s;}
.field input:focus{border-color:var(--p1);}
.btn-submit{width:100%;padding:.8rem;background:linear-gradient(135deg,var(--p1),var(--p2));color:#fff;border:none;border-radius:8px;font-size:.95rem;font-weight:700;cursor:pointer;transition:opacity .18s;margin-top:.5rem;}
.btn-submit:hover{opacity:.9;}
.btn-submit:disabled{opacity:.6;cursor:not-allowed;}
.auth-footer{text-align:center;margin-top:1.25rem;font-size:.83rem;color:var(--tx2);}
.auth-footer a{color:var(--p1);font-weight:600;text-decoration:none;}
.auth-footer a:hover{text-decoration:underline;}
.error-msg{background:#FEE8E8;border:1px solid #FCA5A5;border-radius:8px;padding:.65rem .85rem;font-size:.83rem;color:#B91C1C;margin-bottom:1rem;display:none;}
.success-msg{background:#E8F5EE;border:1px solid var(--p3);border-radius:8px;padding:.65rem .85rem;font-size:.83rem;color:#166534;margin-bottom:1rem;display:none;}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:.4rem;}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
"""


def _login_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Login — PlantDoc AI</title>
  {_auth_styles()}
</head>
<body>
<div class="auth-wrap">
  <div class="auth-logo">
    <h1>🌿 PlantDoc AI</h1>
    <p>Sign in to your farming dashboard</p>
  </div>
  <div class="auth-card">
    <h2>Welcome back 👋</h2>
    <div class="error-msg" id="errMsg"></div>
    <div class="success-msg" id="okMsg"></div>
    <div class="field">
      <label>Email address</label>
      <input type="email" id="email" placeholder="you@example.com" autocomplete="email">
    </div>
    <div class="field">
      <label>Password</label>
      <input type="password" id="password" placeholder="••••••••" autocomplete="current-password">
    </div>
    <button class="btn-submit" id="loginBtn" onclick="doLogin()">Sign In</button>
    <div class="auth-footer">
      Don't have an account? <a href="/register">Create one free</a>
    </div>
  </div>
</div>
<script>
const loginBtn = document.getElementById('loginBtn');
document.getElementById('password').addEventListener('keydown', e => {{ if(e.key==='Enter') doLogin(); }});

async function doLogin() {{
  const email    = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const errEl    = document.getElementById('errMsg');
  const okEl     = document.getElementById('okMsg');
  errEl.style.display = 'none';
  okEl.style.display  = 'none';

  if(!email || !password) {{ showErr('Please enter your email and password.'); return; }}

  loginBtn.disabled = true;
  loginBtn.innerHTML = '<span class="spinner"></span>Signing in…';

  try {{
    const res  = await fetch('/auth/login', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{email, password}})
    }});
    const data = await res.json();
    if(data.status === 'success') {{
      localStorage.setItem('plantdoc_token', data.token);
      localStorage.setItem('plantdoc_name',  data.name);
      localStorage.setItem('plantdoc_email', data.email);
      okEl.textContent = 'Logged in! Redirecting…';
      okEl.style.display = 'block';
      setTimeout(() => window.location.href = '/', 800);
    }} else {{
      showErr(data.detail || 'Login failed');
    }}
  }} catch(e) {{
    showErr('Network error — please try again.');
  }} finally {{
    loginBtn.disabled = false;
    loginBtn.textContent = 'Sign In';
  }}
}}
function showErr(msg) {{
  const el = document.getElementById('errMsg');
  el.textContent = msg;
  el.style.display = 'block';
}}
</script>
</body>
</html>"""


def _register_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Register — PlantDoc AI</title>
  {_auth_styles()}
</head>
<body>
<div class="auth-wrap">
  <div class="auth-logo">
    <h1>🌿 PlantDoc AI</h1>
    <p>Your AI-powered plant health companion</p>
  </div>
  <div class="auth-card">
    <h2>Create your account 🌱</h2>
    <div class="error-msg" id="errMsg"></div>
    <div class="success-msg" id="okMsg"></div>
    <div class="field">
      <label>Full name</label>
      <input type="text" id="name" placeholder="Ravi Kumar" autocomplete="name">
    </div>
    <div class="field">
      <label>Email address</label>
      <input type="email" id="email" placeholder="you@example.com" autocomplete="email">
    </div>
    <div class="field">
      <label>Password <span style="color:var(--tx3);font-weight:400;">(min 8 chars, 1 number)</span></label>
      <input type="password" id="password" placeholder="••••••••" autocomplete="new-password">
    </div>
    <button class="btn-submit" id="regBtn" onclick="doRegister()">Create Account</button>
    <div class="auth-footer">
      Already have an account? <a href="/login">Sign in</a>
    </div>
  </div>
</div>
<script>
const regBtn = document.getElementById('regBtn');
document.getElementById('password').addEventListener('keydown', e => {{ if(e.key==='Enter') doRegister(); }});

async function doRegister() {{
  const name     = document.getElementById('name').value.trim();
  const email    = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const errEl    = document.getElementById('errMsg');
  const okEl     = document.getElementById('okMsg');
  errEl.style.display = 'none';
  okEl.style.display  = 'none';

  if(!name || !email || !password) {{ showErr('All fields are required.'); return; }}

  regBtn.disabled = true;
  regBtn.innerHTML = '<span class="spinner"></span>Creating account…';

  try {{
    const res  = await fetch('/auth/register', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{name, email, password}})
    }});
    const data = await res.json();
    if(data.status === 'success') {{
      localStorage.setItem('plantdoc_token', data.token);
      localStorage.setItem('plantdoc_name',  data.name);
      localStorage.setItem('plantdoc_email', data.email);
      okEl.textContent = 'Account created! Taking you home…';
      okEl.style.display = 'block';
      setTimeout(() => window.location.href = '/', 900);
    }} else {{
      showErr(data.detail || 'Registration failed');
    }}
  }} catch(e) {{
    showErr('Network error — please try again.');
  }} finally {{
    regBtn.disabled = false;
    regBtn.textContent = 'Create Account';
  }}
}}
function showErr(msg) {{
  const el = document.getElementById('errMsg');
  el.textContent = msg;
  el.style.display = 'block';
}}
</script>
</body>
</html>"""
