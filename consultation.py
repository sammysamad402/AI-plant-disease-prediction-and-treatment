"""
consultation.py — AI Expert Consultation with Unified PlantDoc AI Design
"""

from fastapi import APIRouter, Request, Depends
from auth import get_current_user
from fastapi.responses import JSONResponse, HTMLResponse
import os, logging, httpx

logger = logging.getLogger(__name__)
consultation_router = APIRouter()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LANGUAGE_INSTRUCTIONS = {
    "english": "Respond in English.",
    "hindi":   "हमेशा हिंदी में जवाब दें। सरल और आसान हिंदी इस्तेमाल करें।",
    "marathi": "नेहमी मराठीत उत्तर द्या. सोपी मराठी वापरा.",
    "telugu":  "దయచేసి తెలుగులో సమాధానం ఇవ్వండి.",
    "tamil":   "தயவுசெய்து தமிழில் பதில் சொல்லுங்கள்.",
    "kannada": "ದಯವಿಟ್ಟು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "bengali": "দয়া করে বাংলায় উত্তর দিন।",
    "gujarati": "કૃપા કરીને ગુજરાતીમાં જવાબ આપો.",
    "punjabi": "ਕਿਰਪਾ ਕਰਕੇ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।",
}

BASE_SYSTEM_PROMPT = """You are a friendly agricultural expert assistant called "AgriDoc" helping farmers in India understand plant diseases and farming problems.

RULES:
1. Use VERY simple, plain language — imagine talking to a farmer with no technical background
2. Avoid jargon. If you must use a technical term, explain it immediately
3. Give PRACTICAL, actionable advice — what to buy, where to apply, how much to use
4. Be warm, patient, and encouraging
5. When recommending treatments, always mention cheap/local options first, organic alternatives when possible, and safety precautions
6. Use bullet points for steps. Keep answers concise but complete
7. If unsure, say so and recommend seeing a local agricultural officer (Krishi Sevak / KVK)
8. Always end with an encouraging note

Diseases detected by the app:
- Anthracnose: dark spots, fungal, use copper fungicide
- Powdery Mildew: white powder on leaves, use sulfur spray or milk solution
- Leaf Crinkle: viral, spread by insects, remove infected plants
- Yellow Mosaic: viral, yellow patches, control whitefly insects
- Healthy: no disease, keep up good practices"""


@consultation_router.get('/consultation')
async def consultation_page():
    return HTMLResponse(content=_html())


@consultation_router.post('/ask-expert')
async def ask_expert(request: Request,user=Depends(get_current_user)):
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        history = body.get("history", [])
        disease_context = body.get("disease_context", "")
        language = body.get("language", "english").lower()
        image_base64 = body.get("image_base64", "")

        if not user_message and not image_base64:
            return JSONResponse({"status": "error", "detail": "Please type a message or upload a photo."}, status_code=400)
        if not OPENAI_API_KEY:
            return JSONResponse({"status": "error", "detail": "OPENAI_API_KEY not set."}, status_code=500)

        lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"])
        system_prompt = BASE_SYSTEM_PROMPT + f"\n\nLANGUAGE INSTRUCTION: {lang_instr}"
        messages = [{"role": "system", "content": system_prompt}]

        for turn in history[-10:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        display_msg = user_message or "Please analyze this plant image."
        if disease_context:
            display_msg = f"[Plant scan context: {disease_context}]\n\nMy question: {display_msg}"

        if image_base64:
            if "," in image_base64:
                header, raw_b64 = image_base64.split(",", 1)
                mt = "image/png" if "png" in header else ("image/webp" if "webp" in header else "image/jpeg")
            else:
                raw_b64, mt = image_base64, "image/jpeg"
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{raw_b64}", "detail": "low"}},
                {"type": "text", "text": display_msg}
            ]
        else:
            user_content = display_msg

        messages.append({"role": "user", "content": user_content})

        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "max_tokens": 1024, "messages": messages}
            )

        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", resp.text)
            return JSONResponse({"status": "error", "detail": f"OpenAI error: {err}"}, status_code=500)

        reply = resp.json()["choices"][0]["message"]["content"]
        return JSONResponse({"status": "success", "reply": reply})

    except httpx.TimeoutException:
        return JSONResponse({"status": "error", "detail": "Request timed out."}, status_code=504)
    except Exception as e:
        logger.error(f"Consultation error: {e}", exc_info=True)
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


def _html() -> str:
    from shared_ui import shared_head, build_nav, TOAST_SCRIPT
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Expert Consultation")}<style>
.consult-layout{{display:grid;grid-template-columns:280px 1fr;gap:1.5rem;margin-top:1.5rem;}}
@media(max-width:850px){{.consult-layout{{grid-template-columns:1fr;}}
  .sidebar-panel{{order:2;}} .chat-panel{{order:1;}}}}

/* sidebar */
.sidebar-panel{{display:flex;flex-direction:column;gap:1rem;}}

/* chat */
.chat-panel{{display:flex;flex-direction:column;}}
.chat-window{{flex:1;min-height:420px;max-height:520px;overflow-y:auto;padding:1rem;
  display:flex;flex-direction:column;gap:.75rem;background:var(--bg2);
  border-radius:var(--r) var(--r) 0 0;border:1px solid var(--bd);border-bottom:none;}}
.chat-window::-webkit-scrollbar{{width:5px;}}
.chat-window::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:5px;}}

/* messages */
.msg{{display:flex;gap:.6rem;max-width:85%;}}
.msg.user{{align-self:flex-end;flex-direction:row-reverse;}}
.msg-avatar{{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1rem;flex-shrink:0;}}
.msg.user .msg-avatar{{background:var(--p1);}}
.msg.bot .msg-avatar{{background:var(--ac);}}
.msg-bubble{{padding:.65rem .9rem;border-radius:12px;font-size:.87rem;line-height:1.55;}}
.msg.user .msg-bubble{{background:var(--p1);color:#fff;border-radius:12px 4px 12px 12px;}}
.msg.bot .msg-bubble{{background:#fff;color:var(--tx);border:1px solid var(--bd);
  border-radius:4px 12px 12px 12px;}}
.msg-bubble b{{font-weight:700;}}
.msg-bubble ul{{padding-left:1.2rem;margin:.3rem 0;}}
.msg-bubble li{{margin:.15rem 0;}}
.save-reply-btn{{display:block;margin-top:.4rem;font-size:.72rem;color:var(--p2);
  background:none;border:none;cursor:pointer;text-align:left;padding:0;}}
.save-reply-btn:hover{{text-decoration:underline;}}

.typing-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--tx3);margin:0 2px;animation:bounce .9s infinite;}}
.typing-dot:nth-child(2){{animation-delay:.15s;}}
.typing-dot:nth-child(3){{animation-delay:.3s;}}
@keyframes bounce{{0%,80%,100%{{transform:translateY(0)}}40%{{transform:translateY(-6px)}}}}

/* input bar */
.chat-input-bar{{background:#fff;border:1px solid var(--bd);border-radius:0 0 var(--r) var(--r);
  padding:.75rem 1rem;display:flex;gap:.5rem;align-items:flex-end;}}
.chat-textarea{{flex:1;resize:none;border:1.5px solid var(--bd);border-radius:8px;
  padding:.55rem .8rem;font-size:.88rem;font-family:inherit;outline:none;
  max-height:120px;transition:border .18s;line-height:1.5;}}
.chat-textarea:focus{{border-color:var(--p1);}}
.img-strip{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;
  padding:.4rem .7rem;display:none;align-items:center;gap:.5rem;font-size:.78rem;
  color:var(--tx2);margin-bottom:.4rem;}}
.img-strip.visible{{display:flex;}}
.img-thumb{{width:36px;height:36px;border-radius:6px;object-fit:cover;}}

/* voice button */
.btn-voice{{background:var(--bg2);color:var(--tx2);border:1.5px solid var(--bd);}}
.btn-voice.recording{{background:#FEE8E8;color:var(--err);border-color:var(--err);
  animation:pulse 1s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.6}}}}

/* quick questions */
.quick-q{{display:flex;flex-direction:column;gap:.4rem;}}
.q-btn{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;
  padding:.5rem .75rem;font-size:.8rem;text-align:left;cursor:pointer;
  color:var(--tx2);transition:all .18s;font-family:inherit;}}
.q-btn:hover{{background:var(--p1);color:#fff;border-color:var(--p1);}}

/* toggle */
.toggle-row{{display:flex;align-items:center;gap:.6rem;padding:.5rem 0;}}
.toggle-lbl{{font-size:.83rem;color:var(--tx2);flex:1;}}
.toggle-sw{{position:relative;width:38px;height:21px;flex-shrink:0;}}
.toggle-sw input{{opacity:0;width:0;height:0;}}
.toggle-sl{{position:absolute;inset:0;background:#ccc;border-radius:21px;cursor:pointer;transition:.3s;}}
.toggle-sl:before{{content:'';position:absolute;width:15px;height:15px;left:3px;bottom:3px;
  background:#fff;border-radius:50%;transition:.3s;}}
.toggle-sw input:checked+.toggle-sl{{background:var(--p1);}}
.toggle-sw input:checked+.toggle-sl:before{{transform:translateX(17px);}}

.welcome-card{{text-align:center;padding:2.5rem 1.5rem;color:var(--tx2);}}
.welcome-icon{{font-size:3rem;margin-bottom:.75rem;}}
.welcome-card h3{{font-size:1.05rem;font-weight:700;margin-bottom:.4rem;color:var(--tx);}}
.welcome-card p{{font-size:.85rem;}}
</style>
</head>
<body>
{build_nav("/consultation")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>💬 Expert Consultation</h1>
    <p>Chat with AgriDoc AI in your language — type, speak, or upload a plant photo</p>
    <div class="hero-badges">
      <span class="hero-badge">🌐 9 Indian Languages</span>
      <span class="hero-badge">🔊 Voice Replies</span>
      <span class="hero-badge">📔 Save to Diary</span>
    </div>
  </div>
</div>

<div class="pda-container">
  <div class="consult-layout">

    <!-- Sidebar -->
    <div class="sidebar-panel">

      <!-- Language -->
      <div class="pda-card">
        <div class="pda-card-header">🌐 Language</div>
        <div class="pda-card-body">
          <select id="languageSelect" class="form-control">
            <option value="english">🇬🇧 English</option>
            <option value="hindi">🇮🇳 हिंदी (Hindi)</option>
            <option value="marathi">मराठी (Marathi)</option>
            <option value="telugu">తెలుగు (Telugu)</option>
            <option value="tamil">தமிழ் (Tamil)</option>
            <option value="kannada">ಕನ್ನಡ (Kannada)</option>
            <option value="bengali">বাংলা (Bengali)</option>
            <option value="gujarati">ગુજરાતી (Gujarati)</option>
            <option value="punjabi">ਪੰਜਾਬੀ (Punjabi)</option>
          </select>
          <div class="toggle-row" style="margin-top:.6rem;">
            <span class="toggle-lbl">🔊 Auto-speak replies</span>
            <label class="toggle-sw"><input type="checkbox" id="ttsToggle"><span class="toggle-sl"></span></label>
          </div>
        </div>
      </div>

      <!-- Disease context -->
      <div class="pda-card">
        <div class="pda-card-header">🔬 Detection Context</div>
        <div class="pda-card-body">
          <label class="form-label">Paste detection result (optional)</label>
          <textarea id="diseaseContext" class="form-control" rows="3"
            placeholder="e.g. Anthracnose: 87.3% confidence, High severity"></textarea>
          <p style="font-size:.75rem;color:var(--tx3);margin-top:.4rem;">This gives AgriDoc context about your plant's condition.</p>
        </div>
      </div>

      <!-- Quick questions -->
      <div class="pda-card">
        <div class="pda-card-header">⚡ Quick Questions</div>
        <div class="pda-card-body">
          <div class="quick-q">
            <button class="q-btn" onclick="sendQuick(this)">🍂 What causes anthracnose?</button>
            <button class="q-btn" onclick="sendQuick(this)">🌫️ How to treat powdery mildew?</button>
            <button class="q-btn" onclick="sendQuick(this)">🛡️ Best organic fungicides?</button>
            <button class="q-btn" onclick="sendQuick(this)">💧 How often should I water?</button>
            <button class="q-btn" onclick="sendQuick(this)">🌱 Signs of a healthy plant?</button>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="pda-card">
        <div class="pda-card-header">⚙️ Actions</div>
        <div class="pda-card-body" style="display:flex;flex-direction:column;gap:.5rem;">
          <button class="btn btn-outline btn-block btn-sm" onclick="saveFullChat()">📔 Save Chat to Diary</button>
          <button class="btn btn-ghost btn-block btn-sm" onclick="clearChat()">🗑️ Clear Chat</button>
        </div>
      </div>
    </div>

    <!-- Chat panel -->
    <div class="chat-panel pda-card" style="padding:0;">
      <!-- Messages -->
      <div class="chat-window" id="chatMessages">
        <div class="welcome-card">
          <div class="welcome-icon">🌾</div>
          <h3>Namaste! I'm AgriDoc</h3>
          <p>Your AI agricultural expert. Ask me about plant diseases, treatments, farming tips, or upload a photo for analysis!</p>
        </div>
      </div>

      <!-- Image preview strip -->
      <div id="imageStrip" class="img-strip">
        <img id="previewThumb" class="img-thumb" alt="">
        <span id="previewName" style="flex:1;"></span>
        <button class="btn btn-xs btn-ghost" onclick="removeImage()">✕ Remove</button>
      </div>

      <!-- Input bar -->
      <div class="chat-input-bar">
        <label style="cursor:pointer;" title="Upload photo">
          <input type="file" id="photoInput" accept="image/*" style="display:none;" onchange="handlePhoto(event)">
          <span class="btn btn-ghost btn-sm" style="padding:.55rem .7rem;">📷</span>
        </label>
        <button class="btn btn-voice btn-sm" id="voiceBtn" onclick="toggleVoice()" title="Voice input">🎤</button>
        <span id="voiceStatus" style="font-size:.72rem;color:var(--err);display:none;">● Recording...</span>
        <textarea id="chatInput" class="chat-textarea" rows="1" placeholder="Type your question or describe your problem..."
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMessage();}}"></textarea>
        <button id="sendBtn" class="btn btn-primary btn-sm" onclick="sendMessage()">Send ↗</button>
      </div>
    </div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
let chatHistory = [], selectedImage = null, isRecording = false, recognition = null, currentUtterance = null;
let voicesLoaded = false;

window.speechSynthesis.onvoiceschanged = () => {{
  voicesLoaded = true;
}};
const TTS_LANG_MAP = {{
  english:'en-IN', hindi:'hi-IN', marathi:'mr-IN', telugu:'te-IN',
  tamil:'ta-IN', kannada:'kn-IN', bengali:'bn-IN', gujarati:'gu-IN', punjabi:'pa-IN'
}};

// ── Auto-resize textarea ──────────────────────────────────────────────────────
const chatInput = document.getElementById('chatInput');
chatInput.addEventListener('input', () => {{
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}});

// ── Append message ────────────────────────────────────────────────────────────
function appendMessage(role, text, imgBase64, isTyping) {{
  const win = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `msg ${{role}}`;
  div.id = isTyping ? 'typing-indicator' : '';

  const avatarHtml = role === 'user'
    ? '<div class="msg-avatar">👤</div>'
    : '<div class="msg-avatar">🌾</div>';

  let bubbleContent = '';
  if(isTyping) {{
    bubbleContent = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  }} else {{
    if(imgBase64) bubbleContent += `<img src="${{imgBase64}}" style="max-width:180px;border-radius:8px;display:block;margin-bottom:.4rem;">`;
    bubbleContent += formatText(text);
    if(role === 'bot' && text) {{
  bubbleContent += `
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">

      <button onclick="autoSpeak(\`${{text.replace(/`/g,"'")}}\`, true)">🔊 Speak</button>
      <button onclick="stopSpeaking()">⛔ Stop</button>

      <button class="save-reply-btn" onclick="saveSingleReply(\`${{text.replace(/`/g,"'")}}\`, this)">
        📔 Save
      </button>

    </div>
  `;
}}
  }}

  div.innerHTML = role === 'user'
    ? `${{avatarHtml}}<div class="msg-bubble">${{bubbleContent}}</div>`
    : `${{avatarHtml}}<div class="msg-bubble">${{bubbleContent}}</div>`;

  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}}

function formatText(text) {{
  return text
  .replace(/###\\s?(.*)/g, '<br><b>$1</b><br>')              // ✅ REMOVE ###
  .replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>')
  .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
  .replace(/\\n\\n/g, '</p><p>')
  .replace(/\\n- /g, '<br>• ')
  .replace(/\\n/g, '<br>');
}}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {{
  const text = chatInput.value.trim();
  if(!text && !selectedImage) return;

  const imgToSend = selectedImage;
  const displayText = text || '📸 Please analyze this plant photo.';

  chatInput.value = '';
  chatInput.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;

  appendMessage('user', displayText, imgToSend?.base64 || null, false);
  removeImage();
  appendMessage('bot', '', null, true);

  try {{
    const res = await authFetch('/ask-expert', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        message: text || 'Please analyze this plant image for any disease.',
        history: chatHistory,
        disease_context: document.getElementById('diseaseContext').value.trim(),
        language: document.getElementById('languageSelect').value,
        image_base64: imgToSend?.base64 || ''
      }})
    }});

    const data = await res.json();
    document.getElementById('typing-indicator')?.remove();

    if(data.status === 'success') {{
      appendMessage('bot', data.reply, null, false);
      autoSpeak(data.reply);

      chatHistory.push({{ role:'user', content:displayText }});
      chatHistory.push({{ role:'assistant', content:data.reply }});

      if(chatHistory.length > 500) {{
        chatHistory = chatHistory.slice(-500);
      }}
    }} else {{
      appendMessage('bot', 'Error: ' + data.detail, null, false);
    }}

  }} catch(e) {{
    document.getElementById('typing-indicator')?.remove();
    appendMessage('bot', 'Network error. Try again.', null, false);
  }} finally {{
    document.getElementById('sendBtn').disabled = false;
    chatInput.focus();
  }}
}}

// ── Voice input ───────────────────────────────────────────────────────────────
function toggleVoice() {{
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR) {{ alert('Voice input works only in Chrome or Edge.'); return; }}
  isRecording ? stopVoice() : startVoice();
}}
function startVoice() {{
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  const lm = {{english:'en-IN',hindi:'hi-IN',marathi:'mr-IN',telugu:'te-IN',
               tamil:'ta-IN',kannada:'kn-IN',bengali:'bn-IN',gujarati:'gu-IN',punjabi:'pa-IN'}};
  recognition.lang = lm[document.getElementById('languageSelect').value] || 'en-IN';
  recognition.continuous = false; recognition.interimResults = false;
  recognition.onstart = () => {{
    isRecording = true;
    document.getElementById('voiceBtn').classList.add('recording');
    document.getElementById('voiceStatus').style.display = 'inline';
  }};
  recognition.onresult = e => {{ chatInput.value = e.results[0][0].transcript; stopVoice(); }};
  recognition.onerror = e => {{ stopVoice(); if(e.error==='not-allowed') alert('Microphone permission denied.'); }};
  recognition.onend = stopVoice;
  recognition.start();
}}
function stopVoice() {{
  isRecording = false;
  document.getElementById('voiceBtn').classList.remove('recording');
  document.getElementById('voiceStatus').style.display = 'none';
  try {{ recognition?.stop(); }} catch(e) {{}}
}}

// ── TTS ───────────────────────────────────────────────────────────────────────
function autoSpeak(text, force=false) {{
  if((!document.getElementById('ttsToggle').checked && !force) || !window.speechSynthesis) return;

  const lang = document.getElementById('languageSelect').value;

  const clean = text
  .replace(/###/g, '')              // remove ###
  .replace(/\\*\\*/g, '')           // remove **
  .replace(/\\*/g, '')              // remove *
  .replace(/-/g, '')                // remove dashes
  .replace(/\\n/g, ' ')             // remove line breaks
  .replace(/[<>]/g, '')
  .trim();

  const utt = new SpeechSynthesisUtterance(clean);
  utt.lang = TTS_LANG_MAP[lang] || 'en-IN';
  utt.rate = 0.9;

  const voices = window.speechSynthesis.getVoices();
  const match = voices.find(v => v.lang.startsWith(utt.lang.split('-')[0]));
  if(match) utt.voice = match;

  window.speechSynthesis.cancel();
  setTimeout(() => {{
  speakInChunks(clean, lang);
}}, 100);
}}

// ── Photo ─────────────────────────────────────────────────────────────────────
function handlePhoto(e) {{
  const file = e.target.files[0];
  if(!file) return;
  if(file.size > 5*1024*1024) {{ alert('Photo too large. Max 5 MB.'); return; }}
  const reader = new FileReader();
  reader.onload = ev => {{
    selectedImage = {{base64: ev.target.result, name: file.name}};
    document.getElementById('previewThumb').src = ev.target.result;
    document.getElementById('previewName').textContent = file.name;
    document.getElementById('imageStrip').classList.add('visible');
  }};
  reader.readAsDataURL(file);
  e.target.value = '';
}}
function removeImage() {{
  selectedImage = null;
  document.getElementById('imageStrip').classList.remove('visible');
  document.getElementById('previewThumb').src = '';
}}

// ── Quick questions ───────────────────────────────────────────────────────────
function sendQuick(btn) {{
  chatInput.value = btn.textContent.trim().replace(/^[^\\w\\s]+\\s*/u, '').trim();
  sendMessage();
}}

// ── Save to diary ─────────────────────────────────────────────────────────────
async function saveSingleReply(replyText, btn) {{
  await _saveDiary(`AgriDoc Reply — ${{new Date().toLocaleDateString()}}`, chatHistory);
  btn.textContent = '✅ Saved!';
  setTimeout(() => btn.textContent = '📔 Save reply to diary', 3000);
}}
async function saveFullChat() {{
  if(!chatHistory.length) {{ showToast('Nothing to save — have a conversation first!'); return; }}
  const title = `Farm Consultation — ${{new Date().toLocaleDateString('en-IN', {{day:'numeric',month:'short',year:'numeric'}})}}`;
  await _saveDiary(title, chatHistory);
}}
async function _saveDiary(title, messages) {{
  try {{
    const lang = document.getElementById('languageSelect').value;
    const context = document.getElementById('diseaseContext').value.trim();
    const tags = ['consultation'];
    if(context) tags.push(...context.split(',').map(s=>s.trim().split(':')[0].trim()).filter(Boolean).slice(0,3));
    const res = await fetch('/diary/save', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{type:'chat', title, language:lang, content:{{messages, context_used:context}}, tags}})
    }});
    const d = await res.json();
    showToast(d.status === 'success' ? '📔 Saved to Farm Diary!' : 'Could not save: ' + d.detail);
  }} catch(e) {{ showToast('Network error while saving.'); }}
}}
async function translateAndSpeak(select, text) {{
  const lang = select.value;
  if(!lang) return;

  try {{
    const res = await authFetch('/ask-expert', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        message: "Translate this into " + lang + ": " + text,
        history: [],
        language: lang
      }})
    }});

    const data = await res.json();

    if(data.status === 'success') {{
      // ✅ ONLY SPEAK (NO NEW MESSAGE)
      autoSpeak(data.reply, true);
    }} else {{
      alert("Translation failed");
    }}

  }} catch(e) {{
    alert("Error translating");
  }}
}}
function speakInChunks(text, lang) {{
  let chunks = text.match(/.{1,200}(\s|$)/g);

  if(!chunks) {{
    chunks = [text];
  }}

  let i = 0;

  function speakNext() {{
    if(i >= chunks.length) return;

    const utt = new SpeechSynthesisUtterance(chunks[i]);
    utt.lang = TTS_LANG_MAP[lang] || 'en-IN';
    utt.rate = 0.9;

    const voices = window.speechSynthesis.getVoices();
    let match = voices.find(v => v.lang.startsWith(utt.lang.split('-')[0]));

    if(!match) match = voices.find(v => v.lang.startsWith('hi'));
    if(!match) match = voices.find(v => v.lang.startsWith('en'));

    if(match) utt.voice = match;

    utt.onend = () => {{
      i++;
      speakNext();
    }};

    window.speechSynthesis.speak(utt);
  }}

  speakNext();
}}
function stopSpeaking() {{
  if(window.speechSynthesis) {{
    window.speechSynthesis.cancel();
  }}
}}
// ── Clear chat ────────────────────────────────────────────────────────────────
function clearChat() {{
  if(!confirm('Clear the chat?')) return;
  chatHistory = []; removeImage();
  window.speechSynthesis?.cancel();
  document.getElementById('chatMessages').innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">🌾</div>
      <h3>Chat Cleared</h3>
      <p>Start a new conversation — type, speak, or upload a photo!</p>
    </div>`;
}}
</script>
</body>
</html>"""