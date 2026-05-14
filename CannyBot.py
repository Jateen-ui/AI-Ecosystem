"""
⬡ Multimodal Chatbot — Hugging Face Edition
Model  : Google Gemma
Backend: Hugging Face Inference API  (no GPU needed)
UI     : Streamlit
Supports: Text · PDF · Images · Audio · Video
"""

import os
import base64
import tempfile
import mimetypes
from pathlib import Path

import streamlit as st
import fitz                 # PyMuPDF  — PDF → images
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────
# Page config  ← must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Chatbot · HF Gemma",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #151820;
    --border:  #252936;
    --accent:  #ff6b35;
    --accent2: #f7c59f;
    --hf:      #ff9d00;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body { font-family:'Sora',sans-serif !important; background:var(--bg) !important; color:var(--text) !important; }
.stApp { background:var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { background:var(--surface) !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown { color:var(--text) !important; }

.main .block-container { padding-top:1.5rem !important; }

/* Header */
.chat-header { text-align:center; padding:1.2rem 0 0.6rem; border-bottom:1px solid var(--border); margin-bottom:1.2rem; }
.chat-header h1 { font-size:1.9rem; font-weight:700; background:linear-gradient(90deg,#ff6b35,#f7c59f); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
.chat-header p { color:var(--muted); font-size:0.82rem; font-family:'DM Mono',monospace; margin-top:0.3rem; }
.badges { display:flex; gap:0.4rem; justify-content:center; flex-wrap:wrap; margin-top:0.6rem; }
.badge { background:var(--bg); border:1px solid var(--border); color:var(--hf); font-size:0.7rem; font-family:'DM Mono',monospace; padding:0.18rem 0.6rem; border-radius:999px; }

/* Chat bubbles */
.msg-row { display:flex; margin-bottom:1rem; align-items:flex-start; gap:0.6rem; }
.msg-row.user { flex-direction:row-reverse; }
.avatar { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.avatar.user-av { background:linear-gradient(135deg,#ff6b35,#f7c59f); }
.avatar.bot-av  { background:#1e2535; border:1px solid var(--border); }
.bubble { max-width:75%; padding:0.75rem 1rem; border-radius:14px; font-size:0.9rem; line-height:1.6; white-space:pre-wrap; word-break:break-word; }
.bubble.user { background:#2a1a0e; border:1px solid #5a3020; border-top-right-radius:4px; }
.bubble.bot  { background:#131820; border:1px solid var(--border); border-top-left-radius:4px; }
.file-chip { display:inline-block; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.25rem 0.65rem; font-size:0.75rem; font-family:'DM Mono',monospace; color:var(--hf); margin-bottom:0.4rem; }

/* Input */
.stTextArea textarea { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text) !important; font-family:'Sora',sans-serif !important; }
.stTextArea textarea:focus { border-color:var(--accent) !important; box-shadow:none !important; }

/* Buttons */
.stButton > button { background:linear-gradient(135deg,#ff6b35,#f7931e) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-family:'Sora',sans-serif !important; font-weight:600 !important; width:100%; }
.stButton > button:hover { opacity:0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] { background:var(--surface) !important; border:1px dashed var(--border) !important; border-radius:10px !important; }

/* Tip box */
.tip-box { background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:1rem 1.1rem; font-size:0.8rem; color:var(--muted); font-family:'DM Mono',monospace; line-height:1.8; margin-top:1rem; }
.tip-box b { color:var(--text); }
.tip-box .hf  { color:var(--hf); }

hr { border-color:var(--border) !important; }
#MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────
HF_MODEL = "google/gemma-3-27b-it"

# Featherless AI (and most providers) allow at most 8 images per request.
# We use 7 as our hard cap to leave a safe margin.
MAX_IMAGES = 7


# ─────────────────────────────────────────────────────────────
# File processing helpers
# ─────────────────────────────────────────────────────────────

def classify_file(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":                                                   return "pdf"
    if ext in {".jpg",".jpeg",".png",".gif",".webp",".bmp",".tiff"}:   return "image"
    if ext in {".mp3",".wav",".ogg",".flac",".m4a",".aac",".wma"}:     return "audio"
    if ext in {".mp4",".avi",".mov",".mkv",".webm",".flv",".wmv"}:     return "video"
    return "unknown"

def file_emoji(ftype: str) -> str:
    return {"pdf":"📄","image":"🖼️","audio":"🎵","video":"🎬"}.get(ftype,"📎")

def img_bytes_to_b64(data: bytes, fmt="PNG") -> str:
    return base64.b64encode(data).decode()

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def pdf_to_b64_images(data: bytes) -> tuple[list[str], int]:
    """Return (list of base64-PNG strings, total_pages). Pages capped at MAX_IMAGES."""
    doc = fitz.open(stream=data, filetype="pdf")
    total = len(doc)
    pages = []
    for i, page in enumerate(doc):
        if i >= MAX_IMAGES:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pages.append(base64.b64encode(pix.tobytes("png")).decode())
    doc.close()
    return pages, total

def image_to_b64(data: bytes, filename: str) -> tuple[str, str]:
    """Return (base64_string, media_type)."""
    mime, _ = mimetypes.guess_type(filename)
    supported = {"image/jpeg","image/png","image/gif","image/webp"}
    if mime not in supported:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return pil_to_b64(img), "image/png"
    return base64.b64encode(data).decode(), mime

def extract_video_frames(data: bytes, max_frames: int = MAX_IMAGES - 1) -> list[str]:
    """Return list of base64-PNG frame strings."""
    try:
        import cv2
    except ImportError:
        return []
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data); tmp.flush(); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in range(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / max_frames))
            ret, frame = cap.read()
            if not ret: continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_to_b64(img))
    cap.release()
    try: os.unlink(tmp.name)
    except: pass
    return frames

def transcribe_audio(data: bytes, filename: str) -> str:
    """Transcribe audio via Google Speech Recognition (free)."""
    try:
        import speech_recognition as sr
        ext = Path(filename).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(data); tmp.flush(); tmp.close()
        wav = tmp.name
        if ext != ".wav":
            from pydub import AudioSegment
            seg = AudioSegment.from_file(tmp.name)
            tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            seg.export(tmp2.name, format="wav")
            wav = tmp2.name
        r = sr.Recognizer()
        with sr.AudioFile(wav) as src:
            audio = r.record(src)
        return r.recognize_google(audio)
    except Exception as e:
        return f"[Transcription failed: {e}. Please describe the audio in your question.]"


# ─────────────────────────────────────────────────────────────
# Build message content for HF Inference API
# Qwen2.5-VL uses the OpenAI-compatible chat format with
# image_url blocks (base64 data URIs)
# ─────────────────────────────────────────────────────────────

def build_user_message(text: str, uploaded_file) -> tuple[dict, str]:
    """
    Returns:
        message  : dict  — { role: 'user', content: [...] }
        label    : str   — display label for the chat bubble
    """
    content = []
    label   = ""

    if uploaded_file is not None:
        raw   = uploaded_file.read()
        fname = uploaded_file.name
        ftype = classify_file(fname)
        label = f"{file_emoji(ftype)} {fname}"

        if ftype == "pdf":
            pages, total_pages = pdf_to_b64_images(raw)
            truncated = total_pages > len(pages)
            notice = (
                f"📄 PDF attached ({total_pages} pages total; "
                f"showing first {len(pages)} due to provider image limit). Analyse all shown pages."
                if truncated else
                f"📄 PDF attached ({total_pages} page(s)). Analyse all pages."
            )
            content.append({"type": "text", "text": notice})
            for b64 in pages:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        elif ftype == "image":
            b64, mime = image_to_b64(raw, fname)
            content.append({"type":"text","text":"🖼️ Image attached."})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        elif ftype == "audio":
            transcript = transcribe_audio(raw, fname)
            content.append({"type":"text","text":f'🎵 Audio transcript:\n"""\n{transcript}\n"""'})

        elif ftype == "video":
            frames = extract_video_frames(raw)
            if frames:
                content.append({"type":"text","text":f"🎬 Video attached ({len(frames)} frames extracted)."})
                for b64 in frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
            else:
                content.append({"type":"text","text":"🎬 [Video uploaded — OpenCV not available for frame extraction. Describe the video content in your question.]"})

        else:
            content.append({"type":"text","text":f"[Attached: {fname} — unsupported type]"})

    if text.strip():
        content.append({"type":"text","text":text})
    elif not uploaded_file:
        content.append({"type":"text","text":"Hello! What can you help me with?"})

    # ── Global safety cap: never exceed MAX_IMAGES image blocks ──
    img_count  = 0
    safe_content = []
    trimmed    = False
    for block in content:
        if block.get("type") == "image_url":
            if img_count >= MAX_IMAGES:
                trimmed = True
                continue
            img_count += 1
        safe_content.append(block)
    if trimmed:
        safe_content.append({
            "type": "text",
            "text": f"⚠️ Some images were dropped to stay within the {MAX_IMAGES}-image provider limit."
        })

    return {"role": "user", "content": safe_content}, label


# ─────────────────────────────────────────────────────────────
# Call Hugging Face Inference API
# ─────────────────────────────────────────────────────────────

def call_hf_api(messages: list[dict], hf_token: str) -> str:
    """
    Uses the HF Inference API via the huggingface_hub InferenceClient.
    provider="auto" lets HF route to the best available provider for the
    model (nebius, together, hyperbolic, etc.) — avoids the
    'Model not supported by provider hf-inference' error.
    Falls back to explicit nebius provider if auto fails.
    """
    from huggingface_hub import InferenceClient

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful multimodal AI assistant. "
            "You can analyze images (including PDF pages rendered as images), "
            "audio transcripts, and video frames. "
            "Answer the user's question using all provided media. "
            "Be thorough, clear, and concise."
        )
    }
    api_messages = [system_msg] + messages

    # As of May 2026, Featherless AI is the active inference provider
    # for google/gemma-3-27b-it. We try it first, then fall back to
    # the 12B variant which has wider provider support.
    providers_to_try = ["featherless-ai", "auto"]
    last_err = ""

    for provider in providers_to_try:
        try:
            client = InferenceClient(
                provider=provider,
                api_key=hf_token,
            )
            response = client.chat.completions.create(
                model=HF_MODEL,
                messages=api_messages,
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_err = str(e)
            # Don't retry on auth errors — token is wrong
            if "401" in last_err or "authorization" in last_err.lower():
                return (
                    "❌ **Invalid Hugging Face token.**\n"
                    "Please check your token in the sidebar.\n"
                    "Get a free token at: https://huggingface.co/settings/tokens"
                )
            # Don't retry on rate limits
            if "429" in last_err or "rate limit" in last_err.lower():
                return (
                    "⚠️ **Rate limit reached.**\n"
                    "Please wait a moment and try again, "
                    "or upgrade your HF plan at https://huggingface.co/pricing"
                )
            # Otherwise try next provider
            continue

    # All providers failed — try smaller 12B model as last resort
    for provider in ["featherless-ai", "auto", "groq", "cerebras", "novita"]:
        try:
            client = InferenceClient(provider=provider, api_key=hf_token)
            response = client.chat.completions.create(
                model="google/gemma-3-12b-it",
                messages=api_messages,
                max_tokens=1024,
                temperature=0.7,
            )
            reply = response.choices[0].message.content
            return f"_(Responded using gemma-3-12b-it fallback)_\n\n{reply}"
        except Exception:
            continue

    # Everything failed
    return (
        f"❌ **All providers and models failed.**\n\n"
        f"Last error: `{last_err}`\n\n"
        f"**Possible fixes:**\n"
        f"- Ensure your HF token has **Read** permissions\n"
        f"- Accept the Gemma licence at: https://huggingface.co/google/gemma-3-27b-it\n"
        f"- The model may be temporarily unavailable — try again in a minute\n"
        f"- Check provider status at: https://huggingface.co/docs/inference-providers"
    )


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "api_messages" not in st.session_state: st.session_state.api_messages = []
if "hf_token"    not in st.session_state: st.session_state.hf_token    = os.environ.get("HF_TOKEN","")


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤗 Multimodal Chatbot")
    st.markdown(f"**Model:** `{HF_MODEL}`")
    st.markdown("**Provider:** featherless-ai → auto (12B fallback)")
    st.markdown("---")

    # HF Token
    token_input = st.text_input(
        "🔑 Hugging Face Token",
        type="password",
        value=st.session_state.hf_token,
        placeholder="hf_...",
        help="Get your free token at huggingface.co/settings/tokens"
    )
    if token_input != st.session_state.hf_token:
        st.session_state.hf_token = token_input

    st.markdown(
        "[Get a free HF token →](https://huggingface.co/settings/tokens)",
        unsafe_allow_html=False
    )

    st.markdown("---")
    st.markdown("### 📎 Attach a File")
    uploaded_file = st.file_uploader(
        label="Upload file",
        type=["pdf","jpg","jpeg","png","gif","webp","bmp","tiff",
              "mp3","wav","ogg","flac","m4a","aac",
              "mp4","avi","mov","mkv","webm"],
        label_visibility="collapsed",
    )

    st.markdown("")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.api_messages = []
        st.rerun()

    st.markdown("""
    <div class="tip-box">
    <b>Supported formats</b><br>
    📄 <b>PDF</b> — pages → images → model<br>
    🖼️ <b>Image</b> — JPG, PNG, WebP, GIF<br>
    🎵 <b>Audio</b> — MP3, WAV, M4A, FLAC<br>
    🎬 <b>Video</b> — MP4, MOV, AVI, MKV<br>
    💬 <b>Text</b> — plain conversation<br>
    <br>
    <span class="hf">💡 Tip:</span> The model is FREE via<br>
    the HF Serverless Inference API.<br>
    Just add your HF token above.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🤗 Multimodal Chatbot</h1>
    <p>Powered by Google Gemma 3 27B · Hugging Face Inference API · Free &amp; Open Source</p>
    <div class="badges">
        <span class="badge">📄 PDF</span>
        <span class="badge">🖼️ Images</span>
        <span class="badge">🎵 Audio</span>
        <span class="badge">🎬 Video</span>
        <span class="badge">💬 Text</span>
        <span class="badge">Apache-2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role     = msg["role"]
    text     = msg["content"]
    label    = msg.get("label","")
    is_user  = (role == "user")

    row_cls  = "user"   if is_user else "bot"
    bub_cls  = "user"   if is_user else "bot"
    av_cls   = "user-av" if is_user else "bot-av"
    icon     = "👤"     if is_user else "🤖"
    chip     = f'<div class="file-chip">{label}</div><br>' if label else ""

    st.markdown(f"""
    <div class="msg-row {row_cls}">
        <div class="avatar {av_cls}">{icon}</div>
        <div class="bubble {bub_cls}">{chip}{text}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Input row
# ─────────────────────────────────────────────────────────────
st.markdown("---")
col_msg, col_btn = st.columns([5, 1])

with col_msg:
    user_text = st.text_area(
        label="Message",
        placeholder="Ask anything about your file, or just chat…",
        label_visibility="collapsed",
        height=90,
        key="user_text",
    )

with col_btn:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    send = st.button("Send ➤", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Send handler
# ─────────────────────────────────────────────────────────────
if send and (user_text.strip() or uploaded_file):

    hf_token = st.session_state.hf_token or os.environ.get("HF_TOKEN","")
    if not hf_token:
        st.error("⚠️ Please enter your Hugging Face token in the sidebar. Get one free at huggingface.co/settings/tokens")
        st.stop()

    # Build the API message (includes media blocks)
    with st.spinner("🔄 Processing file…" if uploaded_file else "🤖 Thinking…"):
        user_api_msg, file_label = build_user_message(user_text, uploaded_file)

    # Add to full API message history
    st.session_state.api_messages.append(user_api_msg)

    # Add to display history (text only)
    st.session_state.messages.append({
        "role":    "user",
        "content": user_text or "(file uploaded)",
        "label":   file_label,
    })

    # Call the model
    with st.spinner(f"🤗 Asking Gemma 3 27B…"):
        reply = call_hf_api(st.session_state.api_messages, hf_token)

    # Store assistant reply in both histories
    st.session_state.api_messages.append({"role":"assistant","content":reply})
    st.session_state.messages.append({"role":"assistant","content":reply})

    st.rerun()