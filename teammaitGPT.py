import streamlit as st
import time
from textwrap import dedent
from urllib.parse import quote
import json
from datetime import datetime
import os
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from sentence_transformers import SentenceTransformer
import sys
import glob

# ---------- SQLite shim for Chroma ----------
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# Optional SDKs (load if installed)
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    OpenAI = None

# ---------- Page & Theme ----------
st.set_page_config(page_title="TeamMait Private Conversation", page_icon="💬", layout="wide")

# ---------- Simple avatars (SVG data URIs) ----------
from urllib.parse import quote as _quote  # noqa: E402
def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + _quote(svg)

DM_SVG = svg_data_uri(
    dedent("""
    <svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>
      <defs>
        <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
          <stop offset='0%' stop-color='#1f2937'/>
          <stop offset='100%' stop-color='#0b1220'/>
        </linearGradient>
      </defs>
      <circle cx='32' cy='32' r='32' fill='url(#g)'/>
      <text x='50%' y='54%' text-anchor='middle' font-family='Inter,Arial' font-size='24' fill='#e5e7eb' font-weight='700'>U</text>
    </svg>
    """).strip()
)

BOT_SVG = svg_data_uri(
    dedent("""
    <svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>
      <circle cx='32' cy='32' r='32' fill='#111827'/>
      <rect x='18' y='20' width='28' height='22' rx='6' fill='#1f2937' stroke='#374151' stroke-width='2'/>
      <circle cx='26' cy='31' r='4' fill='#93c5fd'/>
      <circle cx='38' cy='31' r='4' fill='#93c5fd'/>
      <rect x='28' y='42' width='8' height='6' rx='3' fill='#4b5563'/>
      <rect x='30' y='12' width='4' height='8' rx='2' fill='#6b7280'/>
    </svg>
    """).strip()
)

# ---------- Helpers ----------
def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

# ---------- Feedback Functions ----------
def collect_feedback(message_id, response_text):
    """Add feedback buttons after each bot response"""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"up_{message_id}", help="Good response"):
            store_feedback(message_id, "positive", response_text)
            st.success("Thanks!")
    
    with col2:
        if st.button("👎", key=f"down_{message_id}", help="Poor response"):
            store_feedback(message_id, "negative", response_text)
            st.warning("Thanks for the feedback")

def store_feedback(message_id, rating, response_text, detail_text=""):
    """Store feedback data"""
    feedback = {
        'message_id': message_id,
        'rating': rating,
        'response_text': response_text[:100] + "..." if len(response_text) > 100 else response_text,
        'username': st.session_state.get('username', 'unknown'),
        'model': st.session_state.get('model', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'session_id': id(st.session_state)  # Simple session identifier
    }
    
    st.session_state.feedback_data.append(feedback)
    
    # Also save to Google Sheets alongside your existing chat data
    try:
        feedback_row = [
            feedback['message_id'],
            feedback['rating'], 
            feedback['response_text'],
            feedback['username'],
            feedback['model'],
            feedback['timestamp']
        ]
        # You might want a separate sheet for feedback or add to existing
        sheet.append_row(feedback_row)
    except Exception as e:
        st.session_state.errors.append({"when": now_ts(), "msg": f"Feedback save error: {e}"})

# ---------- Login dialog ----------
@st.dialog("Login", dismissible=False, width="small")
def get_user_details():
    username = st.text_input("Username")
    email = st.text_input("Email")
    submit = st.button("Submit", type="primary")
    if submit:
        if not username or not email:
            st.warning("Please enter both a username and an email.")
            return
        st.session_state.user_info = {"username": username, "email": email}
        # also mirror to top-level keys for convenience
        st.session_state["username"] = username
        st.session_state["email"] = email
        st.rerun()

if "user_info" not in st.session_state:
    get_user_details()
    st.stop()  # <- block the rest of the script until we have creds

# ---------- Now that we have user_info, continue ----------
username = st.session_state["user_info"]["username"]
email = st.session_state["user_info"]["email"]

# ---------- Databasing (quick and sloppy) ----------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
gs_client = gspread.authorize(creds)
sheet = gs_client.open(st.secrets["SHEET_NAME"]).sheet1

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, my name is TeamMait. Feel free to ask me any questions to ask me about the referenced session transcript? It can be found in the left side panel. When you're done, please make sure to save the chat!",
            "ts": now_ts(),
            "display_name": "TeamMait",
        }
    ]
if "errors" not in st.session_state:
    st.session_state.errors = []
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.markdown(f"**Username:** {username}")
    st.markdown(f"**Email:** {email}")

    with st.expander("Settings", expanded=False):
        stream_on = st.checkbox("Stream responses", value=True)
        show_timestamps = st.checkbox("Display timestamps", value=True)

    model = r"gpt-4o-mini"

    st.session_state['stream_on'] = stream_on
    st.session_state['show_timestamps'] = show_timestamps
    st.session_state['model'] = model

    # User notes
    with st.expander("User Notes", expanded=True):
        user_notes = st.text_area("Enter any feedback you have for the TeamMait here:", height=180)

    # Feedback analytics
    if st.session_state.feedback_data:
        with st.expander("Feedback Summary", expanded=False):
            total_feedback = len(st.session_state.feedback_data)
            positive_feedback = len([f for f in st.session_state.feedback_data if f['rating'] == 'positive'])
            if total_feedback > 0:
                positive_rate = (positive_feedback / total_feedback) * 100
                st.metric("Positive Feedback Rate", f"{positive_rate:.1f}%")
                st.caption(f"Total feedback: {total_feedback}")

    # Exporting
    with st.expander( "Save Data", expanded=True):
        session_name = "tbd_session_name-" + datetime.now().strftime("%Y%m%d")

        metadata = {
            "app_name": "TeamMait Open-Ended Chat",
            "session_name": session_name,
            "username": username,
            "model": model,
            "message_count": len(st.session_state.messages),
            "user_notes": user_notes,
            "exported_at": datetime.now().isoformat(),
        }

        export_data = {
            "metadata": metadata,
            "messages": st.session_state.messages,
            "feedback": st.session_state.feedback_data,
            "errors": st.session_state.errors,
        }
        json_data = json.dumps(export_data, indent=2)
        if st.download_button(
            label="Export chat",
            data=json_data,
            file_name=f"{session_name}.json", 
            mime="application/json",
            type="primary",
        ):
            # Prepare export data
            messages = st.session_state.messages
            timestamp = datetime.now().isoformat()
            # Save to Google Sheets
            sheet.append_row([json.dumps(messages), timestamp])

    # ---------- Chroma initialization (after login) ----------
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)

    chroma_client = chromadb.PersistentClient(
        path="./rag_store",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = chroma_client.get_or_create_collection("therapy", embedding_function=embedding_fn)

    # ---------- Reference Conversation (expander) ----------
    @st.cache_resource
    def load_rag_documents():
        doc_folder = "doc/RAG"
        supporting_folder = os.path.join(doc_folder, "supporting_documents")
        documents = []
        ids = []
        # Load main reference conversation
        ref_path = os.path.join(doc_folder, "116_P8_conversation.json")
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "full_conversation" in data:
                    for i, turn in enumerate(data["full_conversation"]):
                        documents.append(str(turn))
                        ids.append(f"ref_{i}")
        # Load .txt and .json files from supporting_documents
        for txt_path in glob.glob(os.path.join(supporting_folder, "*.txt")):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                ids.append(f"supp_txt_{os.path.basename(txt_path)}")
        for json_path in glob.glob(os.path.join(supporting_folder, "*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        documents.append(str(item))
                        ids.append(f"supp_json_{os.path.basename(json_path)}_{i}")
                elif isinstance(data, dict):
                    for k, v in data.items():
                        documents.append(f"{k}: {v}")
                        ids.append(f"supp_json_{os.path.basename(json_path)}_{k}")
        # Seed collection if empty
        if collection.count() == 0 and documents:
            collection.add(documents=documents, ids=ids)
        return documents

    rag_documents = load_rag_documents()
    # Show only the reference conversation in the expander
    ref_conversation = []
    ref_path = os.path.join("doc/RAG", "116_P8_conversation.json")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "full_conversation" in data:
                ref_conversation = data["full_conversation"]
            elif isinstance(data, list):
                ref_conversation = data
    with st.expander("Show Referenced Full Conversation", expanded=True):
        if ref_conversation:
            for i, turn in enumerate(ref_conversation):
                is_client = turn.strip().startswith("Client: ")
                if is_client:
                    # Right-justify client's messages with custom CSS
                    st.markdown(f"""
                    <div style="text-align: right; margin-left: 0%; padding: 10px; border-radius: 10px;">
                    {turn}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Italicize therapist's messages
                    st.markdown(f"<div style='font-weight:600; font-size:1.08em; margin: 10px 0;'><em>{turn}</em></div>", unsafe_allow_html=True)
        else:
            st.info("No reference conversation found in 116_P8_conversation.json.")

# ---------- Layout CSS ----------
st.markdown(
    """
    <style>
      .app-container { max-width: 700px; margin: 0 auto; padding-top: 0px; }
      .chat-wrapper { min-height: 0px; padding: 0 4px; margin-top: 0px; }
      .msg.user { display: flex; justify-content: flex-end; }
      .msg.user .stChatMessage {
            flex-direction: row-reverse;
            display: flex;
            align-items: flex-start;
            margin-right: 0;
            margin-left: auto;
      }
      .stChatMessage {padding-top: 0rem; padding-bottom: 0rem;}
      .stChatInputContainer {position: sticky; bottom: 0; background: var(--background-color);}
      button[data-testid="stDownloadButton"] {
        background-color: #4F8A8B;
        color: red;
        border-radius: 6px;
        border: none;
      }
      button[data-testid="stDownloadButton"]:hover {
        background-color: #306B6B;
      }
      .feedback-container {
        margin-top: 0.5rem;
        display: flex;
        gap: 0.5rem;
      }
      .feedback-btn {
        background: none;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        cursor: pointer;
        font-size: 0.9rem;
      }
      .feedback-btn:hover {
        background-color: #f0f0f0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.title("TeamMait Private Conversation")

# ---------- Prompt builder ----------
def build_system_prompt() -> str:
    return (
        "You are TeamMait, a peer support assistant to a human clinician who is an expert mental health professsional. "
        "You are designed for calm, precise dialogue. "
        "Keep the discussion tied to the review, assessment, and or evaluation of the skills demonstrated by the therapist transcribed in the referenced conversation. If questions are asked broadly about mental health subjects, then provide the briefest answer as possible (if it is adjacently relevant) before politely and gently refocus the conversation to remain on-topic by asking if they have other questions related to the conversation or the therapist's performance."
        "Adopt an academically neutral tone; do not use emojis. "
        "Prioritize clinical utility: fidelity cues, effective/ineffective moves, missed opportunities, and risk signals. "
        "Anchor claims to transcript content and any supporting document(s); if no citation exists, say so briefly. "
        "Never invent facts; if uncertain, state the uncertainty briefly. "
        "When clarification is essential, ask for a single, decision-relevant question at the end. "
        "Be as succinct as possible in your responses without sacrificing accuracy. "
        "Avoid any language that validates the user unnecessarily, minimizes disagreement, or nudges the user to continue interacting. Prioritize accuracy, neutrality, and brevity over engagement, sychophancy, flattery, or rapport"
    )

# ---------- Provider clients ----------
def get_secret_then_env(name: str) -> str:
    val = None
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if not val:
        val = os.getenv(name)
    return val or ""

def get_anthropic_client():
    pass

def get_openai_client():
    if OpenAI is None:
        st.error("openai package not installed. Run: pip install openai")
        return None
    key = get_secret_then_env("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY. Set it in .streamlit/secrets.toml or as an environment variable.")
        return None
    return OpenAI(api_key=key)

def to_anthropic_messages(history):
    converted = []
    for m in history:
        role = m.get("role")
        if role in ("user", "teammait"):
            converted.append({"role": role, "content": [{"type": "text", "text": m["content"]}]})
    return converted

def claude_complete(history, system_text, model_name, stream=False):
    client = get_anthropic_client()
    if client is None:
        return "" if not stream else iter(())
    MAX_TURNS = 40
    trimmed = history[-MAX_TURNS:]
    if stream:
        try:
            with client.messages.stream(
                model=model_name,
                system=system_text,
                max_tokens=1024,
                messages=to_anthropic_messages(trimmed),
            ) as events:
                for text in events.text_stream:
                    yield text
        except Exception as e:
            yield f"\n[Error: {e}]"
    else:
        try:
            resp = client.messages.create(
                model=model_name,
                system=system_text,
                max_tokens=1024,
                messages=to_anthropic_messages(trimmed),
            )
            out = []
            for block in resp.content:
                if block.type == "text":
                    out.append(block.text)
            return "".join(out).strip()
        except Exception as e:
            return f"[Error: {e}]"

def openai_complete(history, system_text, model_name, stream=False, max_tokens=512):
    client = get_openai_client()
    if client is None:
        return "" if not stream else iter(())
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for m in history:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    if stream:
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=True, max_tokens=max_tokens, temperature=0.3)
            for chunk in resp:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"
    else:
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, max_tokens=max_tokens, temperature=0.3)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[Error: {e}]"


# ---------- Chat history (scrollable) ----------
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

for i, m in enumerate(st.session_state.messages):
    role = m["role"]
    avatar = BOT_SVG if role == "assistant" else DM_SVG
    css_class = "assistant" if role == "assistant" else "user"
    st.markdown(f"<div class='msg {css_class}'>", unsafe_allow_html=True)
    name_for_ui = m.get("display_name", role)
    with st.chat_message(name=name_for_ui, avatar=avatar):
        if st.session_state.get("show_timestamps", False) and "ts" in m:
            st.caption(m["ts"])
        st.markdown(m["content"])
        
        # Add feedback buttons only for assistant messages
        if role == "assistant" and m.get("content", "").strip():
            message_id = f"msg_{i}_{hash(m.get('content', ''))}"
            collect_feedback(message_id, m["content"])
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close chat-wrapper
st.markdown("</div>", unsafe_allow_html=True)  # close app-container


# ---------- Input ----------

prompt = st.chat_input("Talk to your TeamMait here...")
if prompt is not None and prompt.strip() != "":
    user_msg = {"role": "user", "content": prompt, "ts": now_ts(), "display_name": username}
    st.session_state.messages.append(user_msg)

    with st.chat_message(name=username, avatar=DM_SVG):
        if st.session_state.get("show_timestamps", False):
            st.caption(user_msg["ts"])
        st.markdown(prompt)

    # ---------- Retrieve context from vector DB ----------
    results = collection.query(query_texts=[prompt], n_results=3)
    # results["documents"] is a list of lists
    context_parts = []
    for docs in results.get("documents", []):
        context_parts.extend(docs)

    # Always include metadata and supporting documents
    # (rag_documents contains all seeded docs, including metadata/supporting)
    for doc in rag_documents:
        if doc not in context_parts:
            context_parts.append(doc)

    context = " ".join(context_parts)

    # --- Conditional evidence display ---
    show_evidence = any(kw in prompt.lower() for kw in ["evidence", "quote", "source", "show your work"])
    if show_evidence:
        evidence_text = "**Evidence (from transcript) used for this answer:**\n\n"
        for i, evidence in enumerate(context_parts, 1):
            evidence_text += f"> {evidence}\n\n"

        # Show evidence in UI
        st.markdown(evidence_text)

        # Persist evidence in chat history
        st.session_state.messages.append({
            "role": "evidence",
            "content": evidence_text,
            "ts": now_ts(),
            "display_name": "Evidence"
        })

    system_prompt = build_system_prompt() + f"""

    Use the following session context when answering:

    {context}
    """

    # Route provider by model prefix
    use_openai = st.session_state["model"].startswith("gpt-")
    complete_fn = openai_complete if use_openai else claude_complete

    with st.chat_message(name="TeamMait", avatar=BOT_SVG):
        if st.session_state["stream_on"]:
            placeholder = st.empty()
            acc = ""
            for chunk in complete_fn(
                history=st.session_state.messages,
                system_text=system_prompt,
                model_name=st.session_state["model"],
                stream=True,
            ):
                acc += chunk
                placeholder.markdown(acc)
            reply_text = acc.strip()
        else: # Stream off
            reply_text = complete_fn(
                history=st.session_state.messages,
                system_text=system_prompt,
                model_name=st.session_state["model"],
                stream=False,
            )
            st.markdown(reply_text or "")

    if reply_text and reply_text.startswith("[Error:"):
        st.session_state.errors.append({"when": now_ts(), "model": st.session_state["model"], "msg": reply_text})

    teammait_msg = {
        "role": "assistant",
        "content": reply_text,
        "ts": now_ts(),
        "display_name": "TeamMait",
    }
    st.session_state.messages.append(teammait_msg)