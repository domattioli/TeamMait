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
    __import__("pysqlite3")
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

# # ---- Brevity policy & templates (1–5) ----
# BREVITY = {
#     1: dict(name="Detailed Narrative",  max_tokens=900,  bullets=6,  sections=["Strengths","Areas for Growth","Opportunities","Risks/Concerns"], headline=False),
#     2: dict(name="Structured Summary",  max_tokens=650,  bullets=5,  sections=["Strengths","Areas for Growth","Concerns"],                    headline=False),
#     3: dict(name="Balanced Highlights", max_tokens=450,  bullets=4,  sections=None,                                                             headline=False),
#     4: dict(name="Key Points Only",     max_tokens=250,  bullets=3,  sections=None,                                                             headline=False),
#     5: dict(name="Headline Only",       max_tokens=120,  bullets=0,  sections=None,                                                             headline=True),
# }

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.markdown(f"**Username:** {username}")
    st.markdown(f"**Email:** {email}")
    # st.button("Clear chat", type="secondary", on_click=lambda: st.session_state.update(messages=[]))

    with st.expander("Settings", expanded=False):
        # empathy = st.slider("Empathy", 0, 100, 50, 5)
        # brevity = st.slider("Brevity", 1, 5, 3, 1)
        stream_on = st.checkbox("Stream responses", value=True)
        show_timestamps = st.checkbox("Display timestamps", value=True)

    # model = st.selectbox(
    #     "model",
    #     [
    #         "gpt-4o-mini",
    #         "claude-3-5-sonnet-20240620",
    #         "claude-3-5-haiku-20241022",
    #         "claude-3-opus-20240229",
    #     ],
    #     index=0,
    # )
    model = r"gpt-4o-mini"

    # st.session_state['empathy'] = empathy
    # st.session_state['brevity'] = brevity
    st.session_state['stream_on'] = stream_on
    st.session_state['show_timestamps'] = show_timestamps
    st.session_state['model'] = model

    # Meta notes
    # st.markdown("#### Meta Notes")
    # user_notes = st.text_area("Enter your commentary on TeamMait here:", height=180)
    with st.expander("User Notes", expanded=True):
        user_notes = st.text_area("Enter any feedback you have for the TeamMait here:", height=180)

    # Exporting
    # st.caption("Export data as a .json file")
    # session_name = "tbd_session_name-" + datetime.now().strftime("%Y%m%d")
    with st.expander( "Save Data", expanded=True):
        session_name = "tbd_session_name-" + datetime.now().strftime("%Y%m%d")

        metadata = {
            "app_name": "TeamMait Open-Ended Chat",
            "session_name": session_name,
            "username": username,
            "model": model,
            # "empathy": empathy,
            # "brevity": brevity,
            "message_count": len(st.session_state.messages),
            "user_notes": user_notes,
            "exported_at": datetime.now().isoformat(),
        }

        export_data = {
            "metadata": metadata,
            "messages": st.session_state.messages,
            "errors": st.session_state.errors,
        }
        json_data = json.dumps(export_data, indent=2)
        if st.download_button(
            label="Export chat as .json",
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

    # st.divider()
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
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        documents.append(str(item))
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.title("TeamMait Private Conversation")


# # ---------- Load JSON Conversation into Vector Store ----------
# @st.cache_resource
# def load_conversation_and_seed():
#     with open("116_P8_conversation.json") as f:
#         data = json.load(f)
#     if collection.count() == 0:
#         for i, turn in enumerate(data.get("full_conversation", [])):
#             collection.add(documents=[turn], ids=[f"conv_{i}"])
#     return data

# data = load_conversation_and_seed()

# # ---------- Reference Conversation (expander) ----------
# conversation = data.get("full_conversation", [])
# with st.expander("Show Referenced Full Conversation", expanded=False):
#     for turn in conversation:
#         st.markdown(turn)

# ---------- Prompt builder ----------
# def structure_prompt(level:int) -> str:
#     cfg = BREVITY[level]
#     if cfg["headline"]:
#         return (
#             "Output ONE sentence: the single most important clinical takeaway. "
#             "No preamble, no bullets, no quotes."
#         )
#     if cfg["sections"]:
#         # Sectioned bullets for Levels 1–2
#         per = max(1, cfg["bullets"] // len(cfg["sections"]) + 1)
#         sec_lines = "\n".join(f"- {s}: ≤{per} bullets" for s in cfg["sections"])
#         return (
#             "Use sectioned bullets. Each bullet: one actionable point anchored to the transcript; "
#             "start with a strong verb; optional short quote in quotes; ≤25 words per bullet.\n"
#             f"Sections:\n{sec_lines}\n"
#             "No filler, no concluding paragraph."
#         )
#     # Flat bullets for Levels 3–4
#     return (
#         f"Return exactly {cfg['bullets']} bullets. Each bullet ≤25 words, starts with a verb, "
#         "optional (timestamp). No intro or outro."
#     )

# def build_system_prompt(empathy_value: int, brevity_level: int) -> str:
def build_system_prompt() -> str:
    return (
        "You are TeamMait, a peer support assistant to a human clinician who is an expert mental health professsional. "
        "You are designed for calm, precise dialogue. "
        "Adopt an academically neutral tone; do not use emojis. "
        # f"The user has specified an empathy target of {empathy_value} out of maximum of 100, with 0 being not empathetic at all (completely stoic) and 100 being the most possible empathy you are capable of without being sychphantic."
        # f"Brevity level: {brevity_level}/5. "
        "Prioritize clinical utility: fidelity cues, effective/ineffective moves, missed opportunities, and risk signals. "
        "Anchor claims to transcript content; if no citation exists, say so briefly. "
        "Never invent facts; if uncertain, state the uncertainty briefly. "
        "When clarification is essential, ask for a single, decision-relevant question at the end. "
        # + structure_prompt(brevity_level)
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
    if anthropic is None:
        st.error("anthropic package not installed. Run: pip install anthropic")
        return None
    key = get_secret_then_env("ANTHROPIC_API_KEY")
    if not key:
        st.error("Missing ANTHROPIC_API_KEY. Set it in .streamlit/secrets.toml or as an environment variable.")
        return None
    return anthropic.Anthropic(api_key=key)

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

for m in st.session_state.messages:
    role = m["role"]
    avatar = BOT_SVG if role == "assistant" else DM_SVG
    css_class = "assistant" if role == "assistant" else "user"
    st.markdown(f"<div class='msg {css_class}'>", unsafe_allow_html=True)
    name_for_ui = m.get("display_name", role)
    with st.chat_message(name=name_for_ui, avatar=avatar):
        if st.session_state.get("show_timestamps", False) and "ts" in m:
            st.caption(m["ts"])
        st.markdown(m["content"])
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
    context = " ".join(context_parts)

    # --- Conditional evidence display ---
    show_evidence = any(kw in prompt.lower() for kw in ["evidence", "quote", "source", "show your work"])
    if show_evidence:
        st.markdown("**Evidence (from transcript) used for this answer:**")
        for i, evidence in enumerate(context_parts, 1):
            st.markdown(f"> {evidence}")

    # brev_level = st.session_state["brevity"]
    # cfg = BREVITY[brev_level]
    # system_prompt = build_system_prompt(st.session_state["empathy"], brev_level) + f"""
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
                # max_tokens=cfg["max_tokens"],
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
                # max_tokens=cfg["max_tokens"],
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
