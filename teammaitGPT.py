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
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from chromadb.config import Settings



# ---------- Databasing (quick and sloppy) ----------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open(st.secrets["SHEET_NAME"]).sheet1


# Optional SDKs (load if installed)
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Page & Theme ----------
st.set_page_config(page_title="TeamMait Private Conversation", page_icon="💬", layout="wide")

@st.dialog("Login", dismissible=False, width="small" )
def get_user_details():
    username = st.text_input("Username")
    email = st.text_input("Email")
    if st.button("Submit"):
        st.session_state.user_info = {"username": username, "email": email}
        st.rerun()

if "user_info" not in st.session_state:
    get_user_details()


# Load embeddings
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)

# Setup Chroma vector DB
# client = chromadb.PersistentClient(path="./rag_store")
# Use DuckDB backend for ChromaDB
client = chromadb.PersistentClient(
    settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./rag_store"
    )
)
collection = client.get_or_create_collection("therapy", embedding_function=embedding_fn)

# ---------- Load JSON Conversation into Vector Store ----------
@st.cache_resource
def load_conversation():
    with open("116_P8_conversation.json") as f:
        data = json.load(f)

    # Store only once
    if collection.count() == 0:
        for i, turn in enumerate(data["full_conversation"]):
            collection.add(documents=[turn], ids=[f"conv_{i}"])
    return data

data = load_conversation()
    
# Init embeddings + Chroma (do this once, cache with st.cache_resource)
@st.cache_resource
def init_vectorstore():
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path="./rag_store")
    collection = client.get_or_create_collection("therapy", embedding_function=embedding_fn)

    # Load transcript only once
    if collection.count() == 0:
        with open("116_P8_conversation.json") as f:
            data = json.load(f)
        for i, turn in enumerate(data["full_conversation"]):
            collection.add(documents=[turn], ids=[f"conv_{i}"])
    return collection

collection = init_vectorstore()


# ---------- Simple avatars (SVG data URIs) ----------
def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)

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

# ---------- Session state ----------
def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, my name is TeamMait. Do you have any questions to ask me about Session N?",
            "ts": now_ts(),
            "display_name": "TeamMait",
        }
    ]
if "errors" not in st.session_state:
    st.session_state.errors = []

# ---------- Sidebar (settings) ----------
with st.sidebar:
    if "user_info" in st.session_state:
        st.markdown(f"**Username:** {st.session_state.user_info['username']}")
        st.markdown(f"**Email:** {st.session_state.user_info['email']}")
    
    st.button("Clear chat", type="secondary", on_click=lambda: st.session_state.update(messages=[]))
    st.divider()

    st.markdown("#### Settings")
    username = st.session_state.get("user_info", {}).get("username", "unknown")
    email = st.session_state.get("user_info", {}).get("email", "unknown")
    empathy = st.slider("Empathy", 0, 100, 50, 5)
    brevity = st.slider("Brevity", 1, 5, 4, 1)
    stream_on = st.checkbox("Stream responses", value=True)
    show_timestamps = st.checkbox("Always show timestamps", value=False)

    model = st.selectbox(
        "model",
        [
            "gpt-4o-mini",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        index=0,
    )
    # st.session_state['user_info'] = st.session_state.user_info
    st.session_state['empathy'] = empathy
    st.session_state['brevity'] = brevity
    st.session_state['stream_on'] = stream_on
    st.session_state['show_timestamps'] = show_timestamps
    st.session_state['model'] = model

    st.divider()

    # Exporting
    st.caption("Export data as a .json file")
    session_name = "tbd_session_name-" + datetime.now().strftime("%Y%m%d")
    metadata = {
        "app_name": "TeamMait Open-Ended Chat",
        "session_name": session_name,
        **{k: st.session_state[k] for k in ["username", "model", "empathy", "brevity"] if k in st.session_state},
        "message_count": len(st.session_state.messages),
        "exported_at": datetime.now().isoformat(),
    }
    export_data = {
        "metadata": metadata,
        "messages": st.session_state.messages,
        "errors": st.session_state.errors,
    }
    json_data = json.dumps(export_data, indent=2)
    if st.download_button(
            label="Export chat",
            data=json_data,
            file_name=f"{session_name}.json",
            mime="application/json",
        ):

        # Prepare export data
        messages = st.session_state.messages
        timestamp = datetime.now().isoformat()

        # Save to Google Sheets
        sheet.append_row([json.dumps(messages), timestamp])

# ---------- Layout CSS ----------
st.markdown(
    """
    <style>
      .app-container {
        max-width: 700px;
        margin: 0 auto;
        padding-top: 0px;
      }
      .chat-wrapper {
        min-height: 0px;
        padding: 0 4px;
        margin-top: 0px;
      }
      .msg.user { display: flex; justify-content: flex-end; }
      .msg.user .stChatMessage {
            flex-direction: row-reverse;
            display: flex;
            align-items: flex-start;
            margin-right: 0;
            margin-left: auto;
      }
      .stChatMessage {padding-top: 0.25rem; padding-bottom: 0.25rem;}
      .stChatInputContainer {position: sticky; bottom: 0; background: var(--background-color);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.title("TeamMait Private Conversation")

# ---------- Referenced Conversation (docked at top) ----------
# Load the conversation JSON
with open("116_P8_conversation.json", "r") as f:
    data = json.load(f)
conversation = data.get("full_conversation", [])

# Expander pinned below chat and input
with st.expander("Show Reference Full Conversation", expanded=False, ):
    for turn in conversation:
        st.markdown(turn)


# ---------- Provider key loaders (lazy; no top-level secrets access) ----------
def get_secret_then_env(name: str) -> str:
    # Try Streamlit secrets first, but catch when secrets.toml is absent
    val = None
    try:
        val = st.secrets.get(name)  # Streamlit's SecretsDict supports .get
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

# ---------- Prompt builder ----------
def build_system_prompt(empathy_value: int, brevity_level: int) -> str:
    # brevity_map = {
    #     1: "Use very short, bullet-like responses unless asked otherwise.",
    #     2: "Be concise and avoid tangents.",
    #     3: "Be succinct but include essential clarifications.",
    #     4: "Be moderately detailed while remaining concise.",
    #     5: "Provide fuller explanations when helpful.",
    # }
    return (
        "You are TeamMait, a peer support assistant to a human clinician who is an expert mental health professsional. "
        "You are designed for calm, precise dialogue. "
        "Adopt an academically neutral tone; do not use emojis. "
        f"The user has specified an empathy target of {empathy_value} out of maximum of 100, with 0 being not empathetic at all and 100 being the most possible empathy you are capable of without being sychphantic."
        f"The user has specified a brevity/concision level of {brevity_level} out of maximum of 5, with 1 being the lease concise and 5 being the most concise possible without omitting important details. "
        "When uncertain, ask for the single most decision-relevant clarification. "
        "Cite specific content fromthe referenced documents as much as possible. If no citation exists, say so."
        "Never talk about off-topic subjects even if asked. Only talk about the referenced documents."
    )

# ---------- Anthropic helpers ----------
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

# ---------- OpenAI helpers ----------
def openai_complete(history, system_text, model_name, stream=False):
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
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=True)
            for chunk in resp:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"
    else:
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
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
        if show_timestamps and "ts" in m:
            st.caption(m["ts"])
        st.markdown(m["content"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close chat-wrapper
st.markdown("</div>", unsafe_allow_html=True)  # close app-container

# ---------- Input ----------
prompt = st.chat_input("Talk to your TeamMait here...")
if prompt:
    # Store and render user message — canonical role for APIs, keep display name for UI
    user_msg = {"role": "user", "content": prompt, "ts": now_ts(), "display_name": username}
    st.session_state.messages.append(user_msg)

    with st.chat_message(name=username, avatar=DM_SVG):
        if show_timestamps:
            st.caption(user_msg["ts"])
        st.markdown(prompt)


    # ---------- NEW: Retrieve context from vector DB ----------
    results = collection.query(query_texts=[prompt], n_results=3)
    context = " ".join(results["documents"][0])

    # System prompt
    system_prompt = build_system_prompt(empathy, brevity) + f"""

    Use the following session context when answering:

    {context}
    """

    # Route provider by model prefix
    use_openai = model.startswith("gpt-")
    complete_fn = openai_complete if use_openai else claude_complete

    # Assistant turn
    with st.chat_message(name="TeamMait", avatar=BOT_SVG):
        if stream_on:
            placeholder = st.empty()
            acc = ""
            for chunk in complete_fn(
                history=st.session_state.messages,
                system_text=system_prompt,
                model_name=model,
                stream=True,
            ):
                acc += chunk
                placeholder.markdown(acc)
            reply_text = acc.strip()
        else:
            reply_text = complete_fn(
                history=st.session_state.messages,
                system_text=system_prompt,
                model_name=model,
                stream=False,
            )
            st.markdown(reply_text or "")

    if reply_text and reply_text.startswith("[Error:"):
        st.session_state.errors.append({"when": now_ts(), "model": model, "msg": reply_text})

    teammait_msg = {
        "role": "assistant",
        "content": reply_text,
        "ts": now_ts(),
        "display_name": "TeamMait",
    }
    st.session_state.messages.append(teammait_msg)