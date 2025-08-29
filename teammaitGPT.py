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

# ---------- SQLite shim for Chroma ----------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from openai import OpenAI


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

# ---------- Setup Chroma vector DB (new API) ----------
client = chromadb.PersistentClient(
    path="./rag_store",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
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
    client = chromadb.PersistentClient(
        path="./rag_store",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
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

# (rest of your file remains unchanged)
