import streamlit as st
import sys
import os
from datetime import datetime
import json
import glob

# Fix the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# SQLite shim for Chroma
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Setup ----------
st.set_page_config(page_title="Guided Interaction", page_icon="", layout="wide")

# Check login
if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.switch_page("Home.py")
    st.stop()

username = st.session_state["user_info"]["username"]

# ---------- Helper Functions ----------
def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def get_secret_then_env(name: str) -> str:
    """Get secret from Streamlit secrets or environment variable."""
    val = None
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if not val:
        val = os.getenv(name)
    return val or ""

def get_openai_client():
    """Initialize OpenAI client."""
    if OpenAI is None:
        st.error("openai package not installed. Run: pip install openai")
        return None
    key = get_secret_then_env("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY.")
        return None
    return OpenAI(api_key=key)

def build_system_prompt(mode: str = "guided") -> str:
    """Build system prompt for guided mode."""
    base_prompt = (
        "You are TeamMait, a peer-support assistant for expert clinicians reviewing "
        "therapist performance in a transcript. Your scope is limited strictly to "
        "analyzing the therapist's observable skills in the transcript. "
        "Anchor every claim to the transcript (and provided docs). If uncertain, say so briefly. "
        "Be succinct and academically neutral; do not use emojis. "
        "Never invent facts. Cite transcript line references; if no citation exists, say so. "
        "You cannot offer any visual or audio support -- only text responses."
    )
    
    if mode == "guided":
        return base_prompt + (
            "\n\nYou are operating in GUIDED REVIEW mode. The user is responding to structured prompts "
            "about specific clinical domains (Adherence, Procedural, Relational, Structural). "
            "Your role is to: "
            "1) Acknowledge their response type (Accept/Correct/Clarify/Disregard) "
            "2) If they Correct or Clarify, engage briefly with their clinical observation "
            "3) Ask if they want to discuss anything else about this domain before moving on "
            "4) Keep responses concise and focused on the specific prompt context."
        )
    
    return base_prompt

def openai_complete(history, system_text, model_name="gpt-4o-mini", stream=False, max_tokens=512):
    """Complete a chat using OpenAI API."""
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
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.3
            )
            for chunk in resp:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"
    else:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[Error: {e}]"

# ---------- RAG Setup ----------
@st.cache_resource
def initialize_chroma():
    """Initialize ChromaDB client and collection."""
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    )
    
    chroma_client = chromadb.PersistentClient(
        path="./rag_store",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    
    collection = chroma_client.get_or_create_collection(
        "therapy",
        embedding_function=embedding_fn
    )
    
    return chroma_client, collection

@st.cache_resource
def load_rag_documents():
    """Load all RAG documents and seed ChromaDB collection."""
    _, collection = initialize_chroma()
    
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

def retrieve_context(query: str, n_results: int = 5) -> tuple:
    """Retrieve relevant context from ChromaDB for a given query."""
    _, collection = initialize_chroma()
    results = collection.query(query_texts=[query], n_results=n_results)
    
    retrieved_parts = []
    for docs in results.get("documents", []):
        retrieved_parts.extend(docs)
    
    # Get all documents for comprehensive context
    all_documents = load_rag_documents()
    context_parts = list(retrieved_parts)
    for doc in all_documents:
        if doc not in context_parts:
            context_parts.append(doc)
    
    context = " ".join(context_parts)
    return context, retrieved_parts

def load_reference_conversation():
    """Load the reference conversation for display."""
    ref_path = os.path.join("doc/RAG", "116_P8_conversation.json")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "full_conversation" in data:
                return data["full_conversation"]
            elif isinstance(data, list):
                return data
    return []

# ---------- Flowchart Logic ----------
def load_question_bank():
    """Load the question bank from interaction_prompts.json."""
    with open("doc/interaction_prompts/interaction_prompts.json", "r") as f:
        data = json.load(f)
    return data.get("feedback_items", [])

def initialize_flowchart_state():
    """Initialize the flowchart session state."""
    if "flowchart_state" not in st.session_state:
        st.session_state.flowchart_state = {
            "stage": "intro",
            "questions_asked": [],
            "current_question": None,
            "current_response_type": None,
            "needs_followup": False,
            "all_domains_covered": False
        }

def get_next_question():
    """Select a random question from the bank that hasn't been asked yet."""
    import random
    question_bank = load_question_bank()
    asked = st.session_state.flowchart_state["questions_asked"]
    
    available = [q for q in question_bank if q["id"] not in asked]
    
    if not available:
        return None
    
    selected = random.choice(available)
    st.session_state.flowchart_state["questions_asked"].append(selected["id"])
    st.session_state.flowchart_state["current_question"] = selected
    
    return selected

def format_prompt_message(question: dict) -> str:
    """Format a question into a prompt message."""
    return (
        f"**{question['label']}**\n\n"
        f"**Assertion:** {question['assertion']}\n\n"
        f"**Explanation:** {question['explanation']}\n\n"
        f"**Invitation:** {question['invitation']}"
    )

def detect_response_type(user_input: str) -> str:
    """Detect the response type from user input."""
    user_lower = user_input.lower()
    
    accept_keywords = ["accept", "agree", "yes", "correct", "right", "accurate"]
    correct_keywords = ["correct", "actually", "disagree", "no", "wrong", "instead"]
    clarify_keywords = ["clarify", "explain", "elaborate", "more", "unclear", "expand"]
    disregard_keywords = ["disregard", "skip", "pass", "not relevant", "move on"]
    
    if any(kw in user_lower for kw in disregard_keywords):
        return "disregard"
    elif any(kw in user_lower for kw in accept_keywords):
        return "accept"
    elif any(kw in user_lower for kw in correct_keywords):
        return "correct"
    elif any(kw in user_lower for kw in clarify_keywords):
        return "clarify"
    else:
        return "unclear"

def handle_flowchart_transition(user_input: str) -> dict:
    """Handle flowchart state transitions based on user input."""
    state = st.session_state.flowchart_state
    stage = state["stage"]
    
    if stage == "intro":
        question = get_next_question()
        if question:
            return {
                "next_stage": "prompt",
                "bot_response": format_prompt_message(question),
                "show_buttons": True
            }
        else:
            return {
                "next_stage": "complete",
                "bot_response": "All questions have been reviewed. Session complete!",
                "show_buttons": False
            }
    
    elif stage == "prompt":
        response_type = detect_response_type(user_input)
        state["current_response_type"] = response_type
        
        if response_type == "disregard":
            return {
                "next_stage": "anything_else",
                "bot_response": "Noted. Anything else you would like to discuss about this topic before we move on?",
                "show_buttons": False
            }
        elif response_type in ["accept", "correct", "clarify"]:
            if len(user_input.split()) <= 2:
                return {
                    "next_stage": "anything_else",
                    "bot_response": f"You chose to {response_type}. Anything else you would like to discuss about this topic?",
                    "show_buttons": False
                }
            else:
                state["needs_followup"] = True
                return {
                    "next_stage": "active_engagement",
                    "bot_response": None,
                    "show_buttons": False,
                    "use_llm": True
                }
        else:
            return {
                "next_stage": "prompt",
                "bot_response": "I didn't catch that. Please respond with Accept, Correct, Clarify, or Disregard.",
                "show_buttons": True
            }
    
    elif stage == "active_engagement":
        return {
            "next_stage": "anything_else",
            "bot_response": "Does this response indicate active engagement that needs follow-up?",
            "show_buttons": False
        }
    
    elif stage == "anything_else":
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["no", "nope", "nothing", "move on", "next"]):
            if len(state["questions_asked"]) >= 4:
                state["all_domains_covered"] = True
                return {
                    "next_stage": "complete",
                    "bot_response": "All questions have been reviewed. Thank you for your participation!",
                    "show_buttons": False
                }
            else:
                question = get_next_question()
                if question:
                    return {
                        "next_stage": "prompt",
                        "bot_response": format_prompt_message(question),
                        "show_buttons": True
                    }
                else:
                    return {
                        "next_stage": "complete",
                        "bot_response": "All questions reviewed. Session complete!",
                        "show_buttons": False
                    }
        else:
            return {
                "next_stage": "anything_else",
                "bot_response": None,
                "show_buttons": False,
                "use_llm": True
            }
    
    return {
        "next_stage": stage,
        "bot_response": "I'm not sure how to respond. Please try again.",
        "show_buttons": False
    }

# ---------- Session State ----------
initialize_flowchart_state()

if "guided_messages" not in st.session_state:
    st.session_state.guided_messages = [
        {
            "role": "assistant",
            "content": "Hi, my name is TeamMait. Feel free to ask me any questions to ask me about the referenced session transcript. It can be found in the left side panel. If you'd like, I've made a few observations that we can discuss together.\n\nReady to begin?",
            "ts": now_ts()
        }
    ]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"**Username:** {username}")
    
    # Initialize completion status if it doesn't exist
    if "completion_status" not in st.session_state:
        st.session_state["completion_status"] = {}
    
    # Sync the checkbox state with the persistent completion tracker
    persistent_value = st.session_state["completion_status"].get("guided_interaction", False)
    st.session_state["include_guided_interaction"] = persistent_value

    def _on_include_guided_change():
        from utils.streamlit_compat import debug_trace
        # Update the persistent completion tracker when checkbox changes
        current_value = st.session_state.get("include_guided_interaction", False)
        st.session_state["completion_status"]["guided_interaction"] = current_value
        debug_trace("completion_status.guided_interaction", current_value, "Guided Interaction")

    st.checkbox("Check this when done", key="include_guided_interaction", on_change=_on_include_guided_change)
    
    # Progress tracker
    st.markdown("### Progress")
    questions_asked = len(st.session_state.flowchart_state["questions_asked"])
    st.progress(min(questions_asked / 4, 1.0))
    st.caption(f"{questions_asked} / 4 questions reviewed")

    # Copy settings from Open Chat
    with st.expander("Settings", expanded=False):
        stream_on = st.checkbox("Stream responses", value=True)
        show_timestamps = st.checkbox("Display timestamps", value=True)
        model = r"gpt-4o-mini"
        st.session_state['stream_on'] = stream_on
        st.session_state['show_timestamps'] = show_timestamps
        st.session_state['model'] = model
    # Show reference conversation (expanded by default)
    with st.expander("Show Reference Conversation", expanded=True):
        ref_conversation = load_reference_conversation()
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

# ---------- Main Content ----------
st.title("Guided Interaction")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Disclaimer: TeamMait may be incorrect or incomplete. Please verify information.</p>", unsafe_allow_html=True)

# Quick debug: show include flags so we can verify navigation preserves state
st.info(f"Flags: open={st.session_state.get('include_open_chat', False)}, survey={st.session_state.get('include_survey', False)}, guided={st.session_state.get('include_guided_interaction', False)}")

# Display chat history
for m in st.session_state.guided_messages:
    role = m["role"]
    with st.chat_message(role):
        if "ts" in m:
            st.caption(m["ts"])
        st.markdown(m["content"])

# ---------- Response Buttons ----------
def handle_button_click(response_type: str):
    """Handle quick response button clicks."""
    user_msg = {
        "role": "user",
        "content": response_type.capitalize(),
        "ts": now_ts()
    }
    st.session_state.guided_messages.append(user_msg)
    
    # Process the transition
    result = handle_flowchart_transition(response_type)
    st.session_state.flowchart_state["stage"] = result["next_stage"]
    
    if result.get("use_llm"):
        # Use LLM to generate response
        context, _ = retrieve_context(user_msg["content"])
        system_prompt = build_system_prompt(mode="guided") + f"\n\nContext:\n{context}"
        
        response = openai_complete(
            history=st.session_state.guided_messages,
            system_text=system_prompt,
            model_name="gpt-4o-mini",
            stream=False,
            max_tokens=512
        )
        
        bot_msg = {
            "role": "assistant",
            "content": response,
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(bot_msg)
        
        # Follow up with "anything else"
        followup_msg = {
            "role": "assistant",
            "content": "Anything else you would like to discuss about this topic before we move on?",
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(followup_msg)
    elif result["bot_response"]:
        bot_msg = {
            "role": "assistant",
            "content": result["bot_response"],
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(bot_msg)
    
    st.rerun()

# Show buttons only when in prompt stage
if st.session_state.flowchart_state["stage"] == "prompt":
    st.markdown("### Quick Response:")
    cols = st.columns(2)
    with cols[0]:
        st.button("👍", on_click=lambda: handle_button_click("accept"), use_container_width=True)
    with cols[1]:
        st.button("👎", on_click=lambda: handle_button_click("disregard"), use_container_width=True)

# ---------- Text Input ----------
if st.session_state.flowchart_state["stage"] != "complete":
    prompt = st.chat_input("Type your response or elaboration...")
    
    if prompt:
        user_msg = {
            "role": "user",
            "content": prompt,
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process transition
        result = handle_flowchart_transition(prompt)
        st.session_state.flowchart_state["stage"] = result["next_stage"]
        
        if result.get("use_llm"):
            # Generate LLM response
            context, _ = retrieve_context(prompt)
            system_prompt = build_system_prompt(mode="guided") + f"\n\nContext:\n{context}"
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""
                for chunk in openai_complete(
                    history=st.session_state.guided_messages,
                    system_text=system_prompt,
                    model_name="gpt-4o-mini",
                    stream=True,
                    max_tokens=512
                ):
                    acc += chunk
                    placeholder.markdown(acc)
                
                bot_msg = {
                    "role": "assistant",
                    "content": acc.strip(),
                    "ts": now_ts()
                }
                st.session_state.guided_messages.append(bot_msg)
        elif result["bot_response"]:
            with st.chat_message("assistant"):
                st.markdown(result["bot_response"])
                
            bot_msg = {
                "role": "assistant",
                "content": result["bot_response"],
                "ts": now_ts()
            }
            st.session_state.guided_messages.append(bot_msg)
        
        st.rerun()

else:
    st.success("Guided interaction session complete! You may export your data from the sidebar.")
    if st.button("Start New Session"):
        st.session_state.flowchart_state = {
            "stage": "intro",
            "questions_asked": [],
            "current_question": None,
            "current_response_type": None,
            "needs_followup": False,
            "all_domains_covered": False
        }
        st.session_state.guided_messages = [
            {
                "role": "assistant",
                "content": "Welcome back! Ready to review more observations?",
                "ts": now_ts()
            }
        ]
        st.rerun()