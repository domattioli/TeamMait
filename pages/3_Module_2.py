"""
Module 2 - Production Version
Complete implementation with all issues fixed:
- Persistent session storage
- Server-side navigation validation
- Comprehensive error handling
- Robust API retry logic
- Typo-tolerant input parsing
- Analytics logging
- RAG error handling
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import glob
import uuid
import time
import logging
import warnings

# Suppress verbose logging from external libraries EARLY
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging - don't output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/guided_interaction.log"),
    ],
)
logger = logging.getLogger(__name__)

# Suppress verbose loggers
for logger_name in [
    "sentence_transformers",
    "transformers",
    "chromadb",
    "httpx",
    "huggingface",
    "urllib3",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Fix the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# NOTE: Real-time timer updates are NOT possible in Streamlit without user interaction.
# Streamlit only reruns when there's user input. Timer will update when user sends messages.
# This is a Streamlit limitation, not a bug. We've combined timer + time used into one display.

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

try:
    from docx import Document
except ImportError:
    Document = None

# Import utility modules
try:
    from utils.session_manager import SessionManager
    from utils.navigation_validator import NavigationValidator
    from utils.api_handler import OpenAIHandler, APIRetryableError, APIPermanentError
    from utils.input_parser import InputParser, MessageBuffer
    from utils.analytics import get_analytics
except ImportError as e:
    logger.error(f"Failed to import utilities: {e}")
    st.error("System initialization error. Please contact support.")
    st.stop()

# ==================== CONSTANTS ====================

SESSION_DURATION = timedelta(minutes=20)
OBSERVATION_START_TIME_KEY = "observation_start_time"
guided_interaction_conversation = "281_P10_conversation.json"

# ==================== HELPER FUNCTIONS ====================


def now_ts() -> str:
    """Get current timestamp."""
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
    """Initialize OpenAI client with proper error handling."""
    if OpenAI is None:
        logger.error("OpenAI package not installed")
        st.error("System configuration error: OpenAI package not installed.")
        st.stop()

    key = get_secret_then_env("OPENAI_API_KEY")
    if not key:
        logger.error("Missing OPENAI_API_KEY")
        st.error(
            "System configuration error: Missing OPENAI_API_KEY. "
            "Please contact your administrator."
        )
        st.stop()

    try:
        return OpenAI(api_key=key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        st.error("Failed to initialize AI system. Please try again.")
        st.stop()


def build_system_prompt() -> str:
    """Build system prompt for guided mode."""
    return (
        "You are TeamMait, a peer-support assistant for expert clinicians reviewing "
        "therapist performance in a transcript. Your scope is limited strictly to "
        "analyzing the therapist's observable skills in the transcript. "
        "Anchor every claim to the transcript. When citing the transcript, "
        "use the line numbers provided in the format [Line X]. If uncertain, say so briefly. "
        "Be succinct and academically neutral; do not use emojis. "
        "Never invent facts. Always cite specific line references when making claims about the transcript. "
        "Engage conversationally and provide thoughtful analysis. Avoid repetitive phrases. "
        "Provide substantive answers, not just questions."
    )


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text content from a DOCX file."""
    if Document is None:
        return "[Error: python-docx not installed]"

    try:
        doc = Document(docx_path)
        text_parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = [
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                ]
                if row_text:
                    text_parts.append(" | ".join(row_text))

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting DOCX: {e}")
        return f"[Error: {str(e)}]"


# ==================== RAG SETUP ====================


@st.cache_resource(show_spinner=False)
def initialize_chroma():
    """Initialize ChromaDB client and collection with error handling."""
    try:
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
            "therapy", embedding_function=embedding_fn
        )

        logger.info("ChromaDB initialized successfully")
        return chroma_client, collection
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_rag_documents():
    """Load all RAG documents with comprehensive error handling."""
    try:
        _, collection = initialize_chroma()
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return {"reference": False, "supporting": 0, "total": 0, "errors": [str(e)]}

    doc_folder = "doc/RAG"
    load_status = {
        "reference": False,
        "supporting": 0,
        "total": 0,
        "errors": [],
    }

    documents = []
    ids = []

    # ==================== REFERENCE CONVERSATION ====================
    
    ref_path = os.path.join(doc_folder, guided_interaction_conversation )

    if not os.path.exists(ref_path):
        error_msg = f"Reference conversation not found: {ref_path}"
        load_status["errors"].append(error_msg)
        logger.warning(error_msg)
    else:
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "full_conversation" in data:
                try:
                    for i, turn in enumerate(data["full_conversation"]):
                        line_num = i + 1
                        numbered_turn = f"[Line {line_num}] {turn}"
                        documents.append(numbered_turn)
                        ids.append(f"ref_{i}")

                    load_status["reference"] = True
                    logger.info(
                        f"Loaded {len(data['full_conversation'])} reference turns"
                    )
                except Exception as e:
                    error_msg = f"Error processing reference conversation: {e}"
                    load_status["errors"].append(error_msg)
                    logger.error(error_msg)
            else:
                error_msg = "Reference JSON missing 'full_conversation' key"
                load_status["errors"].append(error_msg)
                logger.warning(error_msg)

        except json.JSONDecodeError as e:
            error_msg = f"Reference conversation JSON malformed: {e}"
            load_status["errors"].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error reading reference conversation: {e}"
            load_status["errors"].append(error_msg)
            logger.error(error_msg)

    # ==================== SUPPORTING DOCUMENTS ====================

    supporting_folder = os.path.join(doc_folder, "supporting_documents")

    if not os.path.exists(supporting_folder):
        logger.warning(f"Supporting documents folder not found: {supporting_folder}")
    else:
        # Load TXT files
        for txt_path in glob.glob(os.path.join(supporting_folder, "*.txt")):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    ids.append(f"supp_txt_{os.path.basename(txt_path)}")
                    load_status["supporting"] += 1
            except Exception as e:
                logger.error(f"Error loading {txt_path}: {e}")

        # Load JSON files
        for json_path in glob.glob(os.path.join(supporting_folder, "*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            documents.append(str(item))
                            ids.append(
                                f"supp_json_{os.path.basename(json_path)}_{i}"
                            )
                        load_status["supporting"] += len(data)
                    else:
                        documents.append(str(data))
                        ids.append(f"supp_json_{os.path.basename(json_path)}")
                        load_status["supporting"] += 1
            except Exception as e:
                logger.error(f"Error loading {json_path}: {e}")

        # Load DOCX files
        for docx_path in glob.glob(os.path.join(supporting_folder, "*.docx")):
            try:
                content = extract_text_from_docx(docx_path)
                if content and not content.startswith("[Error"):
                    documents.append(content)
                    ids.append(f"supp_docx_{os.path.basename(docx_path)}")
                    load_status["supporting"] += 1
            except Exception as e:
                logger.error(f"Error loading {docx_path}: {e}")

    # ==================== SEED COLLECTION ====================

    load_status["total"] = len(documents)

    if documents:
        try:
            if collection.count() == 0:
                collection.add(documents=documents, ids=ids)
                logger.info(f"Seeded ChromaDB with {len(documents)} documents")
        except Exception as e:
            error_msg = f"Error seeding ChromaDB: {e}"
            load_status["errors"].append(error_msg)
            logger.error(error_msg)

    return load_status


def retrieve_context(query: str, n_results: int = 5) -> str:
    """Retrieve relevant context from ChromaDB."""
    try:
        _, collection = initialize_chroma()
        results = collection.query(query_texts=[query], n_results=n_results)

        retrieved_parts = []
        for docs in results.get("documents", []):
            retrieved_parts.extend(docs)

        return " ".join(retrieved_parts)
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""


def load_reference_conversation() -> List[str]:
    """Load the reference conversation for display."""
    ref_path = os.path.join("doc/RAG", guided_interaction_conversation )
    if os.path.exists(ref_path):
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "full_conversation" in data:
                    return data["full_conversation"]
        except Exception as e:
            logger.error(f"Error loading reference conversation: {e}")
    return []


def load_question_bank() -> Optional[List[Dict]]:
    """Load and validate the question bank."""
    question_path = "doc/interaction_prompts/interaction_prompts.json"

    # Check file exists
    if not os.path.exists(question_path):
        error_msg = f"Question bank file not found: {question_path}"
        logger.error(error_msg)
        st.error(f"{error_msg}")
        st.stop()

    # Parse JSON
    try:
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Question bank JSON is malformed: {e}"
        logger.error(error_msg)
        st.error(f"{error_msg}")
        st.stop()

    # Extract question items
    if not isinstance(data, dict):
        error_msg = "Question bank root must be a dict"
        logger.error(error_msg)
        st.error(f"{error_msg}")
        st.stop()

    questions = data.get("feedback_items", [])

    # Validate structure
    if not questions:
        error_msg = "Question bank is empty"
        logger.error(error_msg)
        st.error(f"{error_msg}")
        st.stop()

    if len(questions) < 4:
        error_msg = f"Question bank has {len(questions)} questions, need at least 4"
        logger.error(error_msg)
        st.error(f"{error_msg}")
        st.stop()

    # Validate each question
    required_fields = {"assertion", "explanation", "invitation"}
    for i, q in enumerate(questions[:4]):
        if not isinstance(q, dict):
            error_msg = f"Question {i + 1} is not a dict"
            logger.error(error_msg)
            st.error(f"{error_msg}")
            st.stop()

        missing_fields = required_fields - set(q.keys())
        if missing_fields:
            error_msg = f"Question {i + 1} missing fields: {missing_fields}"
            logger.error(error_msg)
            st.error(f"{error_msg}")
            st.stop()

        for field in required_fields:
            if not isinstance(q[field], str) or not q[field].strip():
                error_msg = f"Question {i + 1}: '{field}' is empty"
                logger.error(error_msg)
                st.error(f"{error_msg}")
                st.stop()

    logger.info(f"Successfully validated {len(questions[:4])} questions")
    return questions[:4]


# ==================== STREAMLIT APP ====================

st.set_page_config(
    page_title="Module 2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check login
if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.switch_page("Home.py")
    st.stop()

username = st.session_state["user_info"]["username"]
analytics = get_analytics()

# Initialize OpenAI client
client = get_openai_client()

# Initialize or resume session
if "guided_session_id" not in st.session_state:
    # New session
    st.session_state.guided_session_id = str(uuid.uuid4())
    session_metadata = SessionManager.create_session(
        username, st.session_state.guided_session_id
    )
    st.session_state.guided_session_start = None  # Don't start timer yet - wait for "start"
    logger.info(f"Created new session {st.session_state.guided_session_id}")
    analytics.session_started(username, st.session_state.guided_session_id)
else:
    # Resumed session - verify it still exists
    if not SessionManager.session_exists(
        username, st.session_state.guided_session_id
    ):
        st.warning("Your session has expired. Starting a new one.")
        st.session_state.guided_session_id = str(uuid.uuid4())
        session_metadata = SessionManager.create_session(
            username, st.session_state.guided_session_id
        )
        st.session_state.guided_session_start = None  # Don't start timer yet
        logger.info(
            f"Session expired, created new session {st.session_state.guided_session_id}"
        )

# Update session activity
SessionManager.update_session_activity(username, st.session_state.guided_session_id)

# Load saved conversations
if "all_conversations" not in st.session_state:
    saved_conversations = SessionManager.load_conversations(
        username, st.session_state.guided_session_id
    )
    st.session_state.all_conversations = {
        int(k): v for k, v in saved_conversations.items()
    }

if "guided_phase" not in st.session_state:
    st.session_state.guided_phase = "intro"

if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0

if "question_bank" not in st.session_state:
    st.session_state.question_bank = load_question_bank()

if "message_buffer" not in st.session_state:
    st.session_state.message_buffer = MessageBuffer()

# Load RAG documents
rag_status = load_rag_documents()
if "rag_load_status" not in st.session_state:
    st.session_state.rag_load_status = rag_status
    analytics.rag_load_status(
        username,
        st.session_state.guided_session_id,
        rag_status["reference"],
        rag_status["supporting"],
        rag_status["total"],
        rag_status["errors"],
    )

# ==================== STATE PERSISTENCE ====================
def sync_session_to_storage():
    """Sync current session state to persistent storage."""
    try:
        # Save conversations
        SessionManager.save_conversations(
            username,
            st.session_state.guided_session_id,
            st.session_state.all_conversations,
        )

        # Update metadata
        metadata = SessionManager.load_session_metadata(
            username, st.session_state.guided_session_id
        )
        if metadata:
            metadata.update(
                {
                    "phase": st.session_state.guided_phase,
                    "current_question_idx": st.session_state.current_question_idx,
                    "total_messages": sum(
                        len(conv)
                        for conv in st.session_state.all_conversations.values()
                    ),
                    "last_activity": datetime.now().isoformat(),
                }
            )
            SessionManager.save_session_metadata(
                username, st.session_state.guided_session_id, metadata
            )
    except Exception as e:
        logger.error(f"Error syncing session: {e}")
        # Calculate elapsed time safely
        if st.session_state.guided_session_start is not None:
            elapsed = (datetime.now() - st.session_state.guided_session_start).total_seconds()
        else:
            elapsed = 0
        analytics.error_occurred(
            username,
            st.session_state.guided_session_id,
            type(e).__name__,
            str(e),
            elapsed,
            {"context": "sync_session_to_storage"},
        )


# ==================== NAVIGATION HANDLER ====================
def handle_navigation(
    target_question_idx: int, target_phase: str, log_event: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Handle navigation with server-side validation.

    Returns:
        (success, error_message)
    """
    current_phase = st.session_state.guided_phase
    current_idx = st.session_state.current_question_idx

    # Validate
    is_valid, error_msg = NavigationValidator.validate_navigation(
        current_phase,
        current_idx,
        target_phase,
        target_question_idx,
        len(st.session_state.question_bank),
    )

    if not is_valid:
        return False, error_msg

    # Record timing with defensive check
    if st.session_state.guided_session_start is not None:
        elapsed = (datetime.now() - st.session_state.guided_session_start).total_seconds()
    else:
        elapsed = 0

    # Log event if moving observations
    if target_question_idx != current_idx and log_event:
        analytics.observation_advanced(
            username,
            st.session_state.guided_session_id,
            current_idx,
            target_question_idx,
            elapsed,
            len(st.session_state.all_conversations[current_idx]),
            elapsed,  # TODO: track per-observation timing
        )

    # Apply navigation
    st.session_state.current_question_idx = target_question_idx
    if target_phase != current_phase:
        analytics.phase_transition(
            username,
            st.session_state.guided_session_id,
            current_phase,
            target_phase,
            elapsed,
            {"to_question": target_question_idx},
        )
        st.session_state.guided_phase = target_phase

    sync_session_to_storage()
    return True, None


# ==================== CALCULATE TIME ====================

# Timer only starts after user says "start"
if st.session_state.guided_session_start is None:
    elapsed = timedelta(seconds=0)
    remaining = SESSION_DURATION
    time_expired = False
else:
    elapsed = datetime.now() - st.session_state.guided_session_start
    remaining = SESSION_DURATION - elapsed
    time_expired = remaining.total_seconds() <= 0

# Initialize warning states
if "time_warning_2min_shown" not in st.session_state:
    st.session_state.time_warning_2min_shown = False

# Auto-transition to expired if time is up and in active phase
if time_expired and st.session_state.guided_phase == "active":
    st.session_state.guided_phase = "expired"
    elapsed_seconds = elapsed.total_seconds()
    analytics.session_time_expired(
        username,
        st.session_state.guided_session_id,
        st.session_state.current_question_idx,
        st.session_state.current_question_idx,
        sum(len(c) for c in st.session_state.all_conversations.values()),
        elapsed_seconds,
    )
    sync_session_to_storage()

# 2-minute warning
if (remaining.total_seconds() <= 120 and 
    remaining.total_seconds() > 0 and 
    not st.session_state.time_warning_2min_shown and
    st.session_state.guided_phase in ("active", "review")):
    
    st.session_state.time_warning_2min_shown = True
    st.warning(
        "**2 MINUTES REMAINING!**\n\n"
        "Your session will end in 2 minutes. "
        "Consider wrapping up your current thoughts."
    )
    analytics.log_event("two_minute_warning", {
        "remaining_seconds": remaining.total_seconds(),
        "observation_idx": st.session_state.current_question_idx
    }) if hasattr(analytics, 'log_event') else None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown(f"**Username:** {username}")

    # Completion checkbox
    if "completion_status" not in st.session_state:
        st.session_state["completion_status"] = {}

    persistent_value = st.session_state["completion_status"].get(
        "guided_interaction", False
    )
    st.session_state["include_guided_interaction"] = persistent_value

    def _on_include_guided_change():
        current_value = st.session_state.get("include_guided_interaction", False)
        st.session_state["completion_status"]["guided_interaction"] = current_value

    st.checkbox(
        "Check this when done",
        key="include_guided_interaction",
        on_change=_on_include_guided_change,
    )
    
    # Reminder when time is running low or expired
    if time_expired or (remaining.total_seconds() < 300 and st.session_state.guided_phase in ("active", "review")):
        st.info(
            "**Reminder:** Don't forget to check the "
            "**'Check this when done'** checkbox above when you finish!"
        )

    # Combined Timer Display
    st.markdown("### Session Timer")
    
    # Only show timer if session has started
    if st.session_state.guided_session_start is None:
        st.info("**Timer starts after you tell TeamMait to 'start'**")
    else:
        # Create combined display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time Remaining:**")
            minutes = max(0, int(remaining.total_seconds() // 60))
            seconds = max(0, int(remaining.total_seconds() % 60))
            
            if time_expired:
                st.error(f"🔴 00:00")
            elif remaining.total_seconds() <= 300:  # < 5 min
                st.error(f"🔴 {minutes:02d}:{seconds:02d}")
            elif remaining.total_seconds() <= 600:  # < 10 min
                st.warning(f"🟡 {minutes:02d}:{seconds:02d}")
            else:
                st.success(f"🟢 {minutes:02d}:{seconds:02d}")
        
        with col2:
            st.markdown("**Time Elapsed:**")
            time_used_min = int(elapsed.total_seconds() // 60)
            time_used_sec = int(elapsed.total_seconds() % 60)
            st.metric("", f"{time_used_min}:{time_used_sec:02d}")
        
        # Single progress bar for overall session progress
        time_progress = min(elapsed.total_seconds() / SESSION_DURATION.total_seconds(), 1.0)
        st.progress(time_progress)
        st.caption(f"_Timer updates when you send messages (Streamlit limitation)_")

    # Progress tracker
    st.markdown("### Progress")
    st.metric("Observations", f"{st.session_state.current_question_idx} / 4")

    # Control buttons
    st.markdown("### Controls")
    
    # Help button
    if st.button("ℹ️ Help", use_container_width=True, key="help_button"):
        st.session_state["show_help"] = True
        st.rerun()
    
    # Start button (only in intro phase)
    if st.session_state.guided_phase == "intro":
        if st.button("▶️ Start", use_container_width=True, key="start_button", type="primary"):
            st.session_state.guided_session_start = datetime.now()
            st.session_state.guided_phase = "active"
            st.session_state.current_question_idx = 0
            elapsed_seconds = 0
            analytics.phase_transition(
                username,
                st.session_state.guided_session_id,
                "intro",
                "active",
                elapsed_seconds,
                {"observations_completed": 0},
            )
            sync_session_to_storage()
            st.rerun()
    
    # Next button (only in active/review phases)
    if st.session_state.guided_phase in ("active", "review"):
        if st.button("⏭️ Next", use_container_width=True, key="next_button", type="primary"):
            current_idx = st.session_state.current_question_idx
            next_idx = current_idx + 1
            
            if next_idx >= len(st.session_state.question_bank):
                success, error = handle_navigation(current_idx, "review", log_event=False)
            else:
                success, error = handle_navigation(next_idx, "active")
            
            if not success:
                st.error(f"Cannot proceed: {error}")
            else:
                st.rerun()

    # Show reference conversation
    with st.expander("Reference Conversation", expanded=True):
        ref_conversation = load_reference_conversation()
        if ref_conversation:
            for i, turn in enumerate(ref_conversation, 1):
                is_client = turn.strip().startswith("Client:")
                if is_client:
                    st.caption(f"**[Line {i}]** {turn}")
                else:
                    st.caption(f"*[Line {i}]* {turn}")
        else:
            st.warning("Reference conversation not loaded")

# ==================== MAIN CONTENT ====================

st.title("Module 2")
st.markdown(
    "<p style='font-size:12px;color:#e11d48;margin-top:6px;'>"
    "<strong>Privacy Reminder:</strong> Please do not include any identifying information in your messages.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:12px;color:#6b7280;margin-bottom:6px;'>"
    "Disclaimer: TeamMait may be incorrect or incomplete. Please verify information.</p>",
    unsafe_allow_html=True,
)

# Show RAG load warnings
if (
    not st.session_state.rag_load_status["reference"]
    or st.session_state.rag_load_status["total"] < 5
):
    st.warning(
        "**Limited context available**: "
        "Reference transcript may not be fully loaded. AI responses may be less informed."
    )

# ==================== INTRO PHASE ====================

if st.session_state.guided_phase == "intro":
    # Store that intro has been shown to preserve content
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = True
    
    st.markdown(
        """
    ## Welcome to Module 2

    In this phase, I'll share **4 structured observations** about the therapy session.

    ### How to use:
    1. **Read each observation** - I'll present an assertion and context
    2. **Discuss or advance** - You can:
       - Type a question or comment to discuss the observation further
       - Click the **⏭️ Next** button to move to the next observation
    3. **Review phase** - After all 4 observations, you can revisit any to discuss further
    4. **Time limit** - You have **20 minutes total** for the entire session

    ### Important rules:
    - You can only move **forward** through observations (required for the experimental design)
    - In the review phase, you can go back and revisit any observation
    - All your responses are automatically saved

    **Ready to begin?** Click the **▶️ Start** button in the sidebar.
    """
    )
    
    # Show help if requested
    if st.session_state.get("show_help", False):
        st.info(InputParser.get_help_message())
        st.session_state["show_help"] = False

# ==================== ACTIVE PHASE ====================

elif st.session_state.guided_phase == "active":
    if st.session_state.current_question_idx < len(st.session_state.question_bank):
        current_q = st.session_state.question_bank[
            st.session_state.current_question_idx
        ]
        current_idx = st.session_state.current_question_idx

        # Show observation header
        st.markdown(f"### Observation {current_idx + 1} of 4")
        st.divider()

        # Show the observation with better structure
        with st.container(border=True):
            st.markdown("**Assertion**")
            st.markdown(current_q.get("assertion", "Observation"))

            st.markdown("**Context**")
            st.markdown(current_q.get("explanation", ""))

            st.markdown("**Your thoughts:**")
            st.markdown(f"*{current_q.get('invitation', 'What are your thoughts?')}*")

        st.divider()
        st.info(
            "Use the **⏭️ Next** button to move to the next observation, "
            "or type a response to discuss this observation."
        )
        st.divider()

        # Display conversation history for this observation WITH TIMESTAMPS
        for msg in st.session_state.all_conversations[current_idx]:
            with st.chat_message(msg["role"]):
                # Show timestamp if available
                timestamp = msg.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M")
                        st.caption(f"_{time_str}_")
                    except:
                        pass
                st.markdown(msg["content"])

        # User input
        user_input = st.chat_input("Your response or question...")

        if user_input:
            # Check if time expired while user was typing
            if time_expired:
                st.error("Session time has expired. Your response could not be saved.")
                st.session_state.guided_phase = "expired"
                sync_session_to_storage()
                # st.rerun()

            # Check for navigation intent first
            if InputParser.detect_navigation_intent(user_input):
                st.info(InputParser.get_navigation_redirect_message())
            else:
                # Log user message - handle None start time
                if st.session_state.guided_session_start is None:
                    elapsed_seconds = 0
                else:
                    elapsed_seconds = (
                        datetime.now() - st.session_state.guided_session_start
                    ).total_seconds()
                
                analytics.user_message(
                    username,
                    st.session_state.guided_session_id,
                    current_idx,
                    elapsed_seconds,
                    len(user_input),
                    "message",
                )

                # Check for duplicates
                if not st.session_state.message_buffer.add_message(user_input):
                    st.warning("That looks like the same message. Please type something new.")
                else:
                    # Add to history WITH TIMESTAMP
                    st.session_state.all_conversations[current_idx].append(
                        {
                            "role": "user",
                            "content": user_input,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                    with st.chat_message("user"):
                        st.caption(f"_{datetime.now().strftime('%H:%M')}_")
                        st.markdown(user_input)

                        # Generate AI response
                        current_q_data = st.session_state.question_bank[current_idx]
                        context = retrieve_context(user_input)
                        system_prompt = (
                            build_system_prompt()
                            + f"\n\nCurrent observation:\n"
                            f"Assertion: {current_q_data.get('assertion', '')}\n"
                            f"Explanation: {current_q_data.get('explanation', '')}\n\n"
                            f"Context from transcript:\n{context}"
                        )

                        with st.chat_message("assistant"):
                            placeholder = st.empty()

                            # Show loading state
                            with placeholder.container():
                                st.markdown("*Thinking...*")

                            try:
                                # Generate response
                                acc = ""
                                start_time = time.time()
                                first_chunk = True

                                response_gen = OpenAIHandler.openai_complete(
                                    history=st.session_state.all_conversations[current_idx],
                                    system_text=system_prompt,
                                    client=client,
                                    stream=True,
                                    max_tokens=512,
                                    max_retries=2,
                                    timeout=30,
                                )

                                for chunk in response_gen:
                                    if first_chunk:
                                        placeholder.empty()
                                        first_chunk = False

                                    acc += chunk
                                    placeholder.markdown(acc)

                                generation_time = time.time() - start_time

                                # Save to history WITH TIMESTAMP
                                st.session_state.all_conversations[current_idx].append(
                                    {
                                        "role": "assistant",
                                        "content": acc.strip(),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )

                                # Log response
                                if st.session_state.guided_session_start is not None:
                                    elapsed_seconds = (
                                        datetime.now() - st.session_state.guided_session_start
                                    ).total_seconds()
                                else:
                                    elapsed_seconds = 0
                                tokens_estimated = len(acc) // 4
                                analytics.ai_response(
                                    username,
                                    st.session_state.guided_session_id,
                                    current_idx,
                                    elapsed_seconds,
                                    len(acc),
                                    tokens_estimated,
                                    generation_time,
                                )

                                sync_session_to_storage()
                                st.rerun()

                            except (APIRetryableError, APIPermanentError) as e:
                                error_msg = OpenAIHandler.format_error_message(e)
                                placeholder.error(error_msg)

                                # Remove the user message to keep state clean
                                st.session_state.all_conversations[current_idx].pop()
                                sync_session_to_storage()

                                if st.session_state.guided_session_start is not None:
                                    elapsed_seconds = (
                                        datetime.now() - st.session_state.guided_session_start
                                    ).total_seconds()
                                else:
                                    elapsed_seconds = 0
                                analytics.error_occurred(
                                    username,
                                    st.session_state.guided_session_id,
                                    type(e).__name__,
                                    str(e),
                                    elapsed_seconds,
                                    {"context": "ai_response", "observation_idx": current_idx},
                                )

                                logger.error(f"API error in observation {current_idx}: {e}")

                            except Exception as e:
                                error_msg = "Unexpected error. Please try again."
                                placeholder.error(error_msg)

                                # Remove the user message
                                st.session_state.all_conversations[current_idx].pop()
                                sync_session_to_storage()

                                if st.session_state.guided_session_start is not None:
                                    elapsed_seconds = (
                                        datetime.now() - st.session_state.guided_session_start
                                    ).total_seconds()
                                else:
                                    elapsed_seconds = 0
                                analytics.error_occurred(
                                    username,
                                    st.session_state.guided_session_id,
                                    type(e).__name__,
                                    str(e),
                                    elapsed_seconds,
                                    {"context": "ai_response", "observation_idx": current_idx},
                                )

                                logger.error(
                                    f"Unexpected error in observation {current_idx}: {e}",
                                    exc_info=True,
                                )

    else:
        # All observations done
        st.session_state.guided_phase = "review"
        sync_session_to_storage()
        st.rerun()

# ==================== TIME EXPIRED PHASE ====================

elif st.session_state.guided_phase == "expired":
    # Initialize warning states
    if "time_warning_2min_shown" not in st.session_state:
        st.session_state.time_warning_2min_shown = False
    
    current_idx = min(
        st.session_state.current_question_idx,
        len(st.session_state.question_bank) - 1,
    )
    
    st.error("### Session Time Expired")
    
    st.markdown(
        """
    Your 20-minute session has ended.
    
    ### Your Discussion is Saved
    
    All your messages below are preserved and will be included in your results.
    
    ### One More Chance
    
    You can share **final thoughts** below for a brief synthesis from TeamMait.
    
    Or skip directly to the next step.
    """
    )
    
    st.divider()
    
    # Show previous messages with timestamps (NEVER ERASE!)
    st.markdown("### Your Previous Discussion")
    if st.session_state.all_conversations[current_idx]:
        for msg in st.session_state.all_conversations[current_idx]:
            with st.chat_message(msg["role"]):
                # Show timestamp if available
                timestamp = msg.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M")
                        st.caption(f"_{time_str}_")
                    except:
                        pass
                st.markdown(msg["content"])
    else:
        st.info("No messages yet")
    
    st.divider()
    
    # Input for final thoughts
    st.markdown("### Final Thoughts (Optional)")
    user_input = st.chat_input(
        "Type your final thoughts...",
        placeholder="Leave blank to skip to next step"
    )
    
    if user_input:
        # Add user message with timestamp
        st.session_state.all_conversations[current_idx].append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        with st.chat_message("user"):
            st.caption(f"_{datetime.now().strftime('%H:%M')}_")
            st.markdown(user_input)
        
        # Generate synthesis (NOT summary)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            try:
                with placeholder.container():
                    st.markdown("*Synthesizing your insights...*")
                
                system_prompt = (
                    build_system_prompt()
                    + "\n\n"
                    "Provide a brief synthesis of the key insights and observations "
                    "from this discussion. Do NOT summarize what was said. Instead, "
                    "focus on: What was learned? What patterns emerged? What stands out? "
                    "Keep it to 2-3 sentences maximum. Be insightful, not repetitive."
                )
                
                acc = ""
                start_time = time.time()
                first_chunk = True
                
                response_gen = OpenAIHandler.openai_complete(
                    history=st.session_state.all_conversations[current_idx],
                    system_text=system_prompt,
                    client=client,
                    stream=True,
                    max_tokens=256,
                    max_retries=2,
                )
                
                for chunk in response_gen:
                    if first_chunk:
                        placeholder.empty()
                        first_chunk = False
                    
                    acc += chunk
                    placeholder.markdown(acc)
                
                # Save synthesis with timestamp
                st.session_state.all_conversations[current_idx].append(
                    {
                        "role": "assistant",
                        "content": acc.strip(),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                sync_session_to_storage()
                
            except Exception as e:
                placeholder.error(
                    f"Could not generate synthesis: {str(e)}"
                )
                logger.error(f"Synthesis generation error: {e}")
    
    # Buttons
    st.divider()
    st.markdown("### Next Step")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "Skip Final Thoughts",
            use_container_width=True,
            key="skip_final"
        ):
            st.session_state.guided_phase = "complete"
            sync_session_to_storage()
            st.rerun()
    
    with col2:
        if st.button(
            "Continue to Results",
            type="primary",
            use_container_width=True,
            key="continue_after_expired"
        ):
            st.session_state.guided_phase = "complete"
            sync_session_to_storage()
            st.rerun()
    
    # Reminder
    st.info(
        "**Reminder:** Don't forget to check the "
        "**'Check this when done'** checkbox in the sidebar!"
    )

# ==================== REVIEW PHASE ====================

elif st.session_state.guided_phase == "review":
    reviewed = min(
        st.session_state.current_question_idx,
        len(st.session_state.question_bank),
    )

    st.success(f"### ✅ You've completed all observations!")
    
    remaining_time_sec = max(0, int(remaining.total_seconds()))
    if remaining_time_sec > 0:
        remaining_min = remaining_time_sec // 60
        remaining_sec = remaining_time_sec % 60
        st.markdown(f"**Observations reviewed:** {reviewed} / 4")
        st.markdown(f"**Time remaining:** {remaining_min}:{remaining_sec:02d}")
        st.markdown(
            "Would you like to revisit any of the prior observations to discuss further?"
        )
    else:
        st.warning("Your session time has expired.")
        st.markdown("Please proceed to finish.")

    st.divider()

    if time_expired or remaining_time_sec <= 0:
        if st.button(
            "Finish and Continue to Next Step",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.guided_phase = "complete"
            sync_session_to_storage()
            st.rerun()
    else:
        st.markdown("### Revisit observations:")
        cols = st.columns(4)

        for i in range(min(4, len(st.session_state.question_bank))):
            with cols[i]:
                if st.button(f"Obs {i + 1}", use_container_width=True, key=f"revisit_{i}"):
                    success, error = handle_navigation(i, "review", log_event=False)
                    if success:
                        # Calculate elapsed time safely
                        if st.session_state.guided_session_start is not None:
                            elapsed = (datetime.now() - st.session_state.guided_session_start).total_seconds()
                        else:
                            elapsed = 0
                        analytics.observation_revisited(
                            username,
                            st.session_state.guided_session_id,
                            st.session_state.current_question_idx,
                            i,
                            elapsed,
                        )
                        st.rerun()

        st.divider()

        if st.button(
            "Finish and Continue to Next Step",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.guided_phase = "complete"
            sync_session_to_storage()
            st.rerun()

# ==================== COMPLETE PHASE ====================

elif st.session_state.guided_phase == "complete":
    reviewed = min(st.session_state.current_question_idx, 4)
    total_messages = sum(len(c) for c in st.session_state.all_conversations.values())
    # Defensive check: if session_start is None, default to 0 elapsed time
    if st.session_state.guided_session_start is not None:
        elapsed_seconds = (datetime.now() - st.session_state.guided_session_start).total_seconds()
    else:
        elapsed_seconds = 0

    st.success("### Module Complete")
    st.markdown(
        f"""
    Thank you for completing thi module!

    **Summary:**
    - **Observations reviewed:** {reviewed} / 4
    - **Total messages:** {total_messages}
    - **Time used:** {int(elapsed_seconds // 60)}:{int(elapsed_seconds % 60):02d}
    """
    )

    # Log completion
    analytics.session_completed(
        username,
        st.session_state.guided_session_id,
        reviewed,
        total_messages,
        elapsed_seconds,
        status="success",
    )

    # Mark session as complete in storage
    SessionManager.complete_session(
        username,
        st.session_state.guided_session_id,
        {
            "observations_reviewed": reviewed,
            "total_messages": total_messages,
            "elapsed_seconds": elapsed_seconds,
        },
    )

    # Clean up message buffer
    st.session_state.message_buffer.clear()

    if st.button("Continue to Next Step", type="primary", use_container_width=True):
        st.switch_page("pages/4_Finish.py")