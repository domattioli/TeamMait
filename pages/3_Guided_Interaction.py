import streamlit as st
import sys
import os
from datetime import datetime
import json
import glob
import random

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

try:
    from docx import Document
except ImportError:
    Document = None

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
        "Anchor every claim to the transcript (and provided docs). When citing the transcript, "
        "use the line numbers provided in the format [Line X]. If uncertain, say so briefly. "
        "Be succinct and academically neutral; do not use emojis. "
        "Never invent facts. Always cite specific line references when making claims about the transcript. "
        "You cannot offer any visual or audio support -- only text responses."
    )
    
    if mode == "guided":
        return base_prompt + (
            "\n\nYou are operating in GUIDED REVIEW mode. You present observations about the transcript "
            "and engage in natural discussion with the clinician. When they respond to your observations, "
            "engage conversationally and provide thoughtful analysis. You should respond naturally to their "
            "questions and comments about the transcript without requiring structured response formats. "
            "Keep responses focused and clinically relevant. Always reference specific line numbers "
            "when discussing transcript content (e.g., 'In Line 5, the therapist...')."
        )
    
    return base_prompt

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text content from a DOCX file."""
    if Document is None:
        return f"[Error: python-docx not installed. Cannot read {os.path.basename(docx_path)}]"
    
    try:
        doc = Document(docx_path)
        text_parts = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        return f"[Error reading {os.path.basename(docx_path)}: {str(e)}]"

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
@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_rag_documents():
    """Load all RAG documents and seed ChromaDB collection."""
    _, collection = initialize_chroma()
    
    doc_folder = "doc/RAG"
    supporting_folder = os.path.join(doc_folder, "supporting_documents")
    documents = []
    ids = []
    
    # Load main reference conversation with line numbers
    ref_path = os.path.join(doc_folder, "116_P8_conversation.json")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "full_conversation" in data:
                for i, turn in enumerate(data["full_conversation"]):
                    line_num = i + 1  # 1-indexed line numbers
                    # Add line number to the document for better citation
                    numbered_turn = f"[Line {line_num}] {turn}"
                    documents.append(numbered_turn)
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
    
    # Load .docx files from supporting_documents
    for docx_path in glob.glob(os.path.join(supporting_folder, "*.docx")):
        content = extract_text_from_docx(docx_path)
        if content and not content.startswith("[Error"):
            # Split large documents into chunks for better retrieval
            words = content.split()
            chunk_size = 500  # words per chunk
            
            if len(words) <= chunk_size:
                documents.append(content)
                ids.append(f"supp_docx_{os.path.basename(docx_path)}")
            else:
                # Split into chunks
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    documents.append(chunk)
                    ids.append(f"supp_docx_{os.path.basename(docx_path)}_chunk_{i//chunk_size + 1}")
        else:
            # Add error message as document for debugging
            documents.append(content)
            ids.append(f"supp_docx_error_{os.path.basename(docx_path)}")
    
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
            "all_domains_covered": False,
            "show_feedback_buttons": False  # Track when to show thumbs up/down
        }

def get_next_question():
    """Select a random question from the bank that hasn't been asked yet."""
    question_bank = load_question_bank()
    asked = st.session_state.flowchart_state["questions_asked"]
    
    available = [q for q in question_bank if q["id"] not in asked]
    
    if not available:
        return None
    
    selected = random.choice(available)
    st.session_state.flowchart_state["questions_asked"].append(selected["id"])
    st.session_state.flowchart_state["current_question"] = selected
    
    # Debug: Print when question is added
    current_count = len(st.session_state.flowchart_state["questions_asked"])
    print(f"DEBUG: Added question {selected['id']}, total asked: {current_count}")
    
    return selected

def format_prompt_message(question: dict) -> str:
    """Format a question into a naturalistic prompt message."""
    return (
        f"{question['assertion']} {question['explanation']}\n\n"
        f"{question['invitation']}"
    )

def detect_response_type(user_input: str) -> str:
    """Detect the response type from user input using LLM classification."""
    
    # Get the current prompt context
    current_prompt = st.session_state.flowchart_state.get("current_question", {})
    prompt_context = ""
    if current_prompt:
        prompt_context = f"The observation was: {current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}"
    
    classification_prompt = f"""You are analyzing a user's response to a clinical observation about a therapy transcript. 

{prompt_context}

The user responded: "{user_input}"

Classify this response into exactly ONE of these categories:

1. "accept" - User agrees with, accepts, or acknowledges the observation as accurate
2. "correct" - User disagrees and wants to provide corrections, alternative perspectives, or thinks the observation is wrong
3. "clarify" - User wants more explanation, elaboration, or doesn't understand something about the observation
4. "disregard" - User wants to skip this topic, move on, or considers it not relevant
5. "unclear" - Response doesn't clearly fit the above categories or is ambiguous

Respond with ONLY the single word category (accept, correct, clarify, disregard, or unclear). No explanation needed."""

    client = get_openai_client()
    if client is None:
        # Fallback to keyword matching if OpenAI is unavailable
        user_lower = user_input.lower()
        
        accept_keywords = ["accept", "agree", "yes", "correct", "right", "accurate"]
        correct_keywords = ["correct", "actually", "disagree", "no", "wrong", "instead", "i don't know about that", "i don't think so", "i don't agree", "i don't think that's right", "not quite", "not really"]
        clarify_keywords = [
            "clarify", "explain", "elaborate", "more", "unclear", "expand",
            "what do you mean", "what does that mean", "can you explain", 
            "i don't understand", "i dont understand", "confused", "what",
            "how so", "in what way", "could you elaborate", "tell me more"
        ]
        disregard_keywords = ["disregard", "skip", "pass", "not relevant", "move on"]
        
        if any(kw in user_lower for kw in disregard_keywords):
            return "disregard"
        elif any(kw in user_lower for kw in clarify_keywords):
            return "clarify"
        elif any(kw in user_lower for kw in accept_keywords):
            return "accept"
        elif any(kw in user_lower for kw in correct_keywords):
            return "correct"
        else:
            return "unclear"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate the response is one of our expected categories
        valid_categories = ["accept", "correct", "clarify", "disregard", "unclear"]
        if classification in valid_categories:
            return classification
        else:
            return "unclear"
            
    except Exception as e:
        print(f"Error in LLM classification: {e}")
        return "unclear"

def detect_next_question_request(user_input: str) -> bool:
    """Detect if user is explicitly requesting the next question using LLM classification."""
    
    # Get context about what just happened
    last_message = ""
    if (len(st.session_state.guided_messages) > 0 and 
        st.session_state.guided_messages[-1].get("role") == "assistant"):
        last_message = st.session_state.guided_messages[-1]["content"]
    
    context_info = ""
    if "would you like me to share my next observation" in last_message.lower():
        context_info = "The system just offered to share the next observation about the session."
    elif st.session_state.flowchart_state["stage"] == "intro":
        context_info = "This is the beginning of the session and the user is being asked to request the first question."
    else:
        context_info = "The user is currently in an open discussion about a therapy transcript observation."
    
    classification_prompt = f"""You are analyzing a user's response in a clinical supervision conversation to determine if they want to proceed to the next structured observation/question.

Context: {context_info}

The user said: "{user_input}"

Does this response indicate that the user wants to:
- Move to the next structured observation/question from the question bank
- Start or continue with the structured review process  
- Proceed to the next item in the guided interaction
- Begin the guided interaction (if at the start)

This includes:
- Direct requests like "next question" or "what's next"
- Expressions of readiness like "ready", "let's go", "I'm listening"
- Positive responses to offers like "yes", "sure", "go ahead"
- Indicating they want to continue or proceed

Respond with ONLY "yes" if they want the next question/observation, or "no" if they want to continue current discussion. No explanation needed."""

    client = get_openai_client()
    if client is None:
        # Fallback to keyword matching if OpenAI is unavailable
        user_lower = user_input.lower()
        
        next_question_keywords = [
            "next question", "next observation", "next prompt", "give me another",
            "show me another", "what's next", "another question", "move to next",
            "next item", "continue with questions", "more questions"
        ]
        
        positive_responses = [
            "yes", "yeah", "sure", "ok", "okay", "go ahead", "please", 
            "yes please", "that would be good", "sounds good", "let's do it",
            "lets do it", "i would like that", "id like that"
        ]
        
        readiness_indicators = [
            "ready", "im ready", "i'm ready", "let's go", "lets go", "start", 
            "begin", "let's start", "lets start", "let's begin", "lets begin",
            "go ahead", "go for it", "shoot", "fire away", "hit me", 
            "i'm listening", "im listening", "proceed", "continue"
        ]
        
        return (any(phrase in user_lower for phrase in next_question_keywords) or 
                any(response in user_lower for response in positive_responses) or
                any(indicator in user_lower for indicator in readiness_indicators))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=5,
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Return True if LLM says "yes" (user wants next question)
        return classification == "yes"
            
    except Exception as e:
        print(f"Error in LLM next question detection: {e}")
        # Fallback to keyword matching on error
        user_lower = user_input.lower()
        
        next_question_keywords = [
            "next question", "next observation", "next prompt", "give me another",
            "show me another", "what's next", "another question", "move to next",
            "next item", "continue with questions", "more questions"
        ]
        
        positive_responses = [
            "yes", "yeah", "sure", "ok", "okay", "go ahead", "please", 
            "yes please", "that would be good", "sounds good", "let's do it",
            "lets do it", "i would like that", "id like that"
        ]
        
        readiness_indicators = [
            "ready", "im ready", "i'm ready", "let's go", "lets go", "start", 
            "begin", "let's start", "lets start", "let's begin", "lets begin",
            "go ahead", "go for it", "shoot", "fire away", "hit me", 
            "i'm listening", "im listening", "proceed", "continue"
        ]
        
        return (any(phrase in user_lower for phrase in next_question_keywords) or 
                any(response in user_lower for response in positive_responses) or
                any(indicator in user_lower for indicator in readiness_indicators))

def detect_decline_to_engage(user_input: str) -> bool:
    """Detect if user is declining to engage further with current topic using LLM classification."""
    
    classification_prompt = f"""You are analyzing a user's response in a clinical supervision conversation. The user has been discussing a therapy transcript observation, and you need to determine if they want to disengage from the current topic.

The user said: "{user_input}"

Does this response indicate that the user wants to:
- Stop discussing the current topic
- Move on to something else  
- Skip or avoid this particular observation
- Show disinterest in continuing this line of discussion

Respond with ONLY "yes" if they want to disengage, or "no" if they want to continue engaging. No explanation needed."""

    client = get_openai_client()
    if client is None:
        # Fallback to keyword matching if OpenAI is unavailable
        user_lower = user_input.lower().strip()
        
        decline_patterns = [
            "no", "nope", "not really", "i don't want", "i dont want", 
            "not interested", "let's move on", "lets move on", "move on",
            "skip this", "skip that", "don't want to talk", "dont want to talk",
            "rather not", "not now", "maybe later", "pass"
        ]
        
        return any(pattern in user_lower for pattern in decline_patterns)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=5,
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Return True if LLM says "yes" (user wants to disengage)
        return classification == "yes"
            
    except Exception as e:
        print(f"Error in LLM decline detection: {e}")
        # Fallback to keyword matching on error
        user_lower = user_input.lower().strip()
        decline_patterns = [
            "no", "nope", "not really", "i don't want", "i dont want", 
            "not interested", "let's move on", "lets move on", "move on",
            "skip this", "skip that", "don't want to talk", "dont want to talk",
            "rather not", "not now", "maybe later", "pass"
        ]
        return any(pattern in user_lower for pattern in decline_patterns)

def handle_flowchart_transition(user_input: str) -> dict:
    """Handle flowchart state transitions based on user input."""
    state = st.session_state.flowchart_state
    stage = state["stage"]
    
    # Check for inactivity and suggest moving to next question
    questions_asked = len(st.session_state.flowchart_state["questions_asked"])
    if questions_asked < 4 and stage == "open_discussion":
        # Get the timestamp of the last message
        last_message_time = None
        if st.session_state.guided_messages:
            last_message = st.session_state.guided_messages[-1]
            if "ts" in last_message:
                try:
                    # Parse the timestamp (format: HH:MM:SS)
                    last_ts = datetime.strptime(last_message["ts"], "%H:%M:%S")
                    # Get today's date and combine with the time
                    today = datetime.now().date()
                    last_message_time = datetime.combine(today, last_ts.time())
                    
                    # Handle case where the last message was from yesterday (edge case)
                    current_time = datetime.now()
                    if last_message_time > current_time:
                        # Message was from yesterday, subtract a day
                        from datetime import timedelta
                        last_message_time = last_message_time - timedelta(days=1)
                    
                    # Check if more than 5 minutes have passed
                    time_diff = current_time - last_message_time
                    if time_diff.total_seconds() > 300:  # 5 minutes = 300 seconds
                        # Check if we haven't already offered this in the last few messages
                        recent_offers = False
                        for msg in st.session_state.guided_messages[-3:]:  # Check last 3 messages
                            if ("would you like to move on to the next question" in msg.get("content", "").lower() or
                                "been a while since we" in msg.get("content", "").lower()):
                                recent_offers = True
                                break
                        
                        if not recent_offers:
                            return {
                                "next_stage": stage,  # Stay in current stage
                                "bot_response": "It's been a while since we last discussed this topic. Would you like to move on to the next question, or is there anything else you'd like to explore about this observation?",
                                "show_buttons": False,
                                "show_feedback_buttons": False,
                                "use_llm": False
                            }
                except (ValueError, AttributeError):
                    # If timestamp parsing fails, continue with normal flow
                    pass
    
    # Check if user has completed all questions but hasn't checked the completion box
    completion_checked = st.session_state.get("completion_status", {}).get("guided_interaction", False)
    
    # If they've completed all questions and are trying to move on, remind them to check the box
    if questions_asked >= 4 and not completion_checked:
        # Check if they're trying to end the session or move on
        ending_phrases = [
            "done", "finished", "complete", "that's all", "thats all", "thanks", "thank you",
            "goodbye", "bye", "see you", "i'm finished", "im finished", "wrap up",
            "end session", "all set", "ready to go", "ready to move on"
        ]
        
        if any(phrase in user_input.lower() for phrase in ending_phrases):
            return {
                "next_stage": stage,  # Stay in current stage
                "bot_response": "It looks like you've completed reviewing all the questions! Before you move on, please remember to check the completion box in the sidebar to mark this section as finished. This helps track your progress through the session.",
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
    
    # Check if we just offered analysis (look at last bot message for analysis offer)
    if (len(st.session_state.guided_messages) > 0 and 
        st.session_state.guided_messages[-1].get("role") == "assistant"):
        last_message = st.session_state.guided_messages[-1]["content"]
        print(f"DEBUG: Last message: {last_message[:100]}...")
        if ("would you like me to provide an analysis" in last_message.lower() or 
            "would you like an analysis" in last_message.lower() or
            "provide an analysis" in last_message.lower()):
            print(f"DEBUG: Analysis offer detected! User input: {user_input}")
            # User is responding to analysis offer - check if they're accepting
            user_lower = user_input.lower()
            accept_analysis = any(phrase in user_lower for phrase in [
                "yes", "yeah", "sure", "ok", "okay", "yes please", "please",
                "that would be good", "sounds good", "go ahead", "i would like that"
            ])
            print(f"DEBUG: Accept analysis: {accept_analysis}")
            if accept_analysis:
                # Provide the requested analysis using LLM with specific context
                current_prompt = st.session_state.flowchart_state.get("current_question", {})
                analysis_context = f"""The user has requested an analysis. Based on the previous observation: "{current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}"

Please provide the specific analysis that was offered in the previous message. Focus on the particular aspect mentioned in the analysis offer and provide detailed, evidence-based insights from the transcript with line number citations."""

                return {
                    "next_stage": "open_discussion",
                    "bot_response": None,
                    "show_buttons": False, 
                    "show_feedback_buttons": False,
                    "use_llm": True,
                    "api_context": analysis_context
                }
            # If they didn't accept, continue with normal flow
    
    # PRIORITY CHECK: Check for next question request 
    # BUT NOT if we're in prompt stage (responding to bank question) - let stage logic handle it first
    # ALSO NOT if we just offered analysis and user declined it
    if stage != "prompt" and detect_next_question_request(user_input):
        question = get_next_question()
        if question:
            # Ensure we return ONLY the bank question, no LLM generation
            return {
                "next_stage": "prompt",
                "bot_response": format_prompt_message(question),
                "show_buttons": True,
                "show_feedback_buttons": True,
                "use_llm": False  # Explicitly prevent LLM usage
            }
        else:
            # Check if completion box is checked
            completion_checked = st.session_state.get("completion_status", {}).get("guided_interaction", False)
            if completion_checked:
                completion_msg = "All questions have been reviewed and marked complete! Great work on finishing the guided interaction."
            else:
                completion_msg = "All questions have been reviewed! Please remember to check the completion box in the sidebar to mark this section as finished before moving on to other parts of the session."
            
            return {
                "next_stage": "complete",
                "bot_response": completion_msg,
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
    
    if stage == "intro":
        # First question only starts when user explicitly asks for it
        return {
            "next_stage": "intro",
            "bot_response": "I'm ready to share my observations when you are. Just ask for the next question when you'd like to begin.",
            "show_buttons": False,
            "show_feedback_buttons": False,
            "use_llm": False  # Changed from True to False to prevent unintended LLM calls
        }
    
    elif stage == "prompt":
        response_type = detect_response_type(user_input)
        state["current_response_type"] = response_type
        
        if response_type == "disregard":
            return {
                "next_stage": "open_discussion",
                "bot_response": "Noted. Feel free to ask me anything else about this transcript, or request the next question when you're ready.",
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
        elif response_type == "accept":
            return {
                "next_stage": "open_discussion",
                "bot_response": "I'm glad you agree with that observation. Is there anything else you'd like to discuss about this topic, or are you ready to move on to my next observation?",
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
        elif response_type == "correct":
            return {
                "next_stage": "open_discussion",
                "bot_response": "I appreciate your perspective. What specifically would you like to discuss or clarify about this observation?",
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
        elif response_type == "clarify":
            return {
                "next_stage": "open_discussion",
                "bot_response": "Of course, I'd be happy to clarify. What specific part would you like me to explain further?",
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": False
            }
        else:
            # For unclear responses, use LLM to respond naturally
            current_prompt = st.session_state.flowchart_state.get("current_question", {})
            context = f"""The user said: "{user_input}"

I just shared this observation about their therapy session: {current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}

The user's response doesn't clearly indicate if they accept, want to correct, want clarification, or want to disregard my observation. Please respond naturally and conversationally, asking them to clarify what they mean or help them engage with the observation. Don't mention response categories - just have a natural conversation to understand what they're thinking."""
            
            return {
                "next_stage": "prompt",
                "bot_response": "",  # Will be filled by LLM
                "show_buttons": True,
                "show_feedback_buttons": True,
                "use_llm": True,
                "api_context": context,
                "conversation_context": f"Clarifying unclear response about: {current_prompt.get('assertion', '')}"
            }
    
    elif stage == "open_discussion":
        # Check if user is asking for clarification about the current observation
        clarification_requests = [
            "clarify", "explain", "rationale", "reasoning", "why", "how did you",
            "what do you mean", "can you elaborate", "tell me more", "break down"
        ]
        
        is_clarification = any(phrase in user_input.lower() for phrase in clarification_requests)
        
        if is_clarification:
            # Provide focused clarification about the current observation
            current_prompt = st.session_state.flowchart_state.get("current_question", {})
            clarification_context = f"""The user is asking for clarification about this specific observation: "{current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}"

User's request: "{user_input}"

Please provide a concise, focused clarification of your rationale for this observation. Reference specific line numbers from the transcript to support your reasoning. Keep it brief and directly address what they're asking about.

End with: "Does this help clarify things, or would you like a more in-depth explanation?" """

            return {
                "next_stage": "open_discussion",
                "bot_response": None,
                "show_buttons": False,
                "show_feedback_buttons": False,
                "use_llm": True,
                "api_context": clarification_context
            }
        
        # Check if user is declining to engage further - offer next observation
        if detect_decline_to_engage(user_input):
            # Check if there are more questions available
            asked = st.session_state.flowchart_state["questions_asked"]
            question_bank = load_question_bank()
            available = [q for q in question_bank if q["id"] not in asked]
            
            if available:
                return {
                    "next_stage": "open_discussion",
                    "bot_response": "That's perfectly fine. Would you like me to share my next observation about the session?",
                    "show_buttons": False,
                    "show_feedback_buttons": False,
                    "use_llm": False
                }
            else:
                # All questions completed - check if they've marked it complete
                completion_checked = st.session_state.get("completion_status", {}).get("guided_interaction", False)
                if completion_checked:
                    completion_msg = "Understood. We've covered all my prepared observations and you've marked this section complete. Feel free to ask me anything else about the session."
                else:
                    completion_msg = "Understood. We've covered all my prepared observations. If you're ready to finish this section, please remember to check the completion box in the sidebar. Otherwise, feel free to ask me anything else about the session."
                
                return {
                    "next_stage": "open_discussion", 
                    "bot_response": completion_msg,
                    "show_buttons": False,
                    "show_feedback_buttons": False,
                    "use_llm": False
                }
        
        # Stay in open discussion - answer any questions but don't auto-advance
        return {
            "next_stage": "open_discussion",
            "bot_response": None,
            "show_buttons": False,
            "show_feedback_buttons": False,
            "use_llm": True
        }
    
    # Final catch-all - check if all questions are done and remind about completion
    questions_asked = len(st.session_state.flowchart_state["questions_asked"])
    completion_checked = st.session_state.get("completion_status", {}).get("guided_interaction", False)
    
    if questions_asked >= 4 and not completion_checked:
        return {
            "next_stage": stage,
            "bot_response": "I'm here to help with any questions about the transcript. Since you've reviewed all the guided questions, don't forget to check the completion box in the sidebar when you're ready to finish this section.",
            "show_buttons": False,
            "show_feedback_buttons": False,
            "use_llm": False
        }
    
    return {
        "next_stage": stage,
        "bot_response": "I'm here to help with any questions about the transcript. Ask for the next question when you're ready to continue.",
        "show_buttons": False,
        "show_feedback_buttons": False,
        "use_llm": False
    }

# ---------- Session State ----------
initialize_flowchart_state()

if "guided_messages" not in st.session_state:
    st.session_state.guided_messages = [
        {
            "role": "assistant",
            "content": "Hi, my name is TeamMait. Feel free to ask me any questions about the referenced session transcript. It can be found in the left side panel.\n\nI've made a few observations about the session that we can discuss together. When you're ready, just ask me for the 'next question' and I'll share one with you.",
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
                line_num = i + 1  # 1-indexed line numbers
                is_client = turn.strip().startswith("Client: ")
                if is_client:
                    # Right-justify client's messages with custom CSS
                    st.markdown(f"""
                    <div style="text-align: right; margin-left: 0%; padding: 10px; border-radius: 10px;">
                    <small style="color: #888; font-size: 0.8em;">[Line {line_num}]</small><br>
                    {turn}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Italicize therapist's messages, but keep line numbers unbolded
                    st.markdown(f"<div style='font-weight:600; font-size:1.08em; margin: 10px 0;'><small style='color: #888; font-size: 0.8em; font-weight: normal;'>[Line {line_num}]</small><br><em>{turn}</em></div>", unsafe_allow_html=True)

# ---------- Main Content ----------
st.title("Guided Interaction")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Disclaimer: TeamMait may be incorrect or incomplete. Please verify information.</p>", unsafe_allow_html=True)

# Display chat history
for m in st.session_state.guided_messages:
    role = m["role"]
    # Add role labels for clarity
    role_label = "TeamMait" if role == "assistant" else "User"
    
    with st.chat_message(role):
        # Display role label and timestamp together
        if "ts" in m:
            st.markdown(f"**{role_label}** • <small style='color: #888; font-size: 0.8em;'>*{m['ts']}*</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{role_label}**")
        st.markdown(m["content"])

# ---------- Response Buttons ----------
def handle_button_click(response_type: str):
    """Handle quick response button clicks silently."""
    # Keep buttons visible but record the response
    st.session_state.flowchart_state["current_response_type"] = response_type
    st.session_state.flowchart_state["button_clicked"] = response_type
    
    st.rerun()

# Show buttons only when feedback is expected (after bank questions)
if st.session_state.flowchart_state.get("show_feedback_buttons", False):
    st.markdown("##### Grade Response (optional):")
    cols = st.columns(2)
    
    # Check if a button was clicked to show visual feedback
    button_clicked = st.session_state.flowchart_state.get("button_clicked", None)
    
    with cols[0]:
        button_text = "👍 ✓" if button_clicked == "accept" else "👍"
        st.button(button_text, on_click=lambda: handle_button_click("accept"), use_container_width=True)
    with cols[1]:
        button_text = "👎 ✓" if button_clicked == "disregard" else "👎"
        st.button(button_text, on_click=lambda: handle_button_click("disregard"), use_container_width=True)

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
            st.markdown(f"**User** • <small style='color: #888; font-size: 0.8em;'>*{user_msg['ts']}*</small>", unsafe_allow_html=True)
            st.markdown(prompt)
        
        # Process transition
        result = handle_flowchart_transition(prompt)
        st.session_state.flowchart_state["stage"] = result["next_stage"]
        
        # Update feedback button visibility
        st.session_state.flowchart_state["show_feedback_buttons"] = result.get("show_feedback_buttons", False)
        
        if result.get("use_llm"):
            # Generate LLM response
            if result.get("api_context"):
                # Use specific context for this response
                context = result["api_context"]
                system_prompt = build_system_prompt(mode="guided")
            else:
                # Use standard RAG context
                context, _ = retrieve_context(prompt)
                system_prompt = build_system_prompt(mode="guided") + f"\n\nContext:\n{context}"
            
            with st.chat_message("assistant"):
                # Show header first
                timestamp = now_ts()
                st.markdown(f"**TeamMait** • <small style='color: #888; font-size: 0.8em;'>*{timestamp}*</small>", unsafe_allow_html=True)
                
                placeholder = st.empty()
                acc = ""
                for chunk in openai_complete(
                    history=[{"role": "user", "content": context}],
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
                    "ts": timestamp
                }
                st.session_state.guided_messages.append(bot_msg)
                
                # Clear button state after assistant responds
                if "button_clicked" in st.session_state.flowchart_state:
                    del st.session_state.flowchart_state["button_clicked"]
                    
        elif result["bot_response"]:
            timestamp = now_ts()
            with st.chat_message("assistant"):
                st.markdown(f"**TeamMait** • <small style='color: #888; font-size: 0.8em;'>*{timestamp}*</small>", unsafe_allow_html=True)
                st.markdown(result["bot_response"])
                
            bot_msg = {
                "role": "assistant",
                "content": result["bot_response"],
                "ts": timestamp
            }
            st.session_state.guided_messages.append(bot_msg)
            
            # Clear button state after assistant responds
            if "button_clicked" in st.session_state.flowchart_state:
                del st.session_state.flowchart_state["button_clicked"]
        
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
            "all_domains_covered": False,
            "show_feedback_buttons": False
        }
        st.session_state.guided_messages = [
            {
                "role": "assistant",
                "content": "Welcome back! I'm ready to share more observations when you are. Just ask for the 'next question' to begin.",
                "ts": now_ts()
            }
        ]
        st.rerun()