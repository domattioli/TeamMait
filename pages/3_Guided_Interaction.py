import streamlit as st
import sys
import os
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict
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

# ==================== STATE MACHINE DEFINITION ====================

class ChatbotState(Enum):
    """Define all possible chatbot states."""
    IDLE = "idle"
    QUESTION_PRESENTATION = "question_presentation"
    OPEN_DISCUSSION = "open_discussion"
    SESSION_COMPLETE = "session_complete"

@dataclass
class StateTransition:
    """Represents a state transition with associated data."""
    next_state: ChatbotState
    bot_response: Optional[str] = None
    use_llm: bool = False
    show_buttons: bool = False
    show_feedback_buttons: bool = False
    api_context: Optional[str] = None

class GuidedInteractionStateMachine:
    """State machine for managing guided interaction flow."""
    
    def __init__(self, question_bank: List[Dict], openai_client):
        self.current_state = ChatbotState.IDLE
        self.questions_asked = []
        self.current_question = None
        self.current_response_type = None
        self.question_bank = question_bank
        self.openai_client = openai_client
        self.button_clicked = None
        
    def transition(self, user_input: str) -> StateTransition:
        """Main transition logic - routes to state-specific handlers."""
        
        # Check for next question request (priority across all states except IDLE)
        if self.current_state != ChatbotState.IDLE and self._wants_next_question(user_input):
            return self._advance_to_next_question()
        
        # Route to state-specific handler
        if self.current_state == ChatbotState.IDLE:
            return self._handle_idle(user_input)
        elif self.current_state == ChatbotState.QUESTION_PRESENTATION:
            return self._handle_question_presentation(user_input)
        elif self.current_state == ChatbotState.OPEN_DISCUSSION:
            return self._handle_open_discussion(user_input)
        elif self.current_state == ChatbotState.SESSION_COMPLETE:
            return self._handle_complete(user_input)
        
        # Fallback
        return StateTransition(next_state=self.current_state)
    
    # ========== STATE HANDLERS ==========
    
    def _handle_idle(self, user_input: str) -> StateTransition:
        """Handle IDLE state - waiting for user to request first question."""
        if self._wants_next_question(user_input):
            return self._advance_to_next_question()
        
        return StateTransition(
            next_state=ChatbotState.IDLE,
            bot_response="I'm ready to share my observations when you are. Just ask for the next question when you'd like to begin.",
            show_buttons=False,
            show_feedback_buttons=False
        )
    
    def _handle_question_presentation(self, user_input: str) -> StateTransition:
        """Handle QUESTION_PRESENTATION state - user responding to a bank question."""
        
        # Classify the response
        response_type = self._classify_response(user_input)
        self.current_response_type = response_type
        
        # Handle based on classification
        if response_type == "disregard_active" or response_type == "disregard_passive":
            self.current_state = ChatbotState.OPEN_DISCUSSION
            return self._offer_next_question()
        
        elif response_type.startswith("accept"):
            self.current_state = ChatbotState.OPEN_DISCUSSION
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                bot_response="I'm glad you agree with that observation. Is there anything else you'd like to discuss about this topic, or are you ready to move on to my next observation?",
                show_buttons=False,
                show_feedback_buttons=False
            )
        
        elif response_type.startswith("correct"):
            self.current_state = ChatbotState.OPEN_DISCUSSION
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                bot_response="I appreciate your perspective. What specifically would you like to discuss or clarify about this observation?",
                show_buttons=False,
                show_feedback_buttons=False
            )
        
        elif response_type.startswith("clarify"):
            self.current_state = ChatbotState.OPEN_DISCUSSION
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                bot_response="Of course, I'd be happy to clarify. What specific part would you like me to explain further?",
                show_buttons=False,
                show_feedback_buttons=False
            )
        
        else:  # unclear
            current_prompt = self.current_question or {}
            context = f"""The user said: "{user_input}"

I just shared this observation: {current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}

The user's response is unclear. Please respond naturally to understand what they're thinking."""
            
            return StateTransition(
                next_state=ChatbotState.QUESTION_PRESENTATION,
                use_llm=True,
                show_buttons=True,
                show_feedback_buttons=True,
                api_context=context
            )
    
    def _handle_open_discussion(self, user_input: str) -> StateTransition:
        """Handle OPEN_DISCUSSION state - free-form discussion about current topic."""
        
        # Check for clarification requests
        if self._is_clarification_request(user_input):
            current_prompt = self.current_question or {}
            context = f"""The user is asking for clarification about: "{current_prompt.get('assertion', '')} {current_prompt.get('explanation', '')}"

User's request: "{user_input}"

Provide a concise clarification with specific line numbers from the transcript. End with: "Does this help clarify things, or would you like a more in-depth explanation?" """
            
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                use_llm=True,
                api_context=context
            )
        
        # Check for disengagement
        if self._wants_to_disengage(user_input):
            return self._offer_next_question()
        
        # Continue discussion
        return StateTransition(
            next_state=ChatbotState.OPEN_DISCUSSION,
            use_llm=True
        )
    
    def _handle_complete(self, user_input: str) -> StateTransition:
        """Handle SESSION_COMPLETE state."""
        return StateTransition(
            next_state=ChatbotState.SESSION_COMPLETE,
            bot_response="The guided interaction is complete. You can start a new session or ask questions about the transcript.",
            show_buttons=False,
            show_feedback_buttons=False
        )
    
    # ========== HELPER METHODS ==========
    
    def _get_next_question(self) -> Optional[Dict]:
        """Select a random question that hasn't been asked yet."""
        available = [q for q in self.question_bank if q["id"] not in self.questions_asked]
        
        if not available:
            return None
        
        selected = random.choice(available)
        self.questions_asked.append(selected["id"])
        self.current_question = selected
        
        return selected
    
    def _format_question(self, question: Dict) -> str:
        """Format a question into a prompt message."""
        return (
            f"{question['assertion']} {question['explanation']}\n\n"
            f"{question['invitation']}"
        )
    
    def _advance_to_next_question(self) -> StateTransition:
        """Advance to the next question or complete the session."""
        question = self._get_next_question()
        
        if question:
            self.current_state = ChatbotState.QUESTION_PRESENTATION
            return StateTransition(
                next_state=ChatbotState.QUESTION_PRESENTATION,
                bot_response=self._format_question(question),
                show_buttons=True,
                show_feedback_buttons=True
            )
        else:
            self.current_state = ChatbotState.SESSION_COMPLETE
            completion_msg = "All questions have been reviewed! Please remember to check the completion box in the sidebar to mark this section as finished before moving on to other parts of the session."
            return StateTransition(
                next_state=ChatbotState.SESSION_COMPLETE,
                bot_response=completion_msg,
                show_buttons=False,
                show_feedback_buttons=False
            )
    
    def _offer_next_question(self) -> StateTransition:
        """Offer to move to the next question."""
        available = [q for q in self.question_bank if q["id"] not in self.questions_asked]
        
        if available:
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                bot_response="That's perfectly fine. Would you like me to share my next observation about the session?",
                show_buttons=False,
                show_feedback_buttons=False
            )
        else:
            return StateTransition(
                next_state=ChatbotState.OPEN_DISCUSSION,
                bot_response="Understood. We've covered all my prepared observations. If you're ready to finish this section, please remember to check the completion box in the sidebar. Otherwise, feel free to ask me anything else about the session.",
                show_buttons=False,
                show_feedback_buttons=False
            )
    
    def _wants_next_question(self, user_input: str) -> bool:
        """Detect if user wants the next question using LLM."""
        if not self.openai_client:
            # Fallback to keyword matching
            keywords = [
                "next question", "next observation", "what's next", "another question",
                "move on", "next", "ready", "yes", "sure", "go ahead", "please"
            ]
            return any(kw in user_input.lower() for kw in keywords)
        
        try:
            prompt = f"""Does this user response indicate they want to proceed to the next structured observation/question?

User said: "{user_input}"

Respond with ONLY "yes" or "no"."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower() == "yes"
        except:
            keywords = ["next", "ready", "yes", "sure", "go ahead"]
            return any(kw in user_input.lower() for kw in keywords)
    
    def _classify_response(self, user_input: str) -> str:
        """Classify user's response type using LLM."""
        if not self.openai_client:
            # Fallback to keyword matching
            user_lower = user_input.lower()
            if any(kw in user_lower for kw in ["don't want", "skip", "pass"]):
                return "disregard_active"
            elif any(kw in user_lower for kw in ["yes", "agree", "right"]):
                return "accept_passive"
            elif any(kw in user_lower for kw in ["no", "disagree", "wrong"]):
                return "correct_active"
            elif any(kw in user_lower for kw in ["what", "clarify", "explain"]):
                return "clarify_passive"
            return "unclear"
        
        try:
            current_prompt = self.current_question or {}
            prompt = f"""Classify this response into ONE category:

Observation: {current_prompt.get('assertion', '')}
User response: "{user_input}"

Categories:
1. accept_passive - Brief agreement
2. accept_active - Enthusiastic agreement with elaboration
3. correct_passive - Mild disagreement
4. correct_active - Strong disagreement with reasoning
5. clarify_passive - Basic clarification request
6. clarify_active - Detailed explanation request
7. disregard_passive - Polite deflection
8. disregard_active - Explicit rejection
9. unclear - Doesn't fit above

Respond with ONLY the category name."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower()
        except:
            return "unclear"
    
    def _is_clarification_request(self, user_input: str) -> bool:
        """Check if user is asking for clarification."""
        keywords = [
            "clarify", "explain", "rationale", "reasoning", "why", "how did you",
            "what do you mean", "elaborate", "tell me more"
        ]
        return any(kw in user_input.lower() for kw in keywords)
    
    def _wants_to_disengage(self, user_input: str) -> bool:
        """Check if user wants to disengage from current topic."""
        keywords = ["let's move on", "move on", "skip", "next topic", "something else"]
        return any(kw in user_input.lower() for kw in keywords)

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
    """Initialize OpenAI client."""
    if OpenAI is None:
        st.error("openai package not installed. Run: pip install openai")
        return None
    key = get_secret_then_env("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY.")
        return None
    return OpenAI(api_key=key)

def build_system_prompt() -> str:
    """Build system prompt for guided mode."""
    return (
        "You are TeamMait, a peer-support assistant for expert clinicians reviewing "
        "therapist performance in a transcript. Your scope is limited strictly to "
        "analyzing the therapist's observable skills in the transcript. "
        "Anchor every claim to the transcript (and provided docs). When citing the transcript, "
        "use the line numbers provided in the format [Line X]. If uncertain, say so briefly. "
        "Be succinct and academically neutral; do not use emojis. "
        "Never invent facts. Always cite specific line references when making claims about the transcript. "
        
        "\n\nYou are operating in GUIDED REVIEW mode. You present observations about the transcript "
        "and engage in natural discussion with the clinician. When they respond to your observations, "
        "engage conversationally and provide thoughtful analysis. "
        
        "\n\nCONVERSATION GUIDELINES:"
        "\n1. AVOID REPETITIVE PHRASES: NEVER start with 'It sounds like...' or 'It seems like...'. "
        "\n2. ANSWER DIRECT QUESTIONS: When users ask questions, answer them directly with substantive guidance."
        "\n3. SUBSTANTIVE ENGAGEMENT: Provide analysis and insights, not just questions."
        "\n4. Mimic the tone and style of the collaborative peer clinician you are conversing with."
    )

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text content from a DOCX file."""
    if Document is None:
        return f"[Error: python-docx not installed]"
    
    try:
        doc = Document(docx_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[Error: {str(e)}]"

def openai_complete(history, system_text, client, model_name="gpt-4o-mini", stream=False, max_tokens=512):
    """Complete a chat using OpenAI API."""
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

# ==================== RAG SETUP ====================

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
                    line_num = i + 1
                    numbered_turn = f"[Line {line_num}] {turn}"
                    documents.append(numbered_turn)
                    ids.append(f"ref_{i}")
    
    # Load supporting documents
    for txt_path in glob.glob(os.path.join(supporting_folder, "*.txt")):
        with open(txt_path, "r", encoding="utf-8") as f:
            documents.append(f.read())
            ids.append(f"supp_txt_{os.path.basename(txt_path)}")
    
    for json_path in glob.glob(os.path.join(supporting_folder, "*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    documents.append(str(item))
                    ids.append(f"supp_json_{os.path.basename(json_path)}_{i}")
    
    for docx_path in glob.glob(os.path.join(supporting_folder, "*.docx")):
        content = extract_text_from_docx(docx_path)
        if content and not content.startswith("[Error"):
            documents.append(content)
            ids.append(f"supp_docx_{os.path.basename(docx_path)}")
    
    # Seed collection if empty
    if collection.count() == 0 and documents:
        collection.add(documents=documents, ids=ids)
    
    return documents

def retrieve_context(query: str, n_results: int = 5) -> str:
    """Retrieve relevant context from ChromaDB."""
    _, collection = initialize_chroma()
    results = collection.query(query_texts=[query], n_results=n_results)
    
    retrieved_parts = []
    for docs in results.get("documents", []):
        retrieved_parts.extend(docs)
    
    return " ".join(retrieved_parts)

def load_reference_conversation():
    """Load the reference conversation for display."""
    ref_path = os.path.join("doc/RAG", "116_P8_conversation.json")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "full_conversation" in data:
                return data["full_conversation"]
    return []

def load_question_bank():
    """Load the question bank."""
    with open("doc/interaction_prompts/interaction_prompts.json", "r") as f:
        data = json.load(f)
    return data.get("feedback_items", [])

# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="Guided Interaction", page_icon="💬", layout="wide")

# Check login
if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.switch_page("Home.py")
    st.stop()

username = st.session_state["user_info"]["username"]

# Initialize OpenAI client
client = get_openai_client()

# Initialize state machine
if "state_machine" not in st.session_state:
    question_bank = load_question_bank()
    st.session_state.state_machine = GuidedInteractionStateMachine(question_bank, client)

# Initialize message history
if "guided_messages" not in st.session_state:
    st.session_state.guided_messages = [
        {
            "role": "assistant",
            "content": "Hi, my name is TeamMait. Feel free to ask me any questions about the referenced session transcript. It can be found in the left side panel.\n\nI've made a few observations about the session that we can discuss together. When you're ready, just ask me for the 'next question' and I'll share one with you.",
            "ts": now_ts()
        }
    ]

# Load RAG documents
load_rag_documents()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown(f"**Username:** {username}")
    
    # Completion status
    if "completion_status" not in st.session_state:
        st.session_state["completion_status"] = {}
    
    persistent_value = st.session_state["completion_status"].get("guided_interaction", False)
    st.session_state["include_guided_interaction"] = persistent_value

    def _on_include_guided_change():
        current_value = st.session_state.get("include_guided_interaction", False)
        st.session_state["completion_status"]["guided_interaction"] = current_value

    st.checkbox("Check this when done", key="include_guided_interaction", on_change=_on_include_guided_change)
    
    # Progress tracker
    st.markdown("### Progress")
    questions_asked = len(st.session_state.state_machine.questions_asked)
    st.progress(min(questions_asked / 4, 1.0))
    st.caption(f"{questions_asked} / 4 questions reviewed")
    
    # Show reference conversation
    with st.expander("Show Reference Conversation", expanded=True):
        ref_conversation = load_reference_conversation()
        if ref_conversation:
            for i, turn in enumerate(ref_conversation):
                line_num = i + 1
                is_client = turn.strip().startswith("Client: ")
                if is_client:
                    st.markdown(f"""
                    <div style="text-align: right; margin-left: 0%; padding: 10px; border-radius: 10px;">
                    <small style="color: #888; font-size: 0.8em;">[Line {line_num}]</small><br>
                    {turn}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-weight:600; font-size:1.08em; margin: 10px 0;'><small style='color: #888; font-size: 0.8em; font-weight: normal;'>[Line {line_num}]</small><br><em>{turn}</em></div>", unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

st.title("Guided Interaction")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Disclaimer: TeamMait may be incorrect or incomplete. Please verify information.</p>", unsafe_allow_html=True)

# Display chat history
for m in st.session_state.guided_messages:
    role = m["role"]
    role_label = "TeamMait" if role == "assistant" else "User"
    
    with st.chat_message(role):
        if "ts" in m:
            st.markdown(f"**{role_label}** • <small style='color: #888; font-size: 0.8em;'>*{m['ts']}*</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{role_label}**")
        st.markdown(m["content"])

# Response buttons
def handle_button_click(response_type: str):
    """Handle quick response button clicks."""
    st.session_state.state_machine.button_clicked = response_type
    st.rerun()

if st.session_state.state_machine.current_state == ChatbotState.QUESTION_PRESENTATION:
    st.markdown("##### Grade Response (optional):")
    cols = st.columns(2)
    
    button_clicked = st.session_state.state_machine.button_clicked
    
    with cols[0]:
        button_text = "👍 ✓" if button_clicked == "accept" else "👍"
        st.button(button_text, on_click=lambda: handle_button_click("accept"), use_container_width=True)
    with cols[1]:
        button_text = "👎 ✓" if button_clicked == "disregard" else "👎"
        st.button(button_text, on_click=lambda: handle_button_click("disregard"), use_container_width=True)

# Text input
if st.session_state.state_machine.current_state != ChatbotState.SESSION_COMPLETE:
    prompt = st.chat_input("Type your response or elaboration...")
    
    if prompt:
        # Add user message
        user_msg = {
            "role": "user",
            "content": prompt,
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(f"**User** • <small style='color: #888; font-size: 0.8em;'>*{user_msg['ts']}*</small>", unsafe_allow_html=True)
            st.markdown(prompt)
        
        # Process state transition
        transition = st.session_state.state_machine.transition(prompt)
        
        # Generate response
        if transition.use_llm:
            # Use LLM to generate response
            if transition.api_context:
                context = transition.api_context
            else:
                context = retrieve_context(prompt)
            
            system_prompt = build_system_prompt() + f"\n\nContext:\n{context}"
            
            with st.chat_message("assistant"):
                timestamp = now_ts()
                st.markdown(f"**TeamMait** • <small style='color: #888; font-size: 0.8em;'>*{timestamp}*</small>", unsafe_allow_html=True)
                
                placeholder = st.empty()
                acc = ""
                for chunk in openai_complete(
                    history=[{"role": "user", "content": prompt}],
                    system_text=system_prompt,
                    client=client,
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
                
        elif transition.bot_response:
            timestamp = now_ts()
            with st.chat_message("assistant"):
                st.markdown(f"**TeamMait** • <small style='color: #888; font-size: 0.8em;'>*{timestamp}*</small>", unsafe_allow_html=True)
                st.markdown(transition.bot_response)
                
            bot_msg = {
                "role": "assistant",
                "content": transition.bot_response,
                "ts": timestamp
            }
            st.session_state.guided_messages.append(bot_msg)
        
        # Clear button state
        st.session_state.state_machine.button_clicked = None
        
        st.rerun()

else:
    st.success("Guided interaction session complete!")
    if st.button("Start New Session"):
        question_bank = load_question_bank()
        st.session_state.state_machine = GuidedInteractionStateMachine(question_bank, client)
        st.session_state.guided_messages = [{
                "role": "assistant",
                "content": "Hi, my name is TeamMait. Feel free to ask me any questions about the referenced session transcript. It can be found in the left side panel.\n\nI've made a few observations about the session that we can discuss together. When you're ready, just ask me for the 'next question' and I'll share one with you.",
                "ts": now_ts()
            }
        ]
        st.rerun()