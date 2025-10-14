import streamlit as st
import json
from datetime import datetime

st.set_page_config(
    page_title="TeamMait - Therapy Transcript Review",
    layout="wide"
)

# ---------- Login Dialog ----------
def load_valid_users():
    """Load valid users from users.json file"""
    try:
        with open("doc/users.json", "r") as f:
            data = json.load(f)
            return data.get("users", [])
    except FileNotFoundError:
        st.error("Users configuration file not found.")
        return []
    except json.JSONDecodeError:
        st.error("Invalid users configuration file.")
        return []

def validate_login(username, password):
    """Validate username and password against the users.json file"""
    valid_users = load_valid_users()
    for user in valid_users:
        if user.get("username") == username and user.get("password") == password:
            return True
    return False

@st.dialog("Login", dismissible=False, width="small")
def get_user_details():
    username = st.text_input("Username", value="test")
    password = st.text_input("Password", type="password", value="test")
    submit = st.button("Submit", type="primary")
    
    if submit:
        if not username or not password:
            st.warning("Please enter both a username and a password.")
            return
        
        if not validate_login(username, password):
            st.error("Invalid username or password. Please try again.")
            return
        
        st.session_state.user_info = {
            "username": username,
            "password": password,
            "consent_given": None,
            "consent_timestamp": None
        }
        st.session_state["username"] = username
        st.session_state["password"] = password
        st.rerun()

if "user_info" not in st.session_state:
    get_user_details()
    st.stop()

# (sidebar navigation removed)

# ---------- Main Page Content ----------
st.title("TeamMait: Therapy Transcript Review Assistant")

st.markdown("""
## Welcome!

TeamMait is a peer-support assistant designed to help expert clinicians review and analyze 
therapy session transcripts. This tool provides two modes of interaction to help you evaluate
therapist performance in PE (Prolonged Exposure) therapy sessions.

### Instructions

#### Phase 1: Open Chat
- **Interactive Review**: Ask TeamMait any questions about the therapy transcript, therapist behavior, or therapy processes
- **Natural Conversation**: TeamMait responds when you initiate questions - there are no predetermined prompts
- **Flexible Topics**: Ask about specific concepts (e.g., therapeutic alliance, pacing, exposure structure, empathy) or general impressions
- **Evidence-Based**: Request justification or supporting evidence from TeamMait for any responses, including specific line citations
- **No Wrong Questions**: Use TeamMait however feels natural for supervision - explore freely
- **Duration**: Approximately 20 minutes for interaction

#### Phase 2: Guided Interaction  
- **Structured Observations**: TeamMait will share prepared observations about notable aspects of the transcript
- **Your Choice of Response**: Respond however feels natural - ask for clarification, expand on ideas, disagree, or move on to the next observation
- **Evidence Available**: Request justification or supporting evidence for any observation, with specific transcript references
- **Natural Flow**: Progress through observations at your own pace, with options to explore topics in depth
- **Duration**: Approximately 20 minutes for interaction

Both phases will be followed by brief surveys about your experience (~10 minutes total).

---

### Privacy & Consent
            
**Research Participation Notice**: By proceeding, you acknowledge that:

- **Data Collection**: All interactions with TeamMait are logged for research purposes to improve AI-assisted clinical supervision
- **Anonymization**: Your responses will be anonymized in any publications or research outputs
- **Voluntary Participation**: You may discontinue participation at any time without penalty
- **Transcript Review**: You will be reviewing therapy session transcripts as part of this research study
- **Time Commitment**: Total participation time is approximately 50 minutes (40 minutes interaction + 10 minutes surveys)
- **Research Purpose**: This study aims to understand how AI can best support clinical supervision and training

**Your participation helps advance AI-assisted clinical supervision tools for the therapy community.**
""")

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Sync the consent checkbox with persistent storage
persistent_consent = st.session_state["completion_status"].get("consent_given", False)
st.session_state["consent_given"] = persistent_consent

def _on_consent_change():
    from utils.streamlit_compat import debug_trace
    # Update the persistent completion tracker when consent checkbox changes
    current_value = st.session_state.get("consent_given", False)
    st.session_state["completion_status"]["consent_given"] = current_value
    debug_trace("completion_status.consent_given", current_value, "Home")

# Consent checkbox
st.checkbox("I have read and agree to the consent form", key="consent_given", on_change=_on_consent_change)

# Sync the checkbox state with the persistent completion tracker
persistent_value = st.session_state["completion_status"].get("home", False)
st.session_state["include_instructions_and_consent"] = persistent_value

def _on_include_guided_change():
    from utils.streamlit_compat import debug_trace
    # Update the persistent completion tracker when checkbox changes
    current_value = st.session_state.get("include_instructions_and_consent", False)
    st.session_state["completion_status"]["home"] = current_value
    debug_trace("completion_status.home", current_value, "Home")

# Only enable the "Check this when done" checkbox if consent is given

if st.session_state.get("consent_given", False):
    st.checkbox("Check this when done", key="include_instructions_and_consent", on_change=_on_include_guided_change)
else:
    # Display a disabled checkbox with a message
    st.checkbox("Check this when done", key="include_instructions_and_consent", 
                disabled=True, on_change=_on_include_guided_change)
    st.info("Please read and agree to the consent form first.")



# ---------- Footer ----------
st.divider()
st.caption(f"Logged in as: **{st.session_state['username']}**")