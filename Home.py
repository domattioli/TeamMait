import streamlit as st
import json
from datetime import datetime
from pathlib import Path

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

## Consent document loading
CONSENT_PATH = Path("doc/consent_form.md")
if not CONSENT_PATH.exists():
    st.error("Consent form file not found. Please check doc/consent_form.md.")
    st.stop()
    def load_consent_text():
        consent_text = load_consent_text()

# try:
#     return CONSENT_PATH.read_text(encoding="utf-8")
# except FileNotFoundError:
#     st.error("Consent form file not found. Please check doc/consent_form.md.")
#     return 

# ---------- Main Page Content ----------
st.title("TeamMait: Therapy Transcript Review Assistant")
consent_text = load_consent_text()
st.markdown("""
## Welcome!

You are participating as an expert clinician evaluator in this research study. TeamMait is a peer-support assistant designed to help you review and analyze a therapy session transcript. This tool provides two modes of interaction to help you evaluate therapist performance in PE (Prolonged Exposure) therapy sessions.

**Please begin by reading the therapy session transcript, which you'll find in the sidebar of each interaction page.**

### Instructions

#### Phase 1: Open Chat
- **Interactive Review**: Ask TeamMait any questions about the therapy transcript, therapist behavior, or therapy processes
- **Natural Conversation**: TeamMait responds when you initiate questions - there are no predetermined prompts
- **Flexible Approach**: Use TeamMait however feels natural for your evaluation process
- **Evidence Available**: Feel free to request justification or supporting evidence for any observation, with specific transcript references

A brief survey about your experience will follow this phase.
            
#### Phase 2: Guided Interaction  
- **Structured Observations**: TeamMait will share prepared observations about notable aspects of the transcript
- **Natural Flow**: Progress through observations at your own pace, with options to explore topics further depth or breadth, or to discuss related topics.
- **Evidence Available**: Feel free to request justification or supporting evidence for any observation, with specific transcript references

---

# Research Consent & Privacy Notice
            
""") + st.markdown(consent_text) + st.markdown("""
---

**By checking the consent box below, you acknowledge that you have read and understood this information and agree to participate in this research study.**
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