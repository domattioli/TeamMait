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

# ---------- Main Page Content ----------
st.title("TeamMait: Therapy Transcript Review Assistant")

st.markdown("""
## Welcome!

- TeamMait is a peer-support assistant designed to help you review and analyze a therapy session transcript.
- You will be **role-playing as a clinical supervisor evaluating therapist performance for two Prolonged Exposure (PE) therapy sessions.

### Instructions
- For each of the two modules you can find a transcript between a unique therapist and their unique patient in the left side panel of that page.**

#### Module 1
- Ask TeamMait any questions about the therapy transcript, therapist behavior,  demonstratic clinical skill, or therapuetic process.
- **Natural Conversation**: TeamMait responds when you initiate questions - there are no predetermined prompts.
- Use TeamMait however feels natural for your evaluation process
- Feel free to request justification or supporting evidence for anything TeamMait offers to you.
- This should take approximately 15-20 minutes.

#### Qualtrics Survey
- Complete a survey about your experience in the field and with TeamMait Module 1.
- This should take approximately 5 minutes.
            
#### Module 2  
- TeamMait will share prepared observations about notable aspects of the transcript.
- Progress through observations at your own pace, with options to explore topics further depth or breadth.
- Again, feel free to requiest supporting evidence for anything TeamMait offers to you.
- This should take approximately 20-25 minutes.
            
#### Qualitative Interview
- Complete a brief interview with the proctor to examine your mindset during Module 2.
- This should take approximately 10 minutes.
---

---

# Research Consent & Privacy Notice

#### Study Purpose
This research study investigates how AI can enhance clinical supervision and training in therapy settings. Your participation will help us understand the effectiveness of AI-assisted tools for reviewing therapy session transcripts and supporting clinical decision-making.

#### Data Collection & Privacy
- All conversations with TeamMait are recorded via Zoom for research analysis. Transcripts of your conversations with TeamMait, and your survey and interview answers, will be saved and anonymized..
- Your identity will be removed from all data used in research publications or presentations. Zoom recordings will be deleted after the collected transcripts of your conversations with TeamMait have been verified and stored securely.
- All data is stored securely on Penn State University servers and will only be accessed by authorized research personnel.
- No clinical information will be shared during this study.
- Please do not include identifying information about yourself or others in your responses or discussions.

#### Your Rights
- Your participation is completely voluntary
- You may discontinue participation at any time without explanation or penalty
- You may ask questions about the study at any point
- While there are no direct benefits to you, your participation contributes to advancing clinical training tools

#### Contact Information
If you have questions about this research study, please contact the research team through your institutional channels.
""")

st.markdown("""
---

**By checking the consent box, you acknowledge that you have read and understood this information and agree to participate in this research study.**
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


# Preload Module 2 data in background when user is on Home page
if st.session_state.username:
    try:
        from utils.module_preload import preload_all_modules
        preload_all_modules()
    except:
        pass  # Preload is optional