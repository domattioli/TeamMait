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
    """Load valid users from credentials.json file"""
    try:
        import os
        from pathlib import Path
        
        # Try the simplest relative path first
        if os.path.exists("doc/credentials.json"):
            with open("doc/credentials.json", "r") as f:
                data = json.load(f)
                return data.get("users", [])
        
        # Fallback: try Path-based approach
        script_dir = Path(__file__).parent.resolve()
        creds_file = script_dir / "doc" / "credentials.json"
        
        if creds_file.exists():
            with open(creds_file, "r") as f:
                data = json.load(f)
                return data.get("users", [])
        
        st.error(f"Credentials file not found. Checked: doc/credentials.json and {creds_file}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in credentials file: {e}")
        return []
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
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
    st.markdown("### Login Required")
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", type="password", value="secureAdminPass123!")
    submit = st.button("Login", type="primary", use_container_width=True)
    
    if submit:
        if not username or not password:
            st.error("Username and password are required.")
            return
        
        if not validate_login(username, password):
            st.error("Invalid credentials. Please try again.")
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
### Welcome!
- TeamMait is a peer-support assistant designed to help you review and analyze a therapy session transcript.
- Please read the instructions **and** consent form below before proceeding.
            
# Instructions
- You will be **role-playing** as a clinical supervisor evaluating therapist performance for two Prolonged Exposure (PE) therapy sessions.
- For each of the two modules you can find a PE session transcript in the left side panel of the page.**
- Feel free to ask TeamMait any questions about the transcript and the therapist that may come to mind.
- Between the two modules will be a brief Qualtrics survey about your experience in psychology and with AI.

#### Module 1               ~ 15-20 minutes
- Ask TeamMait any questions about the therapy transcript, therapist behavior, demonstratic clinical skill, or therapuetic process.
- **Natural Conversation**: TeamMait responds when you initiate questions.
- Use TeamMait however feels natural for your evaluation process
- Feel free to request justification or supporting evidence for anything TeamMait offers to you.

#### Qualtrics Survey       ~ 5 minutes
            
#### Module 2               ~ 20 minutes
- TeamMait will share prepared observations about notable aspects of the transcript.
- Progress through observations at your own pace.
- Again, feel free to request TeamMait to expand on anything offers to you.
            
#### Qualitative Interview  ~ 10-15 minutes
- Complete a short interview to explore your experience with TeamMait.
---

---

# Research Consent & Privacy Notice

#### Study Purpose
This research study investigates how AI can enhance clinical supervision and training in therapy settings. Your participation will help us understand the effectiveness of AI-assisted tools for reviewing therapy session transcripts and supporting clinical decision-making.

#### Data Collection & Privacy
- Transcripts and survey/interview responses will be anonymized and stored securely on secured Penn State servers (authorized personnel only). Your identity will be anonymized from all data prior to analysis.
- Your interactions with TeamMait will be recorded via Zoom; recordings will be deleted after transcripts are verified and archived.
- Please avoid including identifying information about yourself or others in your messages.

#### Your Rights
- Participation is completely voluntary; you may discontinue at any time without penalty.
- You may ask any questions to the proctor at any point.
- Your participation contributes to advancing clinical training tools
            
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