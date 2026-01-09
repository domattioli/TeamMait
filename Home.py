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
    """Load valid users from Streamlit secrets"""
    try:
        users = st.secrets.get("credentials", {}).get("users", [])
        if not users:
            st.error("No credentials found in secrets.")
            return []
        return users
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return []

def validate_login(username, password):
    """Validate username and password against secrets"""
    # Allow test mode for demo purposes
    if username == "test mode" and password == "test":
        return True
    
    valid_users = load_valid_users()
    for user in valid_users:
        if user.get("username") == username and user.get("password") == password:
            return True
    return False

@st.dialog("Login", dismissible=False, width="small")
def get_user_details():
    st.markdown("### Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.button("Login", type="primary", use_container_width=True)
    
    if submit:
        if not username or not password:
            st.error("Username and password are required.")
            return
        
        if not validate_login(username, password):
            st.error("Invalid credentials. Please try again.")
            return
        
        # If test user, prompt for API key
        if username == "test mode" and password == "test":
            st.session_state.is_test_user = True
            st.session_state.user_info = {
                "username": username,
                "password": password,
                "consent_given": None,
                "consent_timestamp": None
            }
            st.session_state["username"] = username
            st.session_state["password"] = password
            st.rerun()
        else:
            st.session_state.is_test_user = False
            st.session_state.user_info = {
                "username": username,
                "password": password,
                "consent_given": None,
                "consent_timestamp": None
            }
            st.session_state["username"] = username
            st.session_state["password"] = password
            st.rerun()

# Handle test user API key prompt
if "is_test_user" not in st.session_state:
    st.session_state.is_test_user = False

if "user_info" not in st.session_state:
    get_user_details()
    st.stop()

# If test user and no API key provided yet, show prompt
if st.session_state.is_test_user and "test_api_key" not in st.session_state:
    st.info("ℹ️ You're running in **test mode**. You'll need to provide your own OpenAI API key to use Modules 1 and 2.")
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Your key will only be used for this session and will not be saved."
        )
        if st.form_submit_button("Save API Key", type="primary"):
            if not api_key_input or not api_key_input.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key (starts with 'sk-').")
            else:
                st.session_state.test_api_key = api_key_input
                st.rerun()

# If test user has API key, show option to reset it
if st.session_state.is_test_user and "test_api_key" in st.session_state:
    with st.expander("🔑 API Key Settings"):
        st.write("Your OpenAI API key is set.")
        if st.button("Reset API Key", type="secondary"):
            del st.session_state.test_api_key
            st.rerun()

# (sidebar navigation removed)

# ---------- Main Page Content ----------
st.title("TeamMait: Therapy Transcript Review Assistant")

st.markdown("""
<style>
    body { font-size: 16px; }
    h3, h4 { font-size: 18px; }
    p { font-size: 16px; line-height: 1.6; }
    li { font-size: 16px; line-height: 1.6; }
</style>

### Welcome!
- TeamMait is a peer-support assistant designed to help you review and analyze a therapy session transcript.
- Please read the consent form below before proceeding.

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
If you have questions about this research study, please contact the research team via domattioli@psu.edu.
""", unsafe_allow_html=True)

st.markdown("""
---

**By checking the consent box, you acknowledge that you have read and understood this information and agree to participate in this research study.**
""")

# Consent checkbox
consent = st.checkbox("I have read and agree to the consent form above.", key="consent_checkbox")

if consent:
    st.session_state.user_info["consent_given"] = True
    st.session_state.user_info["consent_timestamp"] = datetime.now().isoformat()
    st.markdown(
        "<p style='font-size: 20px; font-weight: bold; color: #059669; margin-top: 16px;'>"
        "✓ Click the <strong>Module 1</strong> tab in the left sidebar to continue."
        "</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='font-size: 18px; color: #6b7280; margin-top: 16px;'>"
        "Please check the consent box above to continue."
        "</p>",
        unsafe_allow_html=True
    )

# Collapsible instructions panel (always visible)
with st.expander("📋 Instructions (click to expand)"):
    st.markdown("""
### Study Overview
- You will be **role-playing** as a clinical supervisor evaluating therapist performance for two Prolonged Exposure (PE) therapy sessions.
- For each of the two modules you can find a PE session transcript in the left side panel of the page.
- Feel free to ask TeamMait any questions about the transcript and the therapist that may come to mind.
- Between the two modules will be a brief Qualtrics survey about your experience in psychology and with AI.
- The entire study will last approximately 55 minutes.
    - Your proctor will keep track of your time to ensure you stay within the study limits.
                
#### Module 1 ~ 15-20 minutes
- Ask TeamMait any questions about the therapy transcript, therapist behavior, demonstratic clinical skill, or therapuetic process.
- **Natural Conversation**: TeamMait responds when you initiate questions.
- Use TeamMait however feels natural for your evaluation process
- Feel free to request justification or supporting evidence for anything TeamMait offers to you.

#### Qualtrics Survey ~ 5 minutes
        
#### Module 2 ~ 20 minutes
- TeamMait will share prepared observations about notable aspects of the transcript.
- Progress through observations at your own pace.
- Again, feel free to request TeamMait to expand on anything offers to you.
        
#### Qualitative Interview ~ 10-15 minutes
- Complete a short interview to explore your experience with TeamMait.
    """)

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