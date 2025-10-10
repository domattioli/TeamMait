import streamlit as st
import json
from datetime import datetime

st.set_page_config(
    page_title="TeamMait - Therapy Transcript Review",
    layout="wide"
)

# ---------- Login Dialog ----------
@st.dialog("Login", dismissible=False, width="small")
def get_user_details():
    username = st.text_input("Username")
    email = st.text_input("Email")
    submit = st.button("Submit", type="primary")
    
    if submit:
        if not username or not email:
            st.warning("Please enter both a username and an email.")
            return
        
        st.session_state.user_info = {
            "username": username,
            "email": email,
            "consent_given": None,
            "consent_timestamp": None
        }
        st.session_state["username"] = username
        st.session_state["email"] = email
        st.rerun()

if "user_info" not in st.session_state:
    get_user_details()
    st.stop()

# ---------- Main Page Content ----------
st.title("TeamMait: Therapy Transcript Review Assistant")

st.markdown("""
## Welcome!

TeamMait is a peer-support assistant designed to help expert clinicians review and analyze 
therapy session transcripts. This tool provides two modes of interaction:

### Instructions

#### Mode 1: Open-Ended Chat
- Ask any questions about the therapy transcript
- Request evidence and quotes from the session
- Explore clinical observations freely
- Export your conversation for later review

#### Mode 2: Guided Interactions (Structured)
- TBD instructions

---

### Privacy & Consent

By proceeding, you acknowledge that:
- All interactions are logged for research purposes
- Your responses will be anonymized in any publications
- You can export your data at any time
- You may discontinue participation without penalty

---
""")

consent = st.checkbox("I consent to participate in this research study")

st.markdown("""
Your consent was recorded at: `{}`
            
### Get Started
Select one of the two modes below to begin your session.
""".format(st.session_state.user_info.get("consent_timestamp", "N/A")))


# ---------- Navigation Buttons ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Open-Ended Chat")
    st.markdown("Explore the transcript freely with conversational AI assistance.")
    if st.button("Start Open Chat", type="primary", use_container_width=True):
        st.switch_page("pages/1_open_chat.py")

with col2:
    st.markdown("### Guided Interaction")
    st.markdown("Follow a structured review process with targeted questions.")
    if st.button("Start Guided Interaction", type="primary", use_container_width=True):
        st.switch_page("pages/2_guided_interaction.py")

# ---------- Footer ----------
st.divider()
st.caption(f"Logged in as: **{st.session_state['username']}** ({st.session_state['email']})")