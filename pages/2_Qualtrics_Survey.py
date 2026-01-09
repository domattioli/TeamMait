import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Survey", page_icon="")

if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

username = st.session_state["user_info"]["username"]

st.title("Demographics Survey")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Please complete the external Qualtrics survey below and let your proctor know when you're finished.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;margin-top:6px;'><strong style='color:#6b7280;'>Privacy Reminder:</strong><span style='color:#e11d48;'> Please do not include any identifying information in your survey responses.</span></p>", unsafe_allow_html=True)

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Initialize survey link clicked state
if "survey_link_clicked" not in st.session_state:
    st.session_state.survey_link_clicked = False

# Sync the checkbox state with the persistent completion tracker
persistent_value = st.session_state["completion_status"].get("survey", False)
st.session_state["include_survey"] = persistent_value

if st.session_state.survey_link_clicked:
    # Show enabled checkbox + link on same line
    col1, col2 = st.columns([0.03, 0.97])
    with col1:
        checked = st.checkbox("", key="survey_completed_checkbox", 
                             value=st.session_state.get("include_survey", False),
                             label_visibility="collapsed")
    with col2:
        st.markdown(
            '<a href="https://pennstate.qualtrics.com/jfe/form/SV_0pPPg0tAmtv31si" target="_blank" '
            'style="font-size: 16px; color: #1a73e8; text-decoration: underline; line-height: 2.2;">'
            'Click here to take the Qualtrics survey</a> '
            '<span style="font-size: 12px; color: #059669;">(check when completed)</span>',
            unsafe_allow_html=True
        )
    if checked:
        st.session_state["include_survey"] = True
        st.session_state["completion_status"]["survey"] = True
else:
    # Show disabled checkbox + link, with button to confirm link was clicked
    col1, col2 = st.columns([0.03, 0.97])
    with col1:
        st.checkbox("", disabled=True, key="survey_disabled_checkbox", label_visibility="collapsed")
    with col2:
        st.markdown(
            '<a href="https://pennstate.qualtrics.com/jfe/form/SV_0pPPg0tAmtv31si" target="_blank" '
            'style="font-size: 16px; color: #1a73e8; text-decoration: underline; line-height: 2.2;">'
            'Click here to take the Qualtrics survey</a> '
            '<span style="font-size: 12px; color: #6b7280;">(click link to enable checkbox)</span>',
            unsafe_allow_html=True
        )
    
    st.markdown("")  # Spacer
    if st.button("I've opened the survey link", type="secondary"):
        st.session_state.survey_link_clicked = True
        st.rerun()

with st.sidebar:
    st.markdown(f"**Username:** {username}")
