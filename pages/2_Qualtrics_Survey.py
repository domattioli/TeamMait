import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Survey", page_icon="")

if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

username = st.session_state["user_info"]["username"]

st.title("Post-Session Survey")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Please complete the external Qualtrics survey and then mark this page as done before proceeding.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;margin-top:6px;'><strong style='color:#6b7280;'>Privacy Reminder:</strong><span style='color:#e11d48;'> Please do not include any identifying information in your survey responses.</span></p>", unsafe_allow_html=True)

st.markdown("[Click here to take the Qualtrics survey](https://pennstate.qualtrics.com/jfe/form/SV_0pPPg0tAmtv31si)" )

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Sync the checkbox state with the persistent completion tracker
persistent_value = st.session_state["completion_status"].get("survey", False)
st.session_state["include_survey"] = persistent_value

with st.sidebar:
    st.markdown(f"**Username:** {username}")

    # Checkbox removed - completion status tracked automatically
