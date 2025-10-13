import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Survey", page_icon="")

if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

username = st.session_state["user_info"]["username"]

st.title("Post-Session Survey")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Please complete the external Qualtrics survey and then mark this page as done before proceeding.</p>", unsafe_allow_html=True)

st.markdown("[Take the Qualtrics survey](https://www.qualtrics.com)" )

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Sync the checkbox state with the persistent completion tracker
persistent_value = st.session_state["completion_status"].get("survey", False)
st.session_state["include_survey"] = persistent_value

with st.sidebar:
    st.markdown(f"**Username:** {username}")

    def _on_include_survey_change():
        from utils.streamlit_compat import debug_trace
        # Update the persistent completion tracker when checkbox changes
        current_value = st.session_state.get("include_survey", False)
        st.session_state["completion_status"]["survey"] = current_value
        debug_trace("completion_status.survey", current_value, "Survey")

    st.checkbox("Check this when done", key="include_survey", on_change=_on_include_survey_change)

st.markdown(f"Last updated: {datetime.now().isoformat()}")
