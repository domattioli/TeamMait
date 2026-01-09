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

# Survey link
st.markdown(
    '<a href="https://pennstate.qualtrics.com/jfe/form/SV_0pPPg0tAmtv31si" target="_blank" '
    'style="font-size: 18px; color: #1a73e8; text-decoration: underline;">'
    'Click here to take the Qualtrics survey</a>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(f"**Username:** {username}")
