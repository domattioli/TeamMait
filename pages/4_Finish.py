import streamlit as st
import json
from datetime import datetime

st.set_page_config(page_title="Finish Final", page_icon="✅")

if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

username = st.session_state["user_info"]["username"]

# Sidebar
with st.sidebar:
    st.markdown(f"**Username:** {username}")

st.title("Finish")
st.markdown("<p style='font-size:12px;color:#6b7280;margin-top:6px;'>Confirm that each page is marked as done. The Save button will be enabled only when all items are checked.</p>", unsafe_allow_html=True)

# Note: bind the disabled checkboxes directly to the shared session_state keys
# so they reflect changes made on other pages immediately.
st.markdown("### Checklist")

# Define the set of inclusion flags found across the app (pages + Home)
inclusion_items = [
    ("Instructions & Consent (Home)", "include_instructions_and_consent"),
    ("Open Chat", "include_open_chat"),
    ("Survey", "include_survey"),
    ("Guided Interaction", "include_guided_interaction")
]

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Map the items to their completion status keys
completion_mapping = {
    "Instructions & Consent (Home)": "home",
    "Open Chat": "open_chat", 
    "Survey": "survey",
    "Guided Interaction": "guided_interaction"
}

# Show completion status using the persistent tracker
for label, _ in inclusion_items:
    status_key = completion_mapping[label]
    is_complete = st.session_state["completion_status"].get(status_key, False)
    if is_complete:
        st.markdown(f"✅ **{label}** - Complete")
    else:
        st.markdown(f"❌ **{label}** - Not complete")

# Check if all are complete using the persistent tracker
all_checked = all([
    st.session_state["completion_status"].get("home", False),
    st.session_state["completion_status"].get("open_chat", False),
    st.session_state["completion_status"].get("survey", False),
    st.session_state["completion_status"].get("guided_interaction", False)
])

if not all_checked:
    st.warning("All modules must be completed before saving. Visit each page, complete the task, and then click next to 'Check this when done'.")

def build_export():
    session_name = f"teammait_session_{username}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    export = {
        "metadata": {
            "app_name": "TeamMait Consolidated Export",
            "session_name": session_name,
            "username": username,
            "exported_at": datetime.now().isoformat(),
        },
        "parts": {}
    }

    # Read authoritative flags from session_state
    if st.session_state.get("include_open_chat", False):
        export["parts"]["open_chat"] = st.session_state.get("messages", [])
    if st.session_state.get("include_survey", False):
        export["parts"]["survey"] = {"qualtrics_link": "https://www.qualtrics.com", "included": True}
    if st.session_state.get("include_guided_interaction", False):
        export["parts"]["guided_interaction"] = {
            "messages": st.session_state.get("guided_messages", []),
            "flowchart_state": st.session_state.get("flowchart_state", {}),
        }
    if st.session_state.get("include_instructions_and_consent", False):
        export["parts"]["instructions_and_consent"] = {
            "consent_given": st.session_state.get("consent_given", False),
            "included": True
        }

    export["disclaimer"] = "TeamMait may be incorrect or incomplete. Verify important clinical, legal, or safety-related information independently before acting."
    return session_name, json.dumps(export, indent=2)

clicked = st.button("Save and export", disabled=not all_checked)
# To-Do: don't let them click more than once -- once its clicked, lock the entire website.

if clicked:
    if not all_checked:
        st.warning("Cannot save: not all items are marked done.")
    else:
        session_name, json_data = build_export()
        st.download_button(label="Download consolidated JSON", data=json_data, file_name=f"{session_name}.json", mime="application/json")

        # Attempt to append to Google Sheets if configured
        try:
            creds = st.secrets.get("GOOGLE_CREDENTIALS")
            sheet_name = st.secrets.get("SHEET_NAME")
            if creds and sheet_name:
                import json as _json
                from oauth2client.service_account import ServiceAccountCredentials
                import gspread
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds_dict = _json.loads(creds)
                creds_obj = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                gs = gspread.authorize(creds_obj)
                sheet = gs.open(sheet_name).sheet1
                sheet.append_row([json_data, datetime.now().isoformat()])
                st.success("Saved and appended to Google Sheet.")
            else:
                st.info("Saved locally (download available). Google Sheets not configured.")
        except Exception as e:
            st.info(f"Saved locally. Google Sheets append failed or not configured: {e}")
