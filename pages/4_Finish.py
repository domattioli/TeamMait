import streamlit as st
import json
from datetime import datetime
import json as _json
from oauth2client.service_account import ServiceAccountCredentials
import gspread

st.set_page_config(page_title="Finish Final", page_icon="")

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
    ("Module 2", "include_guided_interaction")
]

# Initialize completion status if it doesn't exist
if "completion_status" not in st.session_state:
    st.session_state["completion_status"] = {}

# Map the items to their completion status keys
completion_mapping = {
    "Instructions & Consent (Home)": "home",
    "Module 1": "open_chat", 
    "Survey 1": "survey",
    "Module 2": "guided_interaction"
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
            "version": "2.0",  # Updated version for enhanced export
        },
        "completion_tracking": {
            "completion_status": st.session_state.get("completion_status", {}),
            "checkbox_events": st.session_state.get("debug_trace", []),  # Captures when checkboxes were checked
        },
        "parts": {}
    }

    # Enhanced Open Chat export with all message details
    if st.session_state.get("include_open_chat", False):
        open_chat_messages = st.session_state.get("messages", [])
        enhanced_messages = []
        
        for msg in open_chat_messages:
            enhanced_msg = {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("ts"),
                "display_name": msg.get("display_name", msg.get("role")),
                "message_type": "open_chat",
                "interaction_mode": "free_form"
            }
            enhanced_messages.append(enhanced_msg)
        
        export["parts"]["open_chat"] = {
            "messages": enhanced_messages,
            "total_messages": len(enhanced_messages),
            "conversation_metadata": {
                "start_time": enhanced_messages[0]["timestamp"] if enhanced_messages else None,
                "end_time": enhanced_messages[-1]["timestamp"] if enhanced_messages else None,
                "user_messages": len([m for m in enhanced_messages if m["role"] == "user"]),
                "assistant_messages": len([m for m in enhanced_messages if m["role"] == "assistant"])
            }
        }

    # Survey data
    if st.session_state.get("include_survey", False):
        export["parts"]["survey"] = {
            "qualtrics_link": "https://www.qualtrics.com", 
            "included": True,
            "completion_timestamp": None  # Could be enhanced to track when survey was marked complete
        }

    # Enhanced Module 2 export with detailed classifications and state tracking
    if st.session_state.get("include_guided_interaction", False):
        guided_messages = st.session_state.get("guided_messages", [])
        flowchart_state = st.session_state.get("flowchart_state", {})
        
        enhanced_guided_messages = []
        
        for i, msg in enumerate(guided_messages):
            enhanced_msg = {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("ts"),
                "display_name": msg.get("display_name", msg.get("role")),
                "message_type": "guided_interaction",
                "sequence_number": i + 1
            }
            
            # Add state context for assistant messages
            if msg.get("role") == "assistant":
                # Determine the interaction state when this message was sent
                current_stage = flowchart_state.get("stage", "unknown")
                questions_asked_count = len(flowchart_state.get("questions_asked", []))
                current_question = flowchart_state.get("current_question", {})
                
                enhanced_msg["bot_context"] = {
                    "flowchart_stage": current_stage,
                    "questions_asked_count": questions_asked_count,
                    "current_question_id": current_question.get("id"),
                    "interaction_type": determine_interaction_type(current_stage, current_question),
                    "is_structured_prompt": current_stage == "prompt" and current_question,
                    "prompt_domain": current_question.get("id", "").split("_")[0] if current_question.get("id") else None
                }
            
            # Add classification data for user messages (from LLM response type detection)
            elif msg.get("role") == "user" and i < len(guided_messages) - 1:
                # Look for classification information stored in flowchart state
                enhanced_msg["user_response_analysis"] = {
                    "detected_response_type": flowchart_state.get("current_response_type"),
                    "button_clicked": flowchart_state.get("button_clicked"),
                    "needs_followup": flowchart_state.get("needs_followup", False)
                }
            
            enhanced_guided_messages.append(enhanced_msg)
        
        # Detailed flowchart state export
        enhanced_flowchart_state = {
            "final_stage": flowchart_state.get("stage"),
            "questions_asked": flowchart_state.get("questions_asked", []),
            "total_questions_asked": len(flowchart_state.get("questions_asked", [])),
            "current_question": flowchart_state.get("current_question"),
            "session_complete": flowchart_state.get("stage") == "complete",
            "response_classifications": {
                "final_response_type": flowchart_state.get("current_response_type"),
                "button_interactions": flowchart_state.get("button_clicked"),
                "followup_status": flowchart_state.get("needs_followup", False)
            }
        }
        
        export["parts"]["guided_interaction"] = {
            "messages": enhanced_guided_messages,
            "flowchart_state": enhanced_flowchart_state,
            "interaction_analytics": {
                "total_messages": len(enhanced_guided_messages),
                "user_messages": len([m for m in enhanced_guided_messages if m["role"] == "user"]),
                "assistant_messages": len([m for m in enhanced_guided_messages if m["role"] == "assistant"]),
                "structured_prompts_shown": len(flowchart_state.get("questions_asked", [])),
                "conversation_start": enhanced_guided_messages[0]["timestamp"] if enhanced_guided_messages else None,
                "conversation_end": enhanced_guided_messages[-1]["timestamp"] if enhanced_guided_messages else None
            }
        }

    # Enhanced Instructions & Consent tracking
    if st.session_state.get("include_instructions_and_consent", False):
        export["parts"]["instructions_and_consent"] = {
            "consent_given": st.session_state.get("consent_given", False),
            "consent_timestamp": None,  # Could be enhanced to track when consent was given
            "user_info": st.session_state.get("user_info", {}),
            "included": True
        }

    # Add comprehensive session state snapshot for debugging
    export["session_state_snapshot"] = {
        "all_completion_flags": {
            "include_open_chat": st.session_state.get("include_open_chat", False),
            "include_survey": st.session_state.get("include_survey", False),
            "include_guided_interaction": st.session_state.get("include_guided_interaction", False),
            "include_instructions_and_consent": st.session_state.get("include_instructions_and_consent", False)
        },
        "debug_trace_events": st.session_state.get("debug_trace", []),
        "export_metadata": {
            "total_session_state_keys": len(st.session_state.keys()),
            "session_state_keys": list(st.session_state.keys())
        }
    }

    export["disclaimer"] = "TeamMait may be incorrect or incomplete. Verify important clinical, legal, or safety-related information independently before acting."
    return session_name, json.dumps(export, indent=2)

def determine_interaction_type(stage, current_question):
    """Determine the type of interaction based on stage and question context."""
    if stage == "intro":
        return "introduction"
    elif stage == "prompt" and current_question:
        question_id = current_question.get("id", "")
        if "adherence" in question_id.lower():
            return "structured_prompt_adherence"
        elif "procedural" in question_id.lower():
            return "structured_prompt_procedural" 
        elif "relational" in question_id.lower():
            return "structured_prompt_relational"
        elif "structural" in question_id.lower():
            return "structured_prompt_structural"
        else:
            return "structured_prompt_other"
    elif stage == "open_discussion":
        return "open_discussion_post_prompt"
    elif stage == "complete":
        return "session_complete"
    else:
        return f"flowchart_stage_{stage}"

# Custom CSS for red save button
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #ff4b4b;
    border-color: #ff4b4b;
    color: white;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #ff2b2b;
    border-color: #ff2b2b;
    color: white;
}
div.stButton > button[kind="primary"]:active {
    background-color: #ff1b1b;
    border-color: #ff1b1b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

clicked = st.button("Finish & Exit Study", disabled=not all_checked, type="primary")
# To-Do: don't let them click more than once -- once its clicked, lock the entire website.

if clicked:
    if not all_checked:
        st.warning("Cannot exit: not all items are marked done.")
    else:
        session_name, json_data = build_export()
        st.download_button(label="Download consolidated JSON", data=json_data, file_name=f"{session_name}.json", mime="application/json")

        # Attempt to append to Google Sheets if configured
        try:
            creds = st.secrets.get("GOOGLE_CREDENTIALS")
            sheet_name = st.secrets.get("SHEET_NAME")
            if creds and sheet_name:
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
