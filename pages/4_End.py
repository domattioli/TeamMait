import streamlit as st
import json
from datetime import datetime
import json as _json
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from utils.session_manager import SessionManager

st.set_page_config(page_title="Finish Final", page_icon="")

if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

username = st.session_state["user_info"]["username"]

# Auto-save all session data on End page load to prevent data loss
if "end_page_autosaved" not in st.session_state:
    try:
        # Save Module 1 conversations if they exist
        if "messages" in st.session_state and st.session_state.get("module1_session_id"):
            SessionManager.save_conversations(
                username,
                st.session_state.module1_session_id,
                {"module1": st.session_state.messages}
            )
        
        # Save Module 2 conversations if they exist
        if "all_conversations" in st.session_state and st.session_state.get("guided_session_id"):
            SessionManager.save_conversations(
                username,
                st.session_state.guided_session_id,
                st.session_state.all_conversations
            )
            # Also mark session as completed
            metadata = SessionManager.load_session_metadata(username, st.session_state.guided_session_id)
            if metadata:
                metadata["status"] = "completed"
                metadata["completed_at"] = datetime.now().isoformat()
                SessionManager.save_session_metadata(username, st.session_state.guided_session_id, metadata)
        
        st.session_state.end_page_autosaved = True
    except Exception:
        pass  # Silent fail - don't block user from manual save

# Sidebar
with st.sidebar:
    st.markdown(f"**Username:** {username}")

st.title("Finished?")

st.markdown(
    "<p style='font-size:24px; font-weight: bold; margin: 40px 0;'>"
    "Click the 'Save Responses' button below and then let your proctor know that you're done."
    "</p>",
    unsafe_allow_html=True
)

def calculate_reply_times(messages):
    """Calculate response time metrics between user and assistant messages."""
    reply_times = []
    first_response_time = None
    
    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]
        
        # Calculate time between user message and assistant response
        if prev_msg.get("role") == "user" and curr_msg.get("role") == "assistant":
            prev_ts = prev_msg.get("ts") or prev_msg.get("timestamp")
            curr_ts = curr_msg.get("ts") or curr_msg.get("timestamp")
            
            if prev_ts and curr_ts:
                try:
                    prev_dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00")) if isinstance(prev_ts, str) else prev_ts
                    curr_dt = datetime.fromisoformat(curr_ts.replace("Z", "+00:00")) if isinstance(curr_ts, str) else curr_ts
                    delta = (curr_dt - prev_dt).total_seconds()
                    if delta >= 0:
                        reply_times.append(delta)
                        if first_response_time is None:
                            first_response_time = delta
                except:
                    pass
    
    return {
        "first_response_time_seconds": first_response_time,
        "average_reply_time_seconds": sum(reply_times) / len(reply_times) if reply_times else None,
        "min_reply_time_seconds": min(reply_times) if reply_times else None,
        "max_reply_time_seconds": max(reply_times) if reply_times else None,
        "total_exchanges": len(reply_times)
    }


def calculate_engagement_metrics(messages):
    """Calculate user engagement metrics from message content."""
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    
    user_word_counts = [len(m.get("content", "").split()) for m in user_messages]
    assistant_word_counts = [len(m.get("content", "").split()) for m in assistant_messages]
    
    return {
        "user_message_count": len(user_messages),
        "assistant_message_count": len(assistant_messages),
        "user_total_words": sum(user_word_counts),
        "user_avg_words_per_message": sum(user_word_counts) / len(user_word_counts) if user_word_counts else 0,
        "assistant_total_words": sum(assistant_word_counts),
        "assistant_avg_words_per_message": sum(assistant_word_counts) / len(assistant_word_counts) if assistant_word_counts else 0,
        "conversation_turns": min(len(user_messages), len(assistant_messages))
    }


def build_export():
    session_id = st.session_state.get("open_session_id") or st.session_state.get("guided_session_id") or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    session_name = f"teammait_session_{username}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    export_timestamp = datetime.now()
    
    export = {
        "metadata": {
            "schema_version": "4.0",
            "app_name": "TeamMait",
            "session_id": session_id,
            "session_name": session_name,
            "username": username,
            "exported_at": export_timestamp.isoformat(),
            "timezone": "local",
            "export_timestamp_utc": export_timestamp.astimezone().isoformat() if export_timestamp.tzinfo else None,
        },
        "consent": {
            "given": st.session_state.get("user_info", {}).get("consent_given", False),
            "timestamp": st.session_state.get("user_info", {}).get("consent_timestamp"),
        },
        "modules": {}
    }

    # Module 1: Open Chat
    if st.session_state.get("include_open_chat", False):
        open_chat_messages = st.session_state.get("messages", [])
        enhanced_messages = []
        
        for i, msg in enumerate(open_chat_messages):
            enhanced_msg = {
                "sequence": i + 1,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("ts"),
                "word_count": len(msg.get("content", "").split()),
            }
            enhanced_messages.append(enhanced_msg)
        
        reply_metrics = calculate_reply_times(open_chat_messages)
        engagement_metrics = calculate_engagement_metrics(open_chat_messages)
        
        export["modules"]["module_1_open_chat"] = {
            "messages": enhanced_messages,
            "analytics": {
                "message_count": len(enhanced_messages),
                "start_time": enhanced_messages[0]["timestamp"] if enhanced_messages else None,
                "end_time": enhanced_messages[-1]["timestamp"] if enhanced_messages else None,
                "duration_seconds": None,  # Calculated if timestamps available
                **engagement_metrics,
                **reply_metrics,
            }
        }
        
        # Calculate duration if timestamps available
        if enhanced_messages and enhanced_messages[0]["timestamp"] and enhanced_messages[-1]["timestamp"]:
            try:
                start = datetime.fromisoformat(enhanced_messages[0]["timestamp"])
                end = datetime.fromisoformat(enhanced_messages[-1]["timestamp"])
                export["modules"]["module_1_open_chat"]["analytics"]["duration_seconds"] = (end - start).total_seconds()
            except:
                pass

    # Survey
    if st.session_state.get("include_survey", False):
        export["modules"]["survey"] = {
            "completed": True,
            "qualtrics_redirect": True,
        }

    # Module 2: Guided Observations
    if st.session_state.get("include_guided_interaction", False):
        # Get all conversations from the new structure
        all_conversations = st.session_state.get("all_conversations", {})
        question_bank = st.session_state.get("question_bank", [])
        
        observations_export = []
        all_messages_flat = []
        
        for obs_idx in range(len(question_bank)):
            obs_messages = all_conversations.get(obs_idx, [])
            obs_data = question_bank[obs_idx] if obs_idx < len(question_bank) else {}
            
            enhanced_obs_messages = []
            for i, msg in enumerate(obs_messages):
                enhanced_msg = {
                    "sequence": i + 1,
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp"),
                    "word_count": len(msg.get("content", "").split()),
                }
                enhanced_obs_messages.append(enhanced_msg)
                all_messages_flat.append(msg)
            
            obs_reply_metrics = calculate_reply_times(obs_messages)
            obs_engagement = calculate_engagement_metrics(obs_messages)
            
            observations_export.append({
                "observation_index": obs_idx,
                "observation_id": obs_data.get("id"),
                "observation_style": obs_data.get("style"),
                "observation_title": obs_data.get("title"),
                "messages": enhanced_obs_messages,
                "analytics": {
                    **obs_engagement,
                    **obs_reply_metrics,
                }
            })
        
        # Overall Module 2 metrics
        overall_reply_metrics = calculate_reply_times(all_messages_flat)
        overall_engagement = calculate_engagement_metrics(all_messages_flat)
        
        guided_session_start = st.session_state.get("guided_session_start")
        
        export["modules"]["module_2_guided_observations"] = {
            "observations": observations_export,
            "session_metadata": {
                "session_id": st.session_state.get("guided_session_id"),
                "phase": st.session_state.get("guided_phase"),
                "observations_completed": st.session_state.get("current_question_idx", 0),
                "total_observations": len(question_bank),
                "session_start": guided_session_start.isoformat() if guided_session_start else None,
            },
            "analytics": {
                "total_messages": len(all_messages_flat),
                **overall_engagement,
                **overall_reply_metrics,
            }
        }

    # Session summary
    export["session_summary"] = {
        "modules_completed": [k for k in export["modules"].keys()],
        "total_user_messages": sum(
            m.get("analytics", {}).get("user_message_count", 0) 
            for m in export["modules"].values() 
            if isinstance(m, dict) and "analytics" in m
        ),
        "total_assistant_messages": sum(
            m.get("analytics", {}).get("assistant_message_count", 0) 
            for m in export["modules"].values() 
            if isinstance(m, dict) and "analytics" in m
        ),
    }

    export["disclaimer"] = "TeamMait may be incorrect or incomplete. Verify important clinical, legal, or safety-related information independently before acting."
    return session_name, json.dumps(export, indent=2)

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

clicked = st.button("Save Responses", type="primary")

if clicked:
    session_name, json_data = build_export()
    st.download_button(label="Download consolidated JSON", data=json_data, file_name=f"{session_name}.json", mime="application/json")

    # Only try Google Sheets for non-test users
    is_test_user = st.session_state.get("is_test_user", False)
    
    if is_test_user:
        st.info("✅ Test mode: Data downloaded. (Google Sheets integration not available for test users)")
    else:
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
