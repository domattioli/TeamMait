import streamlit as st
import json
from datetime import datetime
import json as _json
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from utils.session_manager import SessionManager

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

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

    # Module 1: Open Chat - include if messages exist (more than just the initial greeting)
    open_chat_messages = st.session_state.get("messages", [])
    if len(open_chat_messages) > 1:  # More than just the initial greeting
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

    # Survey - include if link was clicked
    if st.session_state.get("survey_link_clicked", False) or st.session_state.get("include_survey", False):
        export["modules"]["survey"] = {
            "completed": st.session_state.get("completion_status", {}).get("survey", False),
            "link_clicked": st.session_state.get("survey_link_clicked", False),
        }

    # Module 2: Guided Observations - include if conversations exist
    all_conversations = st.session_state.get("all_conversations", {})
    question_bank = st.session_state.get("question_bank", [])
    
    # Check if there are any messages in Module 2
    has_module2_data = any(
        len(all_conversations.get(i, [])) > 0 for i in range(len(question_bank))
    ) or len(all_conversations.get("open_chat", [])) > 0
    
    if has_module2_data:
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
        
        # Include open_chat from review phase if exists
        open_chat_messages = all_conversations.get("open_chat", [])
        if open_chat_messages:
            for msg in open_chat_messages:
                all_messages_flat.append(msg)
        
        # Overall Module 2 metrics
        overall_reply_metrics = calculate_reply_times(all_messages_flat)
        overall_engagement = calculate_engagement_metrics(all_messages_flat)
        
        guided_session_start = st.session_state.get("guided_session_start")
        
        export["modules"]["module_2_guided_observations"] = {
            "observations": observations_export,
            "open_chat": [{
                "sequence": i + 1,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("timestamp"),
                "word_count": len(msg.get("content", "").split()),
            } for i, msg in enumerate(open_chat_messages)] if open_chat_messages else [],
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

    return session_name, json.dumps(export, indent=2)


def build_participant_export():
    """Build a clean PDF export for participants - conversations only, no analytics."""
    session_name = f"teammait_transcript_{username}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    
    if not PDF_AVAILABLE:
        # Fallback to JSON if fpdf2 not available
        return session_name, None, "json"
    
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(31, 41, 55)  # Dark gray
    pdf.cell(0, 15, "TeamMait Session Transcript", ln=True, align="C")
    
    # Subtitle with date/time
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(107, 114, 128)  # Gray
    pdf.cell(0, 8, f"{datetime.now().strftime('%B %d, %Y')} at {datetime.now().strftime('%I:%M %p')}", ln=True, align="C")
    pdf.ln(10)
    
    # Divider line
    pdf.set_draw_color(229, 231, 235)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)
    
    has_content = False
    
    # Module 1: Open Chat
    open_chat_messages = st.session_state.get("messages", [])
    if len(open_chat_messages) > 1:
        has_content = True
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 10, "Module 1: Open Review", ln=True)
        pdf.ln(3)
        
        for msg in open_chat_messages:
            if msg.get("role") in ("user", "assistant"):
                speaker = "You" if msg.get("role") == "user" else "TeamMait"
                content = msg.get("content", "")
                
                # Speaker name
                if msg.get("role") == "user":
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(79, 70, 229)  # Indigo for user
                else:
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(16, 185, 129)  # Green for TeamMait
                
                pdf.cell(0, 6, speaker, ln=True)
                
                # Message content
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(55, 65, 81)
                pdf.multi_cell(0, 5, content)
                pdf.ln(4)
        
        pdf.ln(5)
    
    # Module 2: Guided Observations
    all_conversations = st.session_state.get("all_conversations", {})
    question_bank = st.session_state.get("question_bank", [])
    
    for obs_idx in range(len(question_bank)):
        obs_messages = all_conversations.get(obs_idx, [])
        obs_data = question_bank[obs_idx] if obs_idx < len(question_bank) else {}
        obs_title = obs_data.get("title", f"Observation {obs_idx + 1}")
        
        user_messages = [m for m in obs_messages if m.get("role") in ("user", "assistant")]
        if user_messages:
            has_content = True
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(31, 41, 55)
            pdf.cell(0, 10, f"Module 2 - Item {obs_idx + 1}: {obs_title}", ln=True)
            pdf.ln(3)
            
            for msg in obs_messages:
                if msg.get("role") in ("user", "assistant"):
                    speaker = "You" if msg.get("role") == "user" else "TeamMait"
                    content = msg.get("content", "")
                    
                    if msg.get("role") == "user":
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(79, 70, 229)
                    else:
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(16, 185, 129)
                    
                    pdf.cell(0, 6, speaker, ln=True)
                    
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(55, 65, 81)
                    pdf.multi_cell(0, 5, content)
                    pdf.ln(4)
            
            pdf.ln(5)
    
    # Open chat from review phase
    open_chat = all_conversations.get("open_chat", [])
    if open_chat:
        user_messages = [m for m in open_chat if m.get("role") in ("user", "assistant")]
        if user_messages:
            has_content = True
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(31, 41, 55)
            pdf.cell(0, 10, "Module 2 - Follow-up Discussion", ln=True)
            pdf.ln(3)
            
            for msg in open_chat:
                if msg.get("role") in ("user", "assistant"):
                    speaker = "You" if msg.get("role") == "user" else "TeamMait"
                    content = msg.get("content", "")
                    
                    if msg.get("role") == "user":
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(79, 70, 229)
                    else:
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(16, 185, 129)
                    
                    pdf.cell(0, 6, speaker, ln=True)
                    
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(55, 65, 81)
                    pdf.multi_cell(0, 5, content)
                    pdf.ln(4)
    
    if not has_content:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 10, "No conversations recorded.", ln=True, align="C")
    
    # Footer note
    pdf.ln(10)
    pdf.set_draw_color(229, 231, 235)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(156, 163, 175)
    pdf.multi_cell(0, 5, "This is a record of your conversation with TeamMait. Thank you for participating!", align="C")
    
    # Output PDF to bytes
    pdf_bytes = pdf.output()
    return session_name, pdf_bytes, "pdf"

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
    participant_name, participant_data, export_format = build_participant_export()
    
    # Only try Google Sheets for non-test users
    is_test_user = st.session_state.get("is_test_user", False)
    
    if is_test_user:
        # Test users get a download since they don't have Google Sheets
        st.download_button(
            label="Download Results", 
            data=json_data, 
            file_name=f"{session_name}.json", 
            mime="application/json"
        )
        st.info("✅ Test mode: Click above to download your results. (Google Sheets integration not available for test users)")
    else:
        # Real participants - save to Google Sheets
        try:
            creds = st.secrets.get("GOOGLE_CREDENTIALS")
            sheet_name = st.secrets.get("SHEET_NAME")
            if creds and sheet_name:
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds_dict = _json.loads(creds)
                creds_obj = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                gs = gspread.authorize(creds_obj)
                sheet = gs.open(sheet_name).sheet1
                sheet.append_row([json_data])
                st.success("Data successfully saved for research use.")
                
                # Offer clean transcript download to participants (PDF)
                if participant_data and export_format == "pdf":
                    st.download_button(
                        label="Download Your Conversation Transcript (PDF)",
                        data=participant_data,
                        file_name=f"{participant_name}.pdf",
                        mime="application/pdf",
                        help="Download a formatted PDF of your conversations with TeamMait"
                    )
                elif participant_data:
                    # Fallback to JSON if PDF unavailable
                    st.download_button(
                        label="Download Your Conversation Transcript",
                        data=participant_data,
                        file_name=f"{participant_name}.json",
                        mime="application/json",
                        help="Download a copy of your conversations with TeamMait"
                    )
            else:
                st.warning("Could not save to Google Sheets. Please notify your proctor.")
        except Exception as e:
            st.error(f"Error saving responses. Please notify your proctor. (Error: {e})")
