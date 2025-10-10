import streamlit as st
import sys
import os
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_setup import retrieve_context, load_reference_conversation
from utils.llm_clients import build_system_prompt, openai_complete
from utils.flowchart_logic import (
    initialize_flowchart_state,
    handle_flowchart_transition,
    format_prompt_message,
    get_next_question
)

# ---------- Setup ----------
st.set_page_config(page_title="Guided Review", page_icon="📋", layout="wide")

# Check login
if "user_info" not in st.session_state:
    st.warning("Please log in first.")
    st.switch_page("Home.py")
    st.stop()

username = st.session_state["user_info"]["username"]

# ---------- Helper ----------
def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

# ---------- Session State ----------
initialize_flowchart_state()

if "guided_messages" not in st.session_state:
    st.session_state.guided_messages = [
        {
            "role": "assistant",
            "content": "Welcome to the Guided Review session. Let's review a few observations of this transcript together. I'll present statements about the therapy session, and you can respond with:\n\n- **Accept**: You agree with the observation\n- **Correct**: You have a different perspective\n- **Clarify**: You want more information\n- **Disregard**: Not relevant or you want to skip\n\nReady to begin?",
            "ts": now_ts()
        }
    ]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"**Username:** {username}")
    st.button("🏠 Back to Home", on_click=lambda: st.switch_page("Home.py"))
    
    # Progress tracker
    st.markdown("### Progress")
    questions_asked = len(st.session_state.flowchart_state["questions_asked"])
    st.progress(min(questions_asked / 4, 1.0))
    st.caption(f"{questions_asked} / 4 questions reviewed")
    
    # Export
    with st.expander("Save Data", expanded=False):
        session_name = f"guided_review_{username}_{datetime.now().strftime('%Y%m%d')}"
        export_data = {
            "metadata": {
                "app_name": "TeamMait Guided Review",
                "username": username,
                "exported_at": datetime.now().isoformat(),
                "questions_reviewed": questions_asked
            },
            "messages": st.session_state.guided_messages,
            "flowchart_state": st.session_state.flowchart_state
        }
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="Export Session",
            data=json_data,
            file_name=f"{session_name}.json",
            mime="application/json",
            type="primary"
        )
    
    # Show reference conversation
    with st.expander("Show Reference Conversation", expanded=False):
        ref_conversation = load_reference_conversation()
        if ref_conversation:
            for turn in ref_conversation:
                is_client = turn.strip().startswith("Client: ")
                if is_client:
                    st.markdown(f"<div style='text-align: right;'>{turn}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<em>{turn}</em>")

# ---------- Main Content ----------
st.title("📋 Guided Review")

# Display chat history
for m in st.session_state.guided_messages:
    role = m["role"]
    with st.chat_message(role):
        if "ts" in m:
            st.caption(m["ts"])
        st.markdown(m["content"])

# ---------- Response Buttons ----------
def handle_button_click(response_type: str):
    """Handle quick response button clicks."""
    user_msg = {
        "role": "user",
        "content": response_type.capitalize(),
        "ts": now_ts()
    }
    st.session_state.guided_messages.append(user_msg)
    
    # Process the transition
    result = handle_flowchart_transition(response_type)
    st.session_state.flowchart_state["stage"] = result["next_stage"]
    
    if result.get("use_llm"):
        # Use LLM to generate response
        context, _ = retrieve_context(user_msg["content"])
        system_prompt = build_system_prompt(mode="guided") + f"\n\nContext:\n{context}"
        
        response = openai_complete(
            history=st.session_state.guided_messages,
            system_text=system_prompt,
            model_name="gpt-4o-mini",
            stream=False,
            max_tokens=512
        )
        
        bot_msg = {
            "role": "assistant",
            "content": response,
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(bot_msg)
        
        # Follow up with "anything else"
        followup_msg = {
            "role": "assistant",
            "content": "Anything else you would like to discuss about this topic before we move on?",
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(followup_msg)
    elif result["bot_response"]:
        bot_msg = {
            "role": "assistant",
            "content": result["bot_response"],
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(bot_msg)
    
    st.rerun()

# Show buttons only when in prompt stage
if st.session_state.flowchart_state["stage"] == "prompt":
    st.markdown("### Quick Response:")
    cols = st.columns(4)
    with cols[0]:
        st.button("✅ Accept", on_click=lambda: handle_button_click("accept"), use_container_width=True)
    with cols[1]:
        st.button("✏️ Correct", on_click=lambda: handle_button_click("correct"), use_container_width=True)
    with cols[2]:
        st.button("❓ Clarify", on_click=lambda: handle_button_click("clarify"), use_container_width=True)
    with cols[3]:
        st.button("⏭️ Disregard", on_click=lambda: handle_button_click("disregard"), use_container_width=True)

# ---------- Text Input ----------
if st.session_state.flowchart_state["stage"] != "complete":
    prompt = st.chat_input("Type your response or elaboration...")
    
    if prompt:
        user_msg = {
            "role": "user",
            "content": prompt,
            "ts": now_ts()
        }
        st.session_state.guided_messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process transition
        result = handle_flowchart_transition(prompt)
        st.session_state.flowchart_state["stage"] = result["next_stage"]
        
        if result.get("use_llm"):
            # Generate LLM response
            context, _ = retrieve_context(prompt)
            system_prompt = build_system_prompt(mode="guided") + f"\n\nContext:\n{context}"
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""
                for chunk in openai_complete(
                    history=st.session_state.guided_messages,
                    system_text=system_prompt,
                    model_name="gpt-4o-mini",
                    stream=True,
                    max_tokens=512
                ):
                    acc += chunk
                    placeholder.markdown(acc)
                
                bot_msg = {
                    "role": "assistant",
                    "content": acc.strip(),
                    "ts": now_ts()
                }
                st.session_state.guided_messages.append(bot_msg)
        elif result["bot_response"]:
            with st.chat_message("assistant"):
                st.markdown(result["bot_response"])
                
            bot_msg = {
                "role": "assistant",
                "content": result["bot_response"],
                "ts": now_ts()
            }
            st.session_state.guided_messages.append(bot_msg)
        
        st.rerun()

else:
    st.success("✅ Guided review session complete! You may export your data from the sidebar.")
    if st.button("Start New Session"):
        st.session_state.flowchart_state = {
            "stage": "intro",
            "questions_asked": [],
            "current_question": None,
            "current_response_type": None,
            "needs_followup": False,
            "all_domains_covered": False
        }
        st.session_state.guided_messages = [
            {
                "role": "assistant",
                "content": "Welcome back! Ready to review more observations?",
                "ts": now_ts()
            }
        ]
        st.rerun()