import streamlit as st
import time
from datetime import datetime

def safe_rerun():
    """Attempt to rerun the Streamlit app in a way that works across Streamlit versions.

    Try modern st.rerun() first, then fall back to experimental_rerun(), then session_state flag.
    """
    try:
        # Try modern st.rerun() first (available in Streamlit 1.18+)
        st.rerun()
        return
    except AttributeError:
        pass
    except Exception:
        pass
        
    try:
        # Fall back to experimental version for older Streamlit
        st.experimental_rerun()
        return
    except AttributeError:
        pass
    except Exception:
        pass
        
    # Last resort: toggle a session_state key to force a rerun without touching query params
    try:
        st.session_state["__safe_rerun_flag__"] = int(time.time())
        return
    except Exception:
        # if even this fails, there's nothing more we can do gracefully
        return

def debug_trace(key: str, value: any, page: str = "unknown"):
    """Add a timestamped entry to debug trace for session state changes."""
    if "debug_trace" not in st.session_state:
        st.session_state["debug_trace"] = []
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    entry = f"{timestamp} [{page}] {key} = {value}"
    st.session_state["debug_trace"].append(entry)
    
    # Keep only last 50 entries to avoid memory bloat
    if len(st.session_state["debug_trace"]) > 50:
        st.session_state["debug_trace"] = st.session_state["debug_trace"][-50:]

def debug_trace_existence(page: str = "unknown"):
    """Track which include_* keys exist in session state at page load."""
    if "debug_trace" not in st.session_state:
        st.session_state["debug_trace"] = []
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    keys_to_check = ["include_open_chat", "include_survey", "include_guided_interaction", "include_instructions_and_consent"]
    existing_keys = [k for k in keys_to_check if k in st.session_state]
    
    entry = f"{timestamp} [{page}] SESSION_STATE_KEYS: {existing_keys}"
    st.session_state["debug_trace"].append(entry)
    
    if len(st.session_state["debug_trace"]) > 50:
        st.session_state["debug_trace"] = st.session_state["debug_trace"][-50:]
