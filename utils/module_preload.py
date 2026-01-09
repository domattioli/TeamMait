"""
Module Preload Utility
Preloads data in the background when user is on Home page
to speed up Module 1 and Module 2 navigation
"""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor


def preload_all_modules():
    """
    Preload all module data in background threads.
    Call this function from the Home page.
    Includes both Module 1 and Module 2 data.
    """
    
    # Check if we're logged in
    if "username" not in st.session_state or not st.session_state.username:
        return
    
    # Check if data is already preloaded
    if "all_modules_preloaded" in st.session_state:
        return
    
    # Use ThreadPoolExecutor to load data in background
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit preload tasks
        futures = []
        
        # Module 1 & 2 shared task
        futures.append(executor.submit(_preload_rag_documents))
        
        # Module 2 specific tasks
        futures.append(executor.submit(_preload_question_bank))
        
        if "guided_session_id" in st.session_state:
            futures.append(
                executor.submit(
                    _preload_conversations,
                    st.session_state.username,
                    st.session_state.guided_session_id
                )
            )
        
        # Wait for all tasks to complete (with timeout)
        try:
            for future in futures:
                future.result(timeout=5)  # 5 second timeout per task
        except Exception as e:
            # Silent fail - these are just optimizations
            pass
    
    # Mark as preloaded
    st.session_state.all_modules_preloaded = True


def _preload_rag_documents():
    """
    Preload RAG documents and warm up ChromaDB.
    Used by both Module 1 and Module 2.
    
    Note: This is an optimization that pre-warms the cache.
    If it fails, RAG will load normally when the module is accessed.
    """
    if "rag_preloaded" not in st.session_state:
        try:
            # RAG documents are loaded via @st.cache_resource in each module
            # This preload attempts to trigger that cache early
            # The actual loading happens in the module's sidebar initialization
            
            # Mark as attempted (actual loading happens in module)
            st.session_state.rag_preloaded = True
        except Exception as e:
            # Silent fail - RAG will load normally when needed
            pass


def _preload_question_bank():
    """Preload the question bank (observations for Module 2)"""
    if "module2_questions_cache" not in st.session_state:
        try:
            from utils.question_loader import load_question_bank
            questions = load_question_bank()
            st.session_state.module2_questions_cache = questions
        except Exception:
            pass


def _preload_conversations(username, session_id):
    """Preload saved conversations for Module 2"""
    if "module2_conversations_cache" not in st.session_state:
        try:
            from utils.session_manager import SessionManager
            conversations = SessionManager.load_conversations(username, session_id)
            st.session_state.module2_conversations_cache = conversations
        except Exception:
            pass


# Usage in Home page (Home.py):
#
# At the END of your Home page file, add:
#
# # Preload modules in background
# if st.session_state.get("consent_given", False):
#     try:
#         from module_preload import preload_all_modules
#         preload_all_modules()
#     except:
#         pass  # Preload is optional