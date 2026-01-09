"""
Module Preload Utility
Preloads data in the background when user is on Home page
to speed up Module 1 and Module 2 navigation
"""

import streamlit as st
import os
import json
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
        
        # Module 1 & 2 shared task - ChromaDB and embeddings (the slow part)
        futures.append(executor.submit(_preload_chromadb_and_embeddings))
        
        # Module 2 specific tasks
        futures.append(executor.submit(_preload_question_bank))
        
        # Preload reference conversations
        futures.append(executor.submit(_preload_reference_conversations))
        
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
                future.result(timeout=10)  # 10 second timeout per task
        except Exception as e:
            # Silent fail - these are just optimizations
            pass
    
    # Mark as preloaded
    st.session_state.all_modules_preloaded = True


def _preload_chromadb_and_embeddings():
    """
    Preload ChromaDB client and embedding function.
    This is the main source of slowness on first load.
    """
    if "chromadb_preloaded" not in st.session_state:
        try:
            # Import and initialize the embedding function (slow - downloads model if needed)
            from chromadb.utils import embedding_functions
            from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
            import chromadb
            
            embed_model = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
            
            # Store in session state for reuse
            st.session_state.preloaded_embedding_fn = embedding_fn
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path="./rag_store",
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
            st.session_state.preloaded_chroma_client = chroma_client
            
            # Get or create collection (warms up the index)
            collection = chroma_client.get_or_create_collection("therapy", embedding_function=embedding_fn)
            st.session_state.preloaded_collection = collection
            
            st.session_state.chromadb_preloaded = True
        except Exception as e:
            # Silent fail - will load normally when needed
            pass


def _preload_reference_conversations():
    """Preload the reference conversation JSON files for both modules"""
    if "reference_conversations_preloaded" not in st.session_state:
        try:
            doc_folder = "doc/RAG"
            
            # Module 1 conversation
            module1_ref = os.path.join(doc_folder, "116_P8_conversation.json")
            if os.path.exists(module1_ref):
                with open(module1_ref, "r", encoding="utf-8") as f:
                    st.session_state.module1_ref_conversation = json.load(f)
            
            # Module 2 conversation
            module2_ref = os.path.join(doc_folder, "281_P10_conversation.json")
            if os.path.exists(module2_ref):
                with open(module2_ref, "r", encoding="utf-8") as f:
                    st.session_state.module2_ref_conversation = json.load(f)
            
            st.session_state.reference_conversations_preloaded = True
        except Exception:
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