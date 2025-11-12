import streamlit as st
from rag_utils import generate_answer

st.title("TeamMait: Open Chat (RAG-Enhanced)")
st.caption("Ask TeamMait anything about the therapy transcript.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.chat_input("Ask your question about the transcript...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.spinner("Thinking..."):
        answer, context = generate_answer(user_query)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Optionally show retrieved evidence
with st.expander("Show retrieved transcript context"):
    st.text(context)
