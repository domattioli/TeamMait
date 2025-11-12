import streamlit as st
from datetime import datetime
from doc.RAG.rag_utils import generate_answer  # uses your Chroma + GPT-4o pipeline


st.set_page_config(page_title="Chat (RAG)", layout="wide")


def now_ts():
    return datetime.now().strftime("%H:%M:%S")


@st.dialog("Login", dismissible=False, width="small")
def get_user_details():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.button("Submit", type="primary")
    if submit:
        if not username or not password:
            st.warning("Please enter both a username and password.")
            return
        st.session_state["user_info"] = {"username": username, "password": password}
        st.session_state["username"] = username
        st.rerun()

if "user_info" not in st.session_state:
    get_user_details()
    st.stop()

st.title("TeamMait: Chat (RAG)")
st.caption("Grounded in your local transcript data (retrieval-augmented)")


if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm TeamMait. I can reference your local therapy transcripts and supporting documents when responding.",
            "ts": now_ts(),
        }
    ]


for m in st.session_state["rag_messages"]:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['role'].capitalize()}** • *{m['ts']}*")
        st.markdown(m["content"])

query = st.chat_input("Ask about the transcript...")
if query:
    st.session_state["rag_messages"].append(
        {"role": "user", "content": query, "ts": now_ts()}
    )

    with st.chat_message("user"):
        st.markdown(f"**You** • *{now_ts()}*")
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            try:
                answer, context = generate_answer(query)
                st.markdown(answer)
            except Exception as e:
                answer = f"⚠️ Error retrieving response: {e}"
                context = ""
                st.error(answer)

    st.session_state["rag_messages"].append(
        {"role": "assistant", "content": answer, "ts": now_ts()}
    )


if query:
    with st.expander("View retrieved transcript context"):
        st.text(context if context else "No transcript context found.")
