import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import streamlit as st

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set up Chroma client and collection
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="therapy_transcripts",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
)

def load_local_transcripts(folder_path="doc/transcripts"):
    """Reads all .txt transcripts from local folder."""
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                texts[file] = f.read()
    return texts

def index_transcripts(folder_path="doc/transcripts"):
    """Embeds and indexes transcripts into Chroma."""
    transcripts = load_local_transcripts(folder_path)
    for file_name, text in transcripts.items():
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
    print("✅ Indexed transcripts into Chroma")

def retrieve_context(query, n_results=3):
    """Retrieves relevant transcript snippets."""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0] if results and results["documents"] else []
    return "\n\n".join(docs)

def generate_answer(query):
    """Generates grounded response using GPT-4o."""
    context = retrieve_context(query)
    prompt = f"""
You are an assistant helping a clinical evaluator analyze therapy transcripts.
Use the following context from the transcript to answer the question factually.

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, evidence-based answer grounded in the transcript.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer, context
