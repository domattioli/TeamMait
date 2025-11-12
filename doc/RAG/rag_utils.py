import os
import chromadb
from openai import OpenAI
import streamlit as st


try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)



def openai_embedding_function(texts):
    """Return embeddings using the new OpenAI API."""
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]



class CustomEmbeddingFunction:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, input):
        return self.fn(input)

    def embed_documents(self, input):
        return self.fn(input)

    def embed_query(self, input):
        return self.fn(input)

    def name(self):
        return "custom_openai_embedding"


embedding_fn = CustomEmbeddingFunction(openai_embedding_function)


CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="therapy_transcripts",
    embedding_function=embedding_fn
)



def load_local_transcripts(folder_path="doc/transcripts"):
    """Reads all .txt transcripts from local folder."""
    texts = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                texts[file] = f.read()
    return texts



def index_transcripts(folder_path="doc/transcripts"):
    """Embeds and indexes transcripts into Chroma."""
    transcripts = load_local_transcripts(folder_path)
    if not transcripts:
        print("⚠️  No transcripts found in", folder_path)
        return

    for file_name, text in transcripts.items():
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        print(f"📄 Indexed {file_name} ({len(chunks)} chunks)")
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
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer, context
