from rag_utils import index_transcripts

if __name__ == "__main__":
    print("Initializing local RAG index...")
    index_transcripts("doc/transcripts")
    print("✅ All transcripts indexed successfully.")