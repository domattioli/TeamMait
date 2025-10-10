import streamlit as st
import time
from textwrap import dedent
from urllib.parse import quote
import json
from datetime import datetime
import os
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from sentence_transformers import SentenceTransformer
import sys
import glob

# ---------- SQLite shim for Chroma ----------
try:
    import os
    import sys
    import json
    import glob
    from typing import Optional

    # Optional SQLite shim used by some chromadb builds
    try:
        import pysqlite3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        pass

    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

    # Optional SDKs (load if installed)
    try:
        import anthropic
    except Exception:
        anthropic = None
    try:
        from openai import OpenAI
    except Exception:
        OpenAI = None


    def build_system_prompt() -> str:
        return (
            "You are TeamMait, a peer-support assistant for expert clinicians reviewing therapist performance in a transcript. "
            "Your scope is limited strictly to analyzing the therapist’s observable skills in the transcript. "
            "Anchor every claim to the transcript (and provided docs). If uncertain, say so briefly. "
            "Be succinct and academically neutral; do not use emojis. "
            "Engagement policy: Do not propose next steps or include calls-to-action unless asked. "
            "Clarifications: Only ask a single, decision-critical clarification question if strictly necessary. "
            "Never invent facts. Cite transcript line references; if no citation exists, say so."
        )


    def get_secret_then_env(name: str) -> str:
        return os.getenv(name, "")


    def get_anthropic_client():
        if anthropic is None:
            return None
        key = get_secret_then_env("ANTHROPIC_API_KEY")
        if not key:
            return None
        return anthropic.Anthropic(api_key=key)


    def get_openai_client():
        if OpenAI is None:
            print("openai package not installed. Install 'openai' package.")
            return None
        key = get_secret_then_env("OPENAI_API_KEY")
        if not key:
            print("Missing OPENAI_API_KEY environment variable.")
            return None
        return OpenAI(api_key=key)


    def to_anthropic_messages(history):
        converted = []
        for m in history:
            role = m.get("role")
            if role in ("user", "teammait"):
                converted.append({"role": role, "content": [{"type": "text", "text": m["content"]}]})
        return converted


    def claude_complete(history, system_text, model_name, stream: bool = False):
        client = get_anthropic_client()
        if client is None:
            return "" if not stream else iter(())
        MAX_TURNS = 40
        trimmed = history[-MAX_TURNS:]
        if stream:
            try:
                with client.messages.stream(
                    model=model_name,
                    system=system_text,
                    max_tokens=1024,
                    messages=to_anthropic_messages(trimmed),
                ) as events:
                    for text in events.text_stream:
                        yield text
            except Exception as e:
                yield f"\n[Error: {e}]"
        else:
            try:
                resp = client.messages.create(
                    model=model_name,
                    system=system_text,
                    max_tokens=1024,
                    messages=to_anthropic_messages(trimmed),
                )
                out = []
                for block in resp.content:
                    if block.type == "text":
                        out.append(block.text)
                return "".join(out).strip()
            except Exception as e:
                return f"[Error: {e}]"


    def openai_complete(history, system_text, model_name, stream: bool = False, max_tokens: int = 512):
        client = get_openai_client()
        if client is None:
            return "" if not stream else iter(())
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        for m in history:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        if stream:
            try:
                resp = client.chat.completions.create(model=model_name, messages=messages, stream=True, max_tokens=max_tokens, temperature=0.3)
                for chunk in resp:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            except Exception as e:
                yield f"\n[Error: {e}]"
        else:
            try:
                resp = client.chat.completions.create(model=model_name, messages=messages, max_tokens=max_tokens, temperature=0.3)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                return f"[Error: {e}]"


    # ---------- Chroma initialization & RAG documents ----------
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)

    chroma_client = chromadb.PersistentClient(
        path="./rag_store",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = chroma_client.get_or_create_collection("therapy", embedding_function=embedding_fn)


    def load_rag_documents():
        doc_folder = "doc/RAG"
        supporting_folder = os.path.join(doc_folder, "supporting_documents")
        documents = []
        ids = []
        # Load main reference conversation
        ref_path = os.path.join(doc_folder, "116_P8_conversation.json")
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "full_conversation" in data:
                    for i, turn in enumerate(data["full_conversation"]):
                        documents.append(str(turn))
                        ids.append(f"ref_{i}")
        # Load .txt and .json files from supporting_documents
        if os.path.isdir(supporting_folder):
            for txt_path in glob.glob(os.path.join(supporting_folder, "*.txt")):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    ids.append(f"supp_txt_{os.path.basename(txt_path)}")
            for json_path in glob.glob(os.path.join(supporting_folder, "*.json")):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            documents.append(str(item))
                            ids.append(f"supp_json_{os.path.basename(json_path)}_{i}")
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            documents.append(f"{k}: {v}")
                            ids.append(f"supp_json_{os.path.basename(json_path)}_{k}")
        # Seed collection if empty
        if collection.count() == 0 and documents:
            collection.add(documents=documents, ids=ids)
        return documents


    rag_documents = load_rag_documents()


    def query(prompt: str, model: str = "gpt-4o-mini", n_results: int = 5, include_evidence: bool = False, stream: bool = False) -> Optional[str]:
        """Run a RAG-augmented query and return the model reply (or stream it to stdout).

        - prompt: user prompt string
        - model: model name (gpt-* -> OpenAI client; otherwise Claude via anthropic)
        - n_results: number of retrieved docs from Chroma
        - include_evidence: print retrieved passages before answer
        - stream: if True and provider supports streaming, stream to stdout and return None
        """
        results = collection.query(query_texts=[prompt], n_results=n_results)
        retrieved_parts = []
        for docs in results.get("documents", []):
            retrieved_parts.extend(docs)

        context_parts = list(retrieved_parts)
        for doc in rag_documents:
            if doc not in context_parts:
                context_parts.append(doc)

        context = " ".join(context_parts)

        if include_evidence:
            print("\nEvidence (top retrieved):\n")
            for i, ev in enumerate(retrieved_parts, 1):
                print(f"[{i}] {ev}\n")

        system_prompt = build_system_prompt() + f"\n\nUse the following session context when answering:\n\n{context}"

        history = [{"role": "user", "content": prompt}]
        use_openai = model.startswith("gpt-")
        complete_fn = openai_complete if use_openai else claude_complete

        if stream:
            for chunk in complete_fn(history=history, system_text=system_prompt, model_name=model, stream=True):
                print(chunk, end="", flush=True)
            print()
            return None
        else:
            reply = complete_fn(history=history, system_text=system_prompt, model_name=model, stream=False)
            return reply


    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description="Query TeamMait RAG + Chat provider from CLI")
        parser.add_argument("-p", "--prompt", type=str, help="Prompt to send to the model", required=False)
        parser.add_argument("-m", "--model", type=str, default="gpt-4o-mini", help="Model name to use (gpt-* -> OpenAI, otherwise Claude)")
        parser.add_argument("-n", "--n_results", type=int, default=5, help="Number of retrieved RAG results to include")
        parser.add_argument("--evidence", action="store_true", help="Print retrieved evidence before the answer")
        parser.add_argument("--stream", action="store_true", help="Stream completions if supported by provider")
        args = parser.parse_args()

        if args.prompt:
            out = query(args.prompt, model=args.model, n_results=args.n_results, include_evidence=args.evidence, stream=args.stream)
            if out is not None:
                print("\n=== Reply ===\n")
                print(out)
        else:
            print("Enter query (Ctrl-D to finish):")
            user_input = sys.stdin.read().strip()
            if user_input:
                out = query(user_input, model=args.model, n_results=args.n_results, include_evidence=args.evidence, stream=args.stream)
                if out is not None:
                    print("\n=== Reply ===\n")
                    print(out)
                    </div>
