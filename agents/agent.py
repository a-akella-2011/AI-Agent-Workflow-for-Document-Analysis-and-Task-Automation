#!/usr/bin/env python3
"""
Local Llama2 document QA + safe automation agent (llama-cpp-python backend).

This file implements a local LLM backend using llama-cpp-python. Configure via env:
- AGENT_MODE=local
- LLAMA_MODEL_PATH=/path/to/your/ggml-model.bin
- DOCUMENTS_DIR=documents
- SAFE_TOOL_CONFIRMATION=interactive|auto-deny|non-interactive-allow

Usage:
    pip install -r requirements-local.txt
    export LLAMA_MODEL_PATH=/path/to/ggml-model.bin
    export AGENT_MODE=local
    python agents/agent.py
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Embeddings + index
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Local Llama (llama-cpp-python)
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")
AGENT_MODE = os.environ.get("AGENT_MODE", "local").lower()
SAFE_TOOL_CONFIRMATION = os.environ.get("SAFE_TOOL_CONFIRMATION", "interactive").lower()
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

class SimpleRetriever:
    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.docs: List[str] = []

    def build_index(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts to index")
        self.docs = texts
        embs = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs.astype(np.float32))

    def query(self, q: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built")
        q_emb = self.embed_model.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({"text": self.docs[idx], "score": float(dist), "idx": int(idx)})
        return results

class LocalLlamaLLM:
    def __init__(self, model_path: str):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed. Install requirements-local.txt")
        if not model_path:
            raise RuntimeError("LLAMA_MODEL_PATH must be set for local mode")
        self.model = Llama(model_path=model_path)

    def chat(self, messages: List[Dict[str,str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        # messages is a list of dicts like role/content; we convert to a single prompt
        # We only send the user content and contexts; system messages can be prefixed.
        prompt_parts = []
        for m in messages:
            role = m.get('role')
            content = m.get('content','')
            if role == 'system':
                prompt_parts.append(f"[SYSTEM]\n{content}\n\n")
            elif role == 'user':
                prompt_parts.append(f"[USER]\n{content}\n\n")
            elif role == 'assistant':
                prompt_parts.append(f"[ASSISTANT]\n{content}\n\n")
        prompt = "\n".join(prompt_parts)
        # llama-cpp-python create() returns a dict with 'choices'
        resp = self.model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        # join all output content
        out_text = ""
        if isinstance(resp, dict):
            choices = resp.get('choices', [])
            for c in choices:
                out_text += c.get('text', '')
        else:
            # fallback: str
            out_text = str(resp)
        return out_text.strip()

class Agent:
    def __init__(self, retriever: SimpleRetriever, llm=None):
        self.retriever = retriever
        self.llm = llm

    def _compose_prompt(self, question: str, contexts: List[str]) -> List[Dict[str,str]]:
        system = (
            "You are a helpful, conservative assistant that answers questions using the provided document snippets. "
            "If the answer isn't supported by the snippets, say 'I don't know' and do NOT hallucinate."
        )
        context_text = "\n\n---\n\n".join(contexts) if contexts else ""
        user = f"Context:\n{context_text}\n\nQuestion: {question}\nProvide a short, factual answer and list which snippet indices you used."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def answer(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        hits = self.retriever.query(question, top_k=top_k)
        contexts = [h["text"] for h in hits]
        if self.llm:
            prompt = self._compose_prompt(question, contexts)
            answer_text = self.llm.chat(prompt)
        else:
            answer_text = " (LLM not configured) Relevant snippets:\n\n" + "\n\n---\n\n".join(contexts)
        return {"answer": answer_text, "contexts": hits}

# Safe tool layer (same as OpenAI variant)
def safe_confirm(prompt: str) -> bool:
    if SAFE_TOOL_CONFIRMATION == "auto-deny":
        return False
    if SAFE_TOOL_CONFIRMATION == "non-interactive-allow":
        return True
    resp = input(f"{prompt} (type YES to confirm): ").strip()
    return resp == "YES"

def create_issue_stub(title: str, body: str):
    print(f"\n[Tool] Create issue called with title={title!r}")
    if not safe_confirm("Confirm creating GitHub issue?"):
        print("[Tool] Aborted by user")
        return {"status": "aborted"}
    print("[Tool] (stub) Pretending to create issue...")
    return {"status": "ok", "url": "(stubbed)"}

def load_text_documents(path: str) -> List[str]:
    p = Path(path)
    texts = []
    for f in p.glob("**/*"):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            texts.append(f.read_text(encoding="utf-8"))
    return texts

def main():
    print(f"Starting agent in mode={AGENT_MODE!r}")
    docs = load_text_documents(DOCUMENTS_DIR)
    if not docs:
        print(f"No documents found in {DOCUMENTS_DIR!r}. Add .txt or .md files and try again.")
        sys.exit(1)

    retriever = SimpleRetriever()
    retriever.build_index(docs)
    llm = None

    if AGENT_MODE == "local":
        if Llama is None:
            print("llama-cpp-python package not installed. Install requirements-local.txt and try again.")
            sys.exit(1)
        if not LLAMA_MODEL_PATH:
            print("LLAMA_MODEL_PATH not set. Set it to your ggml model path and try again.")
            sys.exit(1)
        llm = LocalLlamaLLM(model_path=LLAMA_MODEL_PATH)
    elif AGENT_MODE == "openai":
        print("OpenAI mode requested, but this file is local-first. Set AGENT_MODE=local to use local Llama.")
        llm = None
    else:
        print("Unknown AGENT_MODE. Use 'local' or 'openai'.")
        sys.exit(1)

    agent = Agent(retriever, llm=llm)

    while True:
        q = input("\nAsk question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q.startswith("tool:"):
            _, rest = q.split(":", 1)
            parts = rest.split("|")
            if parts[0] == "create_issue":
                title = parts[1] if len(parts) > 1 else "No title"
                body = parts[2] if len(parts) > 2 else ""
                print(create_issue_stub(title, body))
            else:
                print("Unknown tool")
            continue

        out = agent.answer(q)
        print("\n=== Answer ===\n")
        print(out["answer"]) 
        print("\n=== Sources ===")
        for i, c in enumerate(out["contexts"], 1):
            snippet = c["text"][:300].replace("\n", " ")
            print(f"[{i}] score={c['score']:.4f} idx={c['idx']} snippet={snippet}...")
        if SAFE_TOOL_CONFIRMATION == "interactive":
            maybe = input("\nDo you want to run a tool based on this answer? (yes/no): ").strip().lower()
            if maybe in {"y", "yes"}:
                print("To run a tool, type 'tool:<toolname>|param1|param2' (e.g. tool:create_issue|Title|Body)")

if __name__ == "__main__":
    main()
