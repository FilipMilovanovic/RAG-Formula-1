import os
import argparse
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Setup and config
load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct-v0.3")

PROCESSED_PATH = os.path.join("data", "processed", "processed_races.csv")
INDEX_PATH = os.path.join("models", "f1_faiss.index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # same model used for indexing

client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# Helpers 
def load_data_and_index():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Missing {PROCESSED_PATH}. Run src/preprocess_data.py first."
        )
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Missing {INDEX_PATH}. Run src/build_embeddings.py first."
        )
    df = pd.read_csv(PROCESSED_PATH)
    index = faiss.read_index(INDEX_PATH)
    return df, index


def embed_query(query: str) -> np.ndarray:
    vec = embedder.encode([query])
    vec = np.asarray(vec, dtype = "float32")
    faiss.normalize_L2(vec)  #normalizing a query vector for cosine search
    return vec

def retrieve_top_k(df: pd.DataFrame, index: faiss.Index, query: str, k: int = 5):
    q = embed_query(query)
    D, I = index.search(q, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(df):
            hits.append({"idx": int(idx), "score": float(score), "text": df.iloc[idx]["text"]})
    return hits


def maybe_prioritize_winner(hits, query_lower: str):
    """
    If the question asks 'who won', keep only entries with 'Finished position: 1'.
    If none found, fall back to original hits.
    """
    if "who won" in query_lower:
        winners = []
        for h in hits:
            t = h["text"]
            if "Finished position: 1" in t or "Finished position: 1," in t:
                winners.append(h)
        return winners if winners else hits
    return hits



def trim_contexts(contexts: list[str], max_chars: int = 2200) -> list[str]:
    """
    Keep adding snippets until we hit a soft character budget.
    Helps local models keep prompt small & fast.
    """
    out, total = [], 0
    for c in contexts:
        if total + len(c) + 4 > max_chars:
            break
        out.append(c)
        total += len(c) + 4
    return out


def build_user_prompt(question: str, contexts: list[str]) -> str:
    """
    LM Studio's Mistral template supports only `user` and `assistant`.
    We put instructions + context + question in a single user message.
    """
    bullet_context = "\n".join(f"- {c}" for c in contexts)
    return (
        "You answer Formula 1 questions strictly using the provided CONTEXT.\n"
        "Rules:\n"
        "1) Use ONLY facts present in CONTEXT. Do not invent.\n"
        "2) If the answer is not in CONTEXT, reply exactly: \"I don't know based on the provided context.\"\n"
        "3) Be concise and precise.\n"
        "4) If asking for a race winner, return just the winner's full name and team.\n\n"
        f"CONTEXT:\n{bullet_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer:"
    )


def generate_answer(question: str, contexts: list[str], temperature: float = 0.0, max_tokens: int = 256) -> str:
    prompt = build_user_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# CLI 
def main():
    parser = argparse.ArgumentParser(description="Retrieve + generate an answer using a local LLM via LM Studio.")
    parser.add_argument("--question", "-q", type=str, required=False,
                        help='e.g., "Who won the 2008 Australian Grand Prix?"')
    parser.add_argument("--top_k", type=int, default=10, help="How many snippets to pass as context.")
    parser.add_argument("--max_context_chars", type=int, default=2200, help="Soft cap on context size.")
    args = parser.parse_args()

    question = args.question or input("Enter your question: ").strip()
    df, index = load_data_and_index()

    # Retrieve
    raw_hits = retrieve_top_k(df, index, question, k=args.top_k)
    hits = maybe_prioritize_winner(raw_hits, question.lower())

    # Build context
    contexts = [h["text"] for h in hits]
    contexts = trim_contexts(contexts, max_chars=args.max_context_chars)

    # Generate
    answer = generate_answer(question, contexts, temperature=0.0, max_tokens=256)

    # Output
    print("\n=== QUESTION ===")
    print(question)
    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES (Top retrieved) ===")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. [score {h['score']:.4f}] {h['text']}")


if __name__ == "__main__":
    main()
