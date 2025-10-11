import os
import re
import argparse
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

"""
Structured QA pipeline for Formula 1 queries.

Improved version:
- Filters FAISS hits by year and GP name to avoid mixing results across seasons.
- Expands winner question detection ("who won", "winner", "won the").
- Keeps CSV lookup as first and fully accurate fallback.
"""

# Load environment variables
load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL_NAME      = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct-v0.3")

PROCESSED_PATH  = os.path.join("data", "processed", "processed_races.csv")
RAW_PATH        = os.path.join("data", "raw")
INDEX_PATH      = os.path.join("models", "f1_faiss.index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

client   = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# --------------------------- HELPERS ---------------------------

def load_data_and_index():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError("Missing processed data. Run preprocess_data.py first.")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Missing FAISS index. Run build_embeddings.py first.")
    df = pd.read_csv(PROCESSED_PATH)
    index = faiss.read_index(INDEX_PATH)
    return df, index


def embed_query(query: str) -> np.ndarray:
    vec = embedder.encode([query])
    vec = np.asarray(vec, dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve_top_k(df: pd.DataFrame, index: faiss.Index, query: str, k: int = 10):
    q = embed_query(query)
    D, I = index.search(q, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(df):
            hits.append({"idx": int(idx), "score": float(score), "text": df.iloc[idx]["text"]})
    return hits


def filter_by_year_and_gp(hits, query_lower):
    """
    Filters FAISS hits to match the year and GP name mentioned in the query.
    Prevents mixing results across different seasons.
    """
    match = re.search(r"(\d{4}) ([a-zA-Z\s]+?) grand prix", query_lower, re.IGNORECASE)
    if not match:
        return hits

    year, gp_name = match.groups()
    filtered = [
        h for h in hits
        if year in h["text"].lower() and gp_name.strip().lower() in h["text"].lower()
    ]
    return filtered if filtered else hits


# --------------------------- WINNER DETECTION ---------------------------

def try_answer_winner_question(hits, question_lower):
    """
    Extracts P1 driver and team directly from FAISS hits when possible.
    """
    if any(kw in question_lower for kw in ["who won", "winner", "won the"]):
        for h in hits:
            text = h["text"]
            if "Finished position: 1" in text:
                parts = text.split(". ")
                driver_team_part = parts[1]
                driver = driver_team_part.split(" (")[0]
                team = driver_team_part.split("drove for ")[1].replace(".", "").strip()
                return {"driver": driver, "team": team}
    return None


def try_answer_winner_question_from_csv(question_lower):
    """
    Direct structured lookup from raw CSVs (always accurate).
    """
    if not any(kw in question_lower for kw in ["who won", "winner", "won the"]):
        return None

    match = re.search(r"(\d{4}) ([a-zA-Z\s]+?) grand prix", question_lower, re.IGNORECASE)
    if not match:
        return None

    year, gp_name = int(match.group(1)), match.group(2).strip()

    results = pd.read_csv(os.path.join(RAW_PATH, "results.csv"))
    races = pd.read_csv(os.path.join(RAW_PATH, "races.csv"))
    drivers = pd.read_csv(os.path.join(RAW_PATH, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(RAW_PATH, "constructors.csv"))

    race_row = races[
        (races["year"] == year)
        & (races["name"].str.lower().str.contains(gp_name.lower()))
    ]
    if race_row.empty:
        return None

    race_id = race_row.iloc[0]["raceId"]

    winner_row = results[
        (results["raceId"] == race_id)
        & (results["positionOrder"] == 1)
    ]
    if winner_row.empty:
        return None

    driver_id = winner_row.iloc[0]["driverId"]
    constructor_id = winner_row.iloc[0]["constructorId"]

    driver_row = drivers[drivers["driverId"] == driver_id]
    constructor_row = constructors[constructors["constructorId"] == constructor_id]

    if driver_row.empty or constructor_row.empty:
        return None

    driver_name = f"{driver_row.iloc[0]['forename']} {driver_row.iloc[0]['surname']}"
    team_name = constructor_row.iloc[0]["name"]

    return {"driver": driver_name, "team": team_name}


# --------------------------- LLM STRUCTURED OUTPUT ---------------------------

def build_structured_prompt(question: str, contexts: list[str]) -> str:
    bullet_context = "\n".join(f"- {c}" for c in contexts)
    return (
        "You are an assistant that answers Formula 1 questions using ONLY the provided CONTEXT.\n"
        "If the question is about a race winner, return the answer strictly as JSON with keys 'driver' and 'team'.\n"
        "Do not add explanations, do not add text outside JSON.\n\n"
        f"CONTEXT:\n{bullet_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Your response must be valid JSON."
    )


def generate_structured_answer(question: str, contexts: list[str]) -> str:
    prompt = build_structured_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# --------------------------- MAIN PIPELINE ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Structured QA over Formula 1 dataset.")
    parser.add_argument("--question", "-q", type=str, required=True,
                        help='e.g., "Who won the 2022 British Grand Prix?"')
    parser.add_argument("--top_k", type=int, default=15,
                        help="How many snippets to pass as context.")
    args = parser.parse_args()

    question = args.question
    question_lower = question.lower()

    # Step 1 - Try CSV lookup first (guaranteed accurate)
    csv_answer = try_answer_winner_question_from_csv(question_lower)
    if csv_answer:
        print("\n=== QUESTION ===")
        print(question)
        print("\n=== ANSWER (CSV LOOKUP) ===")
        print(csv_answer)
        return

    # Step 2 - FAISS retrieval + filtering
    df, index = load_data_and_index()
    hits = retrieve_top_k(df, index, question, k=args.top_k)
    hits = filter_by_year_and_gp(hits, question_lower)
    contexts = [h["text"] for h in hits]

    # Step 3 - Try direct FAISS extraction
    winner_json = try_answer_winner_question(hits, question_lower)
    if winner_json:
        print("\n=== QUESTION ===")
        print(question)
        print("\n=== ANSWER (FAISS TEXT) ===")
        print(winner_json)
    else:
        # Step 4 - LLM fallback
        answer = generate_structured_answer(question, contexts)
        print("\n=== QUESTION ===")
        print(question)
        print("\n=== ANSWER (LLM) ===")
        print(answer)

    # Step 5 - Show retrieved sources
    print("\n=== SOURCES ===")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. [score {h['score']:.4f}] {h['text']}")


if __name__ == "__main__":
    main()
