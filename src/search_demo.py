import pandas as pd
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


"""
Simple demo script for testing FAISS retrieval.
Loads the processed dataset and FAISS index, embeds an example query
and prints the top retrieved results with similarity scores.
"""

# Paths to processed data and the FAISS index
processed_folder = "data/processed"
models_folder = "models"

# Loading processed dataset and FAISS index
df = pd.read_csv(os.path.join(processed_folder, "processed_races.csv"))
index = faiss.read_index(os.path.join(models_folder, "f1_faiss.index"))

# Loading the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example query to test the search
query = "Who won the 2008 Australian Grand Prix?"
print(f"Query: {query}")

# Embedding the query and searching the closest five FAISS indexes
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
D, I = index.search(query_embedding, k=5)

# Print the results with their similarity scores
print("\nTop results:")
for idx, score in zip(I[0], D[0]):
    print(f"Score: {score:.4f} | Text: {df.iloc[idx]['text']}")
