import pandas as pd
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths to processed data and the FAISS index
processed_folder = "data/processed"
models_folder = "models"

# Load the processed dataset with race, driver, and constructor text
df = pd.read_csv(os.path.join(processed_folder, "processed_races.csv"))

# Load the pre-built FAISS index
index = faiss.read_index(os.path.join(models_folder, "f1_faiss.index"))

# Load the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example query to test the search
query = "Who won the 2008 Australian Grand Prix?"
print(f"Query: {query}")

# Convert the query into an embedding vector
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Search the FAISS index and return the top 5 closest matches
D, I = index.search(query_embedding, k=5)

# Print the results with their similarity scores
print("\nTop results:")
for idx, score in zip(I[0], D[0]):
    print(f"Score: {score:.4f} | Text: {df.iloc[idx]['text']}")
