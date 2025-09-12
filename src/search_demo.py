import pandas as pd
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Folders
processed_folder = "data/processed"
models_folder = "models"

# Učitaj processed fajl
df = pd.read_csv(os.path.join(processed_folder, "processed_races.csv"))

# Učitaj FAISS index
index = faiss.read_index(os.path.join(models_folder, "f1_faiss.index"))

# Učitaj model za embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Primer upita
query = "Who won the 2008 Australian Grand Prix?"
print(f"Query: {query}")

# Kreiraj embedding upita
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Pretraga u FAISS (vrati top 5)
D, I = index.search(query_embedding, k=5)

print("\nTop results:")
for idx, score in zip(I[0], D[0]):
    print(f"Score: {score:.4f} | Text: {df.iloc[idx]['text']}")
