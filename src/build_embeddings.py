"""
Build sentence embeddings for the processed Formula 1 dataset
and store them in a FAISS index for fast similarity search.

Input
  data/processed/processed_races.csv

Output
  models/f1_faiss.index  (FAISS index with embeddings)
"""

import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths for processed data and model outputs
processed_folder = "data/processed"
models_folder = "models"

# Load the processed dataset created in preprocess_data.py
df = pd.read_csv(os.path.join(processed_folder, "processed_races.csv"))

# Load a pre-trained SentenceTransformer model
# all-MiniLM-L6-v2 maps each text into a 384-dimensional vector
print("Loading embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for every text document
# The progress bar helps track long runs
print("Creating embeddings")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Convert to float32 as required by FAISS
embeddings = np.array(embeddings).astype("float32")

# Normalizing vectors 
faiss.normalize_L2(embeddings)
# Build a simple FAISS index with Innner Product for cosine similarity
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Save the FAISS index to disk so we can reload it later for search
os.makedirs(models_folder, exist_ok=True)
faiss.write_index(index, os.path.join(models_folder, "f1_faiss.index"))

print("FAISS index saved to models/f1_faiss.index")
