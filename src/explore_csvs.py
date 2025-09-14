"""
Quick inspection script for the processed Formula 1 dataset.

This is just to check what the processed file looks like
before moving on to embeddings and retrieval.
"""

import pandas as pd
import os

# Path to the processed file created by preprocess_data.py
processed_path = os.path.join("data", "processed", "processed_races.csv")

# Loading the dataset into a pandas DataFrame
df = pd.read_csv(processed_path)

# Printing the first 5 rows to get a sense of the structure
print(df.head())

# Show the list of column names
print("\nColumns:", df.columns.tolist())

# Show the total number of documents (rows)
print("\nNumber of documents:", len(df))
