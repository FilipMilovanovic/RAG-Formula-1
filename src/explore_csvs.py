"""
Provides a quick inspection of the processed Formula 1 dataset.

Loads the processed file, prints the first few rows, shows column names
and reports the total number of documents. 
"""

import pandas as pd
import os

processed_path = os.path.join("data", "processed", "processed_races.csv")

df = pd.read_csv(processed_path)

print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nNumber of documents:", len(df))
