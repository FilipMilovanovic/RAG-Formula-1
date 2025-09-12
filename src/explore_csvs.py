import pandas as pd
import os

# Putanja do processed fajla
processed_path = os.path.join("data", "processed", "processed_races.csv")

# Učitavanje
df = pd.read_csv(processed_path)

# Prvih 5 redova
print(df.head())

# Vidi kolone
print("\nKolone:", df.columns.tolist())

# Vidi koliko ima dokumenata
print("\nBroj redova:", len(df))
