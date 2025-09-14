import pandas as pd
import os

"""
Preprocess Formula 1 tabular data into short natural language documents.

Input
  data/raw/*.csv
    races, results, drivers, constructors, circuits

Output
  data/processed/processed_races.csv
    columns: raceId, driverId, text
    each row is a compact narrative for one driver in one race
"""



# Folders
raw_folder = "data/raw"
processed_folder = "data/processed"

# Loading the core CSV files we need for a first RAG pass



# races: year, race name, date, circuitId
races = pd.read_csv(os.path.join(raw_folder, "races.csv"))
# results: finishing position and points per driver per race
results = pd.read_csv(os.path.join(raw_folder, "results.csv"))
# drivers: driver names and nationality
drivers = pd.read_csv(os.path.join(raw_folder, "drivers.csv"))
# constructors: team names and nationality
constructors = pd.read_csv(os.path.join(raw_folder, "constructors.csv"))
# circuits: circuit name, location, country
circuits = pd.read_csv(os.path.join(raw_folder, "circuits.csv"))

# Joining all tables into a single DataFrame of race results with rich context
# Starting from results, then adding race info, driver info, constructor info, and circuit info
# Applying suffixes only when column names collide
# For example, both races and circuits have a column called 'name'
# After merging we will have 'name' from races, and 'name_circuit' from circuits
results = results.merge(races, on="raceId", how="left", suffixes=("", "_race"))
results = results.merge(drivers, on="driverId", how="left", suffixes=("", "_driver"))
results = results.merge(constructors, on="constructorId", how="left", suffixes=("", "_constructor"))
results = results.merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit"))

# Building a short narrative per driver per race
# We are keeping it concise and structured so sentence embeddings work well
documents = []
for _, row in results.iterrows():
    # race name comes from races as name
    # circuit name comes from circuits as name_circuit because of the merge suffix
    doc_text = (
        f"{row['year']} {row['name']} at {row['name_circuit']}, "
        f"{row['location']}, {row['country']}. "
        f"{row['forename']} {row['surname']} ({row['nationality']}) "
        f"drove for {row['name_constructor']}. "
        f"Finished position: {row['positionText']}, points: {row['points']}."
    )
    documents.append(
        {
            "raceId": row["raceId"],
            "driverId": row["driverId"],
            "text": doc_text,
        }
    )

# Saving the processed dataset that will be embedded in the next step
processed_df = pd.DataFrame(documents)
os.makedirs(processed_folder, exist_ok=True)
processed_df.to_csv(os.path.join(processed_folder, "processed_races.csv"), index=False)

print("Processed data saved to data/processed/processed_races.csv")