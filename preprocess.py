import pandas as pd
from pathlib import Path

# 1. Load datasets
data_dir = Path("data/billboard")

tracks = pd.read_csv(data_dir / "tracks.csv")
charts = pd.read_csv(data_dir / "chart_entries.csv")

# 2. Keep the earliest chart year for each song
charts = charts.sort_values("year")
charts = charts.drop_duplicates(subset=["title", "artist_name"])

# 3. Merge chart years with audio features
df = pd.merge(
    tracks,
    charts[["title", "artist_name", "year"]],
    on=["title", "artist_name"],
    how="inner"
)

# 4. Select features and remove missing values
features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]

df = df[["title", "artist_name", "year"] + features]
df = df.dropna(subset=features)

# Remove songs with identical audio features
df = df.drop_duplicates(subset=features, keep="first")

# Remove duplicate audio after sorting by chart year
df = df.sort_values("year")
df = df.drop_duplicates(subset=features, keep="first")

# 5. Create era labels
def get_era(year):
    if year < 2010:
        return "2000s"
    elif year < 2020:
        return "2010s"
    else:
        return "2020s"

df["era"] = df["year"].apply(get_era)

# 6. Save processed dataset
df.to_csv(data_dir / "processed.csv", index=False)

print("Total songs:", len(df))
print("\nSongs per era:")
print(df["era"].value_counts().sort_index())
print("\nSaved to:", data_dir / "processed.csv")
