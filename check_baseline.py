import pandas as pd
import matplotlib.pyplot as plt

# 1. Load baseline results
history = pd.read_csv("results/baseline/training_history.csv")
df = pd.read_csv("results/baseline/data_splits.csv")

# 2. Plot training curves
for metric in ["loss", "accuracy"]:
    plt.figure(figsize=(7, 4))
    plt.plot(history[metric], label="Train")
    plt.plot(history["val_" + metric], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Baseline {metric.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/baseline/{metric}_curve.png", dpi=200)
    plt.close()

# 3. Check duplicated songs
print("\n=== Duplicate Check ===")
print("Same title and artist:",
      df.duplicated(["title", "artist_name"]).sum())

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]

duplicates = df[df.duplicated(features, keep=False)].copy()
cross_split = duplicates.groupby(features)["split"].transform("nunique") > 1
leakage = duplicates[cross_split]

print("Identical audio feature rows:", len(duplicates))
print("Cross-split duplicate rows:", len(leakage))

leakage.to_csv("results/baseline/possible_leakage.csv", index=False)

# 4. Print training summary
print("\n=== Training Summary ===")
print("Total epochs:", len(history))
print("Best validation loss:", round(history["val_loss"].min(), 4))
print("Final train accuracy:", round(history["accuracy"].iloc[-1], 4))
print("Final validation accuracy:", round(history["val_accuracy"].iloc[-1], 4))
