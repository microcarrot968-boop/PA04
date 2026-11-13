import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 1. Load saved data splits
baseline = pd.read_csv("results/baseline/data_splits.csv")
disjoint = pd.read_csv("results/artist_disjoint/data_splits.csv")
balanced = pd.read_csv("results/undersampling/balanced_train.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]

# 2. Check duplicates and artist separation
print("=== Data Verification ===")
print("Total songs:", len(baseline))
print("Duplicate audio rows:", baseline.duplicated(features).sum())

artists = [
    set(disjoint[disjoint["split"] == name]["artist_name"])
    for name in ["train", "val", "test"]
]

print("Artist overlap (train/val):", len(artists[0] & artists[1]))
print("Artist overlap (train/test):", len(artists[0] & artists[2]))
print("Artist overlap (val/test):", len(artists[1] & artists[2]))

# 3. Check dataset distributions
print("\n=== Baseline Split ===")
print(pd.crosstab(baseline["split"], baseline["era"]))

print("\n=== Artist Disjoint Split ===")
print(pd.crosstab(disjoint["split"], disjoint["era"]))

print("\n=== Undersampled Training Set ===")
print(balanced["era"].value_counts().sort_index())
print("All rows marked train:", (balanced["split"] == "train").all())

# 4. Recalculate metrics from saved confusion matrices
print("\n=== Saved Metrics Verification ===")

experiments = {
    "Baseline": "baseline",
    "Artist Added": "artist_added",
    "Artist Disjoint": "artist_disjoint",
    "Undersampling": "undersampling"
}

for name, folder in experiments.items():
    path = f"results/{folder}/"
    cm = pd.read_csv(path + "confusion_matrix.csv", index_col=0).values
    metric_file = "baseline_metrics.csv" if folder == "baseline" else "metrics.csv"
    saved = pd.read_csv(path + metric_file).iloc[0]

    accuracy = np.trace(cm) / cm.sum()

    # Compute macro F1 from confusion matrix
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(axis=0),
                          out=np.zeros(3), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1),
                       out=np.zeros(3), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=(precision + recall) != 0)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f} | Saved: {saved['accuracy']:.4f}")
    print(f"Macro F1: {f1.mean():.4f} | Saved: {saved['macro_f1']:.4f}")

    assert np.isclose(accuracy, saved["accuracy"])
    assert np.isclose(f1.mean(), saved["macro_f1"])

print("\nAll saved metrics verified!")
