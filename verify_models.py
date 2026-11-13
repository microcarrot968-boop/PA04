import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Load test data and define features
baseline = pd.read_csv("results/baseline/data_splits.csv")
disjoint = pd.read_csv("results/artist_disjoint/data_splits.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]
labels = {"2000s": 0, "2010s": 1, "2020s": 2}

# 2. Define paths for each experiment
experiments = {
    "baseline": (
        "models/baseline/baseline.keras",
        "models/baseline/baseline_scaler.pkl",
        baseline
    ),
    "artist_added": (
        "models/artist_added/artist_added.keras",
        "models/artist_added/scaler.pkl",
        baseline
    ),
    "artist_disjoint": (
        "models/artist_disjoint/artist_disjoint.keras",
        "models/artist_disjoint/scaler.pkl",
        disjoint
    ),
    "undersampling": (
        "models/undersampling/undersampling.keras",
        "models/undersampling/scaler.pkl",
        baseline
    )
}

# 3. Load each model and prepare test inputs
for name, (model_path, scaler_path, df) in experiments.items():
    print(f"\n=== {name} ===")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    test = df[df["split"] == "test"]
    X_test = scaler.transform(test[features])

    # 4. Restore artist encoding for RQ1
    if name == "artist_added":
        with open("models/artist_added/artist_columns.json") as f:
            artist_columns = json.load(f)

        artists = pd.get_dummies(test["artist_name"], dtype=int)
        artists = artists.reindex(columns=artist_columns, fill_value=0)

        X_test = np.hstack([X_test, artists.values]).astype("float32")

    # 5. Predict and calculate metrics
    assert X_test.shape[1] == model.input_shape[-1]

    y_test = test["era"].map(labels)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # 6. Compare with previously saved results
    result_path = f"results/{name}/"
    metric_file = "baseline_metrics.csv" if name == "baseline" else "metrics.csv"

    saved = pd.read_csv(result_path + metric_file).iloc[0]
    saved_cm = pd.read_csv(
        result_path + "confusion_matrix.csv", index_col=0
    ).values

    print(f"Test samples: {len(test)}")
    print(f"Input features: {X_test.shape[1]}")
    print(f"Accuracy: {accuracy:.4f} / Saved: {saved['accuracy']:.4f}")
    print(f"Macro F1: {macro_f1:.4f} / Saved: {saved['macro_f1']:.4f}")

    assert np.isclose(accuracy, saved["accuracy"], atol=1e-6)
    assert np.isclose(macro_f1, saved["macro_f1"], atol=1e-6)
    assert np.array_equal(cm, saved_cm)

    print("PASS: Saved results reproduced!")

print("\n=== All models verified successfully! ===")
