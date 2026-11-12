import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset and set parameters
df = pd.read_csv("data/billboard/processed.csv").reset_index(drop=True)

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]
labels = {"2000s": 0, "2010s": 1, "2020s": 2}
seeds = [42, 43, 44, 45, 46]


# 2. Create artist-disjoint train/val/test splits
def artist_split(seed):
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(df, df["era"], df["artist_name"]))

    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.setdiff1d(
        np.arange(len(df)), np.concatenate([val_idx, test_idx])
    )

    return train_idx, val_idx, test_idx


# 3. Create random splits with identical era counts
def random_split(reference, seed):
    rng = np.random.default_rng(seed)
    result = [[], [], []]

    for era in labels:
        indices = np.where(df["era"].to_numpy() == era)[0]
        rng.shuffle(indices)

        counts = [
            np.sum(df.iloc[idx]["era"] == era)
            for idx in reference
        ]

        boundaries = np.cumsum(counts)[:-1]
        parts = np.split(indices, boundaries)

        for i in range(3):
            result[i].extend(parts[i])

    return [np.array(idx, dtype=int) for idx in result]


# 4. Train and evaluate the same MLP
def train_model(split, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    train_idx, val_idx, test_idx = split
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    test = df.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])

    y_train = train["era"].map(labels)
    y_val = val["era"].map(labels)
    y_test = test["era"].map(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(features),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "test_size": len(test),
        "test_2000s": (test["era"] == "2000s").sum(),
        "test_2010s": (test["era"] == "2010s").sum(),
        "test_2020s": (test["era"] == "2020s").sum()
    }


# 5. Repeat experiments using five random seeds
results = []

for seed in seeds:
    disjoint = artist_split(seed)
    random = random_split(disjoint, seed)

    # Verify identical class distributions in each split
    for a, b in zip(disjoint, random):
        assert df.iloc[a]["era"].value_counts().to_dict() == \
               df.iloc[b]["era"].value_counts().to_dict()

    # Verify no shared artists across disjoint splits
    artist_sets = [set(df.iloc[idx]["artist_name"]) for idx in disjoint]
    assert all(
        artist_sets[i].isdisjoint(artist_sets[j])
        for i in range(3) for j in range(i + 1, 3)
    )

    for name, split in [("Random", random), ("Artist Disjoint", disjoint)]:
        metrics = train_model(split, seed)
        results.append({"seed": seed, "split_type": name, **metrics})

        print(
            f"Seed {seed} | {name:15s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )


# 6. Save individual results and summary statistics
output = Path("results/artist_disjoint")
output.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
summary = results_df.groupby("split_type")[["accuracy", "macro_f1"]].agg(
    ["mean", "std"]
).round(4)

results_df.to_csv(output / "split_comparison.csv", index=False)
summary.to_csv(output / "split_comparison_summary.csv")

print("\n=== Final Comparison (Mean ± SD) ===")
print(summary)
print("\nResults saved!")
