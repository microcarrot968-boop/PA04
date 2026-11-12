import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Load data and set random seed
tf.keras.utils.set_random_seed(42)

df = pd.read_csv("data/billboard/processed.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]
labels = {"2000s": 0, "2010s": 1, "2020s": 2}

# 2. Split data by artist (60% train, 20% val, 20% test)
sgkf = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

folds = list(sgkf.split(df, df["era"], groups=df["artist_name"]))

test_idx = folds[0][1]
val_idx = folds[1][1]
train_idx = np.setdiff1d(
    np.arange(len(df)), np.concatenate([val_idx, test_idx])
)

train = df.iloc[train_idx]
val = df.iloc[val_idx]
test = df.iloc[test_idx]

print("\n=== Data Split ===")
print("Train:", len(train), "Val:", len(val), "Test:", len(test))
print("\nEra distribution:")
print(pd.crosstab(df["era"], [
    "train" if i in train_idx else
    "val" if i in val_idx else "test"
    for i in range(len(df))
]))

# 3. Standardize audio features
scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_val = scaler.transform(val[features])
X_test = scaler.transform(test[features])

y_train = train["era"].map(labels)
y_val = val["era"].map(labels)
y_test = test["era"].map(labels)

# 4. Build classification model
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

# 5. Train model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate on test set
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

print("\n=== Artist Disjoint Results ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("Confusion Matrix:")
print(cm)

# 7. Save model and results
Path("models/artist_disjoint").mkdir(parents=True, exist_ok=True)
Path("results/artist_disjoint").mkdir(parents=True, exist_ok=True)

model.save("models/artist_disjoint/artist_disjoint.keras")
joblib.dump(scaler, "models/artist_disjoint/scaler.pkl")

df["split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

df.to_csv("results/artist_disjoint/data_splits.csv", index=False)

pd.DataFrame(history.history).to_csv(
    "results/artist_disjoint/training_history.csv", index=False
)

pd.DataFrame(
    cm,
    index=["2000s", "2010s", "2020s"],
    columns=["2000s", "2010s", "2020s"]
).to_csv("results/artist_disjoint/confusion_matrix.csv")

pd.DataFrame([{
    "accuracy": accuracy,
    "macro_f1": macro_f1
}]).to_csv("results/artist_disjoint/metrics.csv", index=False)

print("\nArtist Disjoint model and results saved!")
