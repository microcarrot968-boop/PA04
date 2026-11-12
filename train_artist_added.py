import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Load data and split sets
tf.keras.utils.set_random_seed(42)

df = pd.read_csv("results/baseline/data_splits.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]
labels = {"2000s": 0, "2010s": 1, "2020s": 2}

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

# 2. Standardize audio features
scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_val = scaler.transform(val[features])
X_test = scaler.transform(test[features])

# 3. One-hot encode artist names
train_artists = pd.get_dummies(train["artist_name"], dtype=int)
artist_columns = train_artists.columns.tolist()

val_artists = pd.get_dummies(val["artist_name"], dtype=int)
test_artists = pd.get_dummies(test["artist_name"], dtype=int)

val_artists = val_artists.reindex(columns=artist_columns, fill_value=0)
test_artists = test_artists.reindex(columns=artist_columns, fill_value=0)

X_train = np.hstack([X_train, train_artists.values]).astype("float32")
X_val = np.hstack([X_val, val_artists.values]).astype("float32")
X_test = np.hstack([X_test, test_artists.values]).astype("float32")

y_train = train["era"].map(labels)
y_val = val["era"].map(labels)
y_test = test["era"].map(labels)

# 4. Build classification model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
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

print("\n=== Artist Added Results ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("Confusion Matrix:")
print(cm)

# 7. Save model and results
Path("models/artist_added").mkdir(parents=True, exist_ok=True)
Path("results/artist_added").mkdir(parents=True, exist_ok=True)

model.save("models/artist_added/artist_added.keras")
joblib.dump(scaler, "models/artist_added/scaler.pkl")

with open("models/artist_added/artist_columns.json", "w") as f:
    json.dump(artist_columns, f)

pd.DataFrame(history.history).to_csv(
    "results/artist_added/training_history.csv", index=False
)

pd.DataFrame(
    cm,
    index=["2000s", "2010s", "2020s"],
    columns=["2000s", "2010s", "2020s"]
).to_csv("results/artist_added/confusion_matrix.csv")

pd.DataFrame([{
    "accuracy": accuracy,
    "macro_f1": macro_f1
}]).to_csv("results/artist_added/metrics.csv", index=False)

print("\nArtist Added model and results saved!")
