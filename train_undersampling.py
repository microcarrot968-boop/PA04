import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# 1. Load the same dataset splits as the baseline
tf.keras.utils.set_random_seed(42)

df = pd.read_csv("results/baseline/data_splits.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]
labels = {"2000s": 0, "2010s": 1, "2020s": 2}
eras = list(labels.keys())

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

# 2. Undersample the training set only
print("\n=== Original Training Distribution ===")
print(train["era"].value_counts().sort_index())

min_count = train["era"].value_counts().min()

train_balanced = (
    train.groupby("era", group_keys=False)
    .sample(n=min_count, random_state=42)
    .sample(frac=1, random_state=42)
)

print("\n=== Balanced Training Distribution ===")
print(train_balanced["era"].value_counts().sort_index())
print(f"\nValidation: {len(val)} | Test: {len(test)} (unchanged)")

# 3. Standardize audio features using balanced training data
scaler = StandardScaler()

X_train = scaler.fit_transform(train_balanced[features])
X_val = scaler.transform(val[features])
X_test = scaler.transform(test[features])

y_train = train_balanced["era"].map(labels)
y_val = val["era"].map(labels)
y_test = test["era"].map(labels)

# 4. Build the same MLP as the baseline
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

# 5. Train with the same hyperparameters
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

# 6. Evaluate on the original test set
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
recalls = recall_score(y_test, y_pred, labels=[0, 1, 2], average=None)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

print("\n=== Undersampling Results ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

for era, recall in zip(eras, recalls):
    print(f"{era} Recall: {recall:.4f}")

print("Confusion Matrix:")
print(cm)

# 7. Save model, training history, and evaluation results
model_dir = Path("models/undersampling")
result_dir = Path("results/undersampling")
model_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)

model.save(model_dir / "undersampling.keras")
joblib.dump(scaler, model_dir / "scaler.pkl")

train_balanced.to_csv(result_dir / "balanced_train.csv", index=False)
pd.DataFrame(history.history).to_csv(
    result_dir / "training_history.csv", index=False
)

pd.DataFrame(
    cm, index=eras, columns=eras
).to_csv(result_dir / "confusion_matrix.csv")

pd.DataFrame([{
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "recall_2000s": recalls[0],
    "recall_2010s": recalls[1],
    "recall_2020s": recalls[2]
}]).to_csv(result_dir / "metrics.csv", index=False)

print("\nUndersampling model and results saved!")
