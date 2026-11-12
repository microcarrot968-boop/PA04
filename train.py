import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Set random seed and load data
tf.keras.utils.set_random_seed(42)

df = pd.read_csv("data/billboard/processed.csv")

features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]

labels = {"2000s": 0, "2010s": 1, "2020s": 2}

X = df[features]
y = df["era"].map(labels)

# 2. Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    stratify=y_temp, random_state=42
)

# Save data splits for reproducibility
df["split"] = "train"
df.loc[X_val.index, "split"] = "val"
df.loc[X_test.index, "split"] = "test"

# 3. Standardize features using training data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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

print("\n=== Baseline Results ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("Confusion Matrix:")
print(cm)

# 7. Save model and results
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

model.save("models/baseline.keras")
joblib.dump(scaler, "models/baseline_scaler.pkl")

df.to_csv("results/data_splits.csv", index=False)
pd.DataFrame(history.history).to_csv(
    "results/training_history.csv", index=False
)

pd.DataFrame(
    cm,
    index=["2000s", "2010s", "2020s"],
    columns=["2000s", "2010s", "2020s"]
).to_csv("results/confusion_matrix.csv")

pd.DataFrame([{
    "accuracy": accuracy,
    "macro_f1": macro_f1
}]).to_csv("results/baseline_metrics.csv", index=False)

print("\nModel and results saved successfully!")
