from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import precision_recall_curve, average_precision_score

# --------------------------------------------------
# 1. Output folder and figure style
# --------------------------------------------------
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# muted but visible colors
BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#B279A2"
TEAL = "#72B7B2"
GRAY = "#7F7F7F"
LIGHT_GRID = "#D8D8D8"

def style_axis(ax, grid_axis="y"):
    ax.grid(
        True,
        axis=grid_axis,
        linestyle=(0, (2, 4)),
        linewidth=0.7,
        color=LIGHT_GRID
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")

# --------------------------------------------------
# 2. Load saved results
# --------------------------------------------------
def load_metrics(folder):
    if folder == "baseline":
        path = f"results/{folder}/baseline_metrics.csv"
    else:
        path = f"results/{folder}/metrics.csv"
    return pd.read_csv(path).iloc[0]

def load_cm(folder):
    return pd.read_csv(
        f"results/{folder}/confusion_matrix.csv", index_col=0
    ).to_numpy()

baseline = load_metrics("baseline")
artist_added = load_metrics("artist_added")
undersampling = load_metrics("undersampling")

cm_baseline = load_cm("baseline")
cm_artist = load_cm("artist_added")
cm_under = load_cm("undersampling")

history = pd.read_csv("results/baseline/training_history.csv")
comparison = pd.read_csv("results/artist_disjoint/split_comparison.csv")
splits_df = pd.read_csv("results/baseline/data_splits.csv")

eras = ["2000s", "2010s", "2020s"]

# --------------------------------------------------
# 3. Figure 01: Model benchmark comparison
# --------------------------------------------------
models = ["Baseline", "Artist Added", "Undersampling"]
accuracy = [
    baseline["accuracy"],
    artist_added["accuracy"],
    undersampling["accuracy"]
]
macro_f1 = [
    baseline["macro_f1"],
    artist_added["macro_f1"],
    undersampling["macro_f1"]
]

x = np.arange(len(models))
width = 0.34

fig, ax = plt.subplots(figsize=(4.6, 3.0))

ax.bar(
    x - width / 2,
    accuracy,
    width,
    color=BLUE,
    edgecolor="black",
    linewidth=0.6,
    label="Accuracy"
)
ax.bar(
    x + width / 2,
    macro_f1,
    width,
    color=ORANGE,
    edgecolor="black",
    linewidth=0.6,
    label="Macro F1"
)

ax.set_xticks(x)
ax.set_xticklabels(["Baseline", "Artist\nAdded", "Undersampling"])
ax.set_ylabel("Score")
ax.set_ylim(0, 0.80)
ax.legend(frameon=False, loc="upper right")
style_axis(ax)

fig.tight_layout()
save(fig, "fig01_benchmark")

# --------------------------------------------------
# 4. Figure 02: Precision-Recall curve for 2020s
# --------------------------------------------------
features = [
    "danceability", "energy", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "loudness", "duration_ms"
]

test_df = splits_df[splits_df["split"] == "test"].copy()
y_true = (test_df["era"] == "2020s").astype(int)

model_info = {
    "Baseline": (
        "models/baseline/baseline.keras",
        "models/baseline/baseline_scaler.pkl",
        BLUE,
        "-"
    ),
    "Undersampling": (
        "models/undersampling/undersampling.keras",
        "models/undersampling/scaler.pkl",
        RED,
        "--"
    )
}

fig, ax = plt.subplots(figsize=(4.0, 3.2))

for name, (model_path, scaler_path, color, line_style) in model_info.items():
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    X_test = scaler.transform(test_df[features])
    y_prob = model.predict(X_test, verbose=0)[:, 2]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    ax.step(
        recall,
        precision,
        where="post",
        color=color,
        linestyle=line_style,
        label=f"{name} (AP={ap:.3f})"
    )

    print(f"{name} Average Precision (2020s): {ap:.4f}")

ax.axhline(
    y=y_true.mean(),
    color=GRAY,
    linestyle=":",
    linewidth=1.2,
    label="Class prevalence"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="upper right")
style_axis(ax, grid_axis="both")

fig.tight_layout()
save(fig, "fig02_pr_curve")

# --------------------------------------------------
# 5. Figure 03: Baseline learning curves
# --------------------------------------------------
epochs = np.arange(1, len(history) + 1)
best_epoch = history["val_loss"].idxmin() + 1

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

# loss
axes[0].plot(epochs, history["loss"], color=BLUE, label="Train")
axes[0].plot(epochs, history["val_loss"], color=ORANGE, linestyle="--", label="Validation")
axes[0].axvline(best_epoch, color=GRAY, linestyle=":", linewidth=1.0)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(frameon=False)
style_axis(axes[0], grid_axis="both")

# accuracy
axes[1].plot(epochs, history["accuracy"], color=BLUE, label="Train")
axes[1].plot(epochs, history["val_accuracy"], color=ORANGE, linestyle="--", label="Validation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend(frameon=False)
style_axis(axes[1], grid_axis="both")

fig.tight_layout()
save(fig, "fig03_learning_curve")

# --------------------------------------------------
# 6. Figure 04: RQ2 mean ± SD
# --------------------------------------------------
summary = comparison.groupby("split_type")[["accuracy", "macro_f1"]].agg(["mean", "std"])

groups = ["Random", "Artist Disjoint"]
x = np.arange(len(groups))

fig, ax = plt.subplots(figsize=(4.2, 3.0))

for shift, metric, color, marker, label in [
    (-0.06, "accuracy", BLUE, "s", "Accuracy"),
    (0.06, "macro_f1", ORANGE, "o", "Macro F1"),
]:
    means = [summary.loc[g, (metric, "mean")] for g in groups]
    stds = [summary.loc[g, (metric, "std")] for g in groups]

    ax.errorbar(
        x + shift,
        means,
        yerr=stds,
        fmt=marker,
        color=color,
        ecolor=color,
        markersize=6,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        linewidth=1.4,
        label=label
    )

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Score (mean ± SD)")
ax.set_ylim(0.48, 0.65)
ax.legend(frameon=False, loc="lower left")
style_axis(ax, grid_axis="both")

fig.tight_layout()
save(fig, "fig04_rq2_errorbar")

# --------------------------------------------------
# 7. Figure 05: RQ2 repeated experiments across seeds
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

for ax, metric, ylabel in [
    (axes[0], "accuracy", "Accuracy"),
    (axes[1], "macro_f1", "Macro F1")
]:
    for split_name, color, marker, line_style in [
        ("Random", BLUE, "o", "-"),
        ("Artist Disjoint", RED, "s", "--")
    ]:
        temp = comparison[comparison["split_type"] == split_name].sort_values("seed")
        ax.plot(
            temp["seed"],
            temp[metric],
            color=color,
            marker=marker,
            markersize=4,
            linestyle=line_style,
            label=split_name
        )

    ax.set_xlabel("Random seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(comparison["seed"].unique()))
    style_axis(ax, grid_axis="both")

axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "fig05_rq2_repeated")

# --------------------------------------------------
# 8. Helper for confusion matrices
# --------------------------------------------------
def draw_confusion_pair(cm1, cm2, titles, filename, normalize=False, cmap="Blues"):
    mats = [cm1.astype(float), cm2.astype(float)]

    if normalize:
        mats = [m / m.sum(axis=1, keepdims=True) for m in mats]

    vmax = 1.0 if normalize else max(cm1.max(), cm2.max())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))

    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(eras)
        ax.set_yticklabels(eras)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=9)

        # cell borders
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(3):
            for j in range(3):
                value = mat[i, j]
                text = f"{value:.1%}" if normalize else f"{int(value)}"
                color = "white" if value > vmax * 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    save(fig, filename)

# --------------------------------------------------
# 9. Figure 06: RQ1 confusion matrix
# --------------------------------------------------
draw_confusion_pair(
    cm_baseline,
    cm_artist,
    ["Baseline", "Artist Added"],
    "fig06_rq1_confusion",
    normalize=False,
    cmap="YlGnBu"
)

# --------------------------------------------------
# 10. Figure 07: RQ3 normalized confusion matrix
# --------------------------------------------------
draw_confusion_pair(
    cm_baseline,
    cm_under,
    ["Baseline", "Undersampling"],
    "fig07_rq3_confusion",
    normalize=True,
    cmap="OrRd"
)

# --------------------------------------------------
# 11. Figure 08: Class-wise recall
# --------------------------------------------------
recall_base = np.diag(cm_baseline) / cm_baseline.sum(axis=1)
recall_under = np.diag(cm_under) / cm_under.sum(axis=1)

x = np.arange(len(eras))
width = 0.34

fig, ax = plt.subplots(figsize=(4.2, 3.0))

ax.bar(
    x - width / 2,
    recall_base,
    width,
    color=BLUE,
    edgecolor="black",
    linewidth=0.6,
    label="Baseline"
)
ax.bar(
    x + width / 2,
    recall_under,
    width,
    color=RED,
    edgecolor="black",
    linewidth=0.6,
    label="Undersampling"
)

ax.set_xticks(x)
ax.set_xticklabels(eras)
ax.set_ylabel("Recall")
ax.set_ylim(0, 0.8)
ax.legend(frameon=False, loc="upper right")
style_axis(ax)

fig.tight_layout()
save(fig, "fig08_class_recall")

# --------------------------------------------------
# 12. Figure 09: Dataset distribution
# --------------------------------------------------
counts = splits_df["era"].value_counts().reindex(eras)

fig, ax = plt.subplots(figsize=(4.0, 3.0))

bars = ax.bar(
    eras,
    counts.values,
    color=[BLUE, ORANGE, GREEN],
    edgecolor="black",
    linewidth=0.6,
    width=0.58
)

for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 15,
        str(count),
        ha="center",
        va="bottom",
        fontsize=8
    )

ax.set_ylabel("Number of songs")
ax.set_ylim(0, 950)
style_axis(ax)

fig.tight_layout()
save(fig, "fig09_data_distribution")

print("\nAll figures created successfully!")
print("Saved to:", OUT)