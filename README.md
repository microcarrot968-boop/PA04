# PA04 — Music Era Classification of Billboard Year-End Tracks (2000–2024)

## 1. Project Overview

This project classifies songs from the Billboard Hot 100 Year-End Tracks (2000–2024) dataset into three eras (2000s, 2010s, and 2020s) using numerical audio features and a multilayer perceptron (MLP). The label is based on the **first year each song appeared on the Billboard Year-End chart**, not its release year.

Three comparative experiments were conducted:

1. **Artist Added:** Examine whether adding artist names improves classification performance.
2. **Artist Disjoint:** Examine the performance of an audio-feature-based model on songs by artists not included in the training data.
3. **Undersampling:** Examine how balancing the training classes affects performance for the 2020s and the other eras.

## 2. Dataset and Preprocessing

**Dataset source:** [Billboard Hot 100 Year-End Tracks (2000–2024) — Kaggle](https://www.kaggle.com/datasets/matthewwilliams333/billboard-hot-100-year-end-tracks-2000-2024)

The chart records and audio features were linked by `title` and `artist_name`. Era labels were assigned using the earliest Year-End chart appearance:

| Era | First chart appearance | Songs |
| --- | --- | ---: |
| 2000s | 2000–2009 | 828 |
| 2010s | 2010–2019 | 823 |
| 2020s | 2020–2024 | 268 |
| **Total** | | **1,919** |

The baseline model used 10 numerical audio features:

`danceability`, `energy`, `tempo`, `valence`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `duration_ms`.

Song titles and chart years were not used as model inputs. After missing values were removed, 1,926 records remained. Data validation identified 14 records with identical values across all 10 audio features; 12 had appeared across different initial data splits. Removing seven duplicate feature vectors resulted in 1,919 records.

The processed dataset used for the reported experiments is **`data/billboard/processed.csv`**. The repository also includes `data/billboard/tracks.csv` and `data/billboard/chart_entries.csv`, so **downloading the dataset again is not required to verify the saved results**.

**Data reproduction note:** `download_data.py` uses KaggleHub to download the original dataset. Its current `output_dir="./billboard"` is different from the `data/billboard/` path expected by `preprocess.py`. If downloading from scratch, check the downloaded file paths and place the required CSV files under `data/billboard/` before running preprocessing. Do not overwrite the committed `processed.csv` without first comparing the output with the saved data.

## 3. Model and Experimental Settings

The baseline MLP uses the following structure:

`10 inputs → 64 ReLU → 32 ReLU → 3 Softmax`

Training settings:

| Setting | Value |
| --- | --- |
| Loss | Sparse Categorical Crossentropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early stopping | Validation loss, patience 10, restore best weights |
| Baseline random seed | 42 |

The baseline stratified split consisted of **1,151 training songs**, **384 validation songs**, and **384 test songs** (60% / 20% / 20%). StandardScaler was fitted using only the training data.

- **Artist Added:** 611 one-hot-encoded artist columns were added to the 10 audio features (621 input dimensions). The baseline data split was retained.
- **Artist Disjoint:** The same artist name did not appear across training, validation, and test sets. The model still used only the 10 audio features. A separate comparison between Random and Artist-Disjoint splits was repeated with seeds **42–46**, matching era counts within each split.
- **Undersampling:** Only the training set was reduced to **161 songs per era (483 songs total)**. Validation and test sets remained unchanged. The scaler was fitted on the balanced training data.

Evaluation included Accuracy, Macro F1, class-wise Recall, and confusion matrices.

## 4. Original Experimental Environment

The models were trained in a **university-provided NVIDIA GPU Cloud session** using the **TensorFlow (NGC) 24.05** environment. The session settings used for training were:

| Setting | Configuration |
| --- | --- |
| Platform | NVIDIA GPU Cloud (university session) |
| Environment image | TensorFlow (NGC) 24.05 |
| Python | 3.10 |
| TensorFlow | 2.15.0 |
| CUDA | 12.4 |
| CPU | 2 cores |
| Memory | 10 GiB |
| AI accelerator allocation | 1.0 fGPU |
| Session / node / container | 1 session, single node, 1 container |

**Keras 3.4.1** was confirmed in a school session using the same environment image (with the accelerator allocation set to 0 for that check). This is not a guarantee that an arbitrary fresh pip installation of TensorFlow 2.15.0 and Keras 3.4.1 will resolve dependencies: pip reports a dependency conflict between them.

**Recommended environment:** Use the school's preconfigured **TensorFlow (NGC) 24.05** session. Do not replace the preinstalled TensorFlow/Keras packages just to follow `requirements.txt`. The `environment.yml` and `requirements.txt` are included for documenting the software dependencies, but installing them alone does **not** guarantee reproduction of the NGC image or GPU environment.

The saved models were successfully loaded and their reported metrics verified in the school's preconfigured environment using **`/usr/bin/python3`**. The Conda `PA04` environment in that session did not contain all required packages and was not used for the successful verification.

## 5. Repository Structure

```text
PA04/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── download_data.py
├── preprocess.py
├── train.py
├── train_artist_added.py
├── train_artist_disjoint.py
├── train_undersampling.py
├── compare_splits.py
├── experiments.py
├── evaluate.py
├── check_baseline.py
├── make_figures.py
├── verify_experiments.py
├── verify_models.py
├── data/
│   └── billboard/
│       ├── chart_entries.csv
│       ├── tracks.csv
│       └── processed.csv
├── models/
│   ├── baseline/
│   ├── artist_added/
│   ├── artist_disjoint/
│   └── undersampling/
└── results/
    ├── baseline/
    ├── artist_added/
    ├── artist_disjoint/
    ├── undersampling/
    └── figures/
```

The model folders include saved `.keras` files and fitted `.pkl` scalers. The Artist Added folder also includes **`artist_columns.json`**, which records the artist-encoding columns.

The results folders include metrics, training histories, confusion matrices, and relevant data-split files. `results/figures/` contains the figures used in the paper and additional diagnostic figures.

## 6. How to Reproduce the Saved Results (Verified)

The following procedure **was successfully tested in the university NGC session**. It loads the saved models and reproduces their evaluation metrics; it does **not** train the models again.

### Step 1 — Open the university session

Select **TensorFlow (NGC) 24.05**, Python 3.10, CUDA 12.4. The original training session used 2 CPU cores, 10 GiB memory, and 1.0 fGPU allocation. GPU allocation is not required merely to check the saved models if the session's Python environment can load them.

### Step 2 — Open the project directory

Keep the repository in the persistent workspace rather than the home directory:

```bash
git clone https://github.com/microcarrot968-boop/PA04.git
cd PA04
```

If the repository was already cloned under `/home/work/ef2039_workspace/PA04`, use:

```bash
cd /home/work/ef2039_workspace/PA04
```

### Step 3 — Verify with the system Python

**Use the preinstalled system Python, not the incomplete Conda `PA04` environment.**

```bash
/usr/bin/python3 verify_experiments.py
/usr/bin/python3 verify_models.py
```

Both commands were run successfully in the school session on **November 15, 2026**.

Expected final messages:

```text
All saved metrics verified!
PASS: Saved results reproduced!
=== All models verified successfully! ===
```

The model verification reported **PASS** for all four saved models: Baseline, Artist Added, Artist Disjoint, and Undersampling. It verified the saved test predictions against the reported Accuracy and Macro F1 values.

## 7. Full Training and Figure Generation

The repository includes these scripts:

| Script | Purpose |
| --- | --- |
| `download_data.py` | Download the source dataset using KaggleHub |
| `preprocess.py` | Merge chart and audio data, clean missing values, prepare era labels |
| `train.py` | Train the baseline MLP |
| `train_artist_added.py` | Train the model with one-hot artist information |
| `train_artist_disjoint.py` | Train the artist-disjoint model |
| `train_undersampling.py` | Train on the balanced training set |
| `compare_splits.py` | Compare Random vs Artist-Disjoint splits across five seeds |
| `evaluate.py` | Evaluate experiment results |
| `check_baseline.py` | Baseline diagnostics |
| `make_figures.py` | Generate figures from results |
| `verify_experiments.py` | Check data splits and saved metrics |
| `verify_models.py` | Reload saved models and verify metrics |

For a **fresh full training run**, use the school NGC environment and consult the scripts in this repository. Their intended high-level order is preprocessing, baseline training, comparative training, repeated split comparison, evaluation, and figure generation.

**Important:** The verification commands in Section 6 have been tested. A complete from-scratch training run using every script in sequence has **not** been independently verified as part of this README. Re-running `preprocess.py` or the training scripts may overwrite the committed datasets, model files, or results. Make a copy of the repository first if you want to compare new training outputs with the original results.

## 8. Expected Results

### Single-run results

| Model | Test samples | Accuracy | Macro F1 | 2020s Recall |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 384 | 59.11% | 0.5472 | 32.08% |
| Artist Added | 384 | 70.57% | 0.6454 | 35.85% |
| Artist Disjoint | 371 | 58.76% | 0.5163 | 22.03% |
| Undersampling | 384 | 48.96% | 0.4776 | 66.04% |

The Artist-Disjoint single-run experiment used a different test set from the baseline. Its single-run performance should not be treated as a direct comparison on identical test songs.

### Five-seed Random vs Artist-Disjoint comparison

| Split | Accuracy (mean ± SD) | Macro F1 (mean ± SD) |
| --- | --- | --- |
| Random | 60.07 ± 1.98% | 0.5387 ± 0.0173 |
| Artist Disjoint | 58.87 ± 1.63% | 0.5235 ± 0.0156 |

The mean Accuracy difference was **1.20 percentage points**. Artist Disjoint obtained higher Accuracy in two out of five runs, so the evidence of a generalization decrease is limited.

Undersampling increased 2020s Recall from **32.08% to 66.04%** while overall Accuracy and Macro F1 decreased. The 2010s Recall decreased from **63.64% to 37.58%**.

Saved result files include:

- `results/baseline/baseline_metrics.csv`
- `results/artist_added/metrics.csv`
- `results/artist_disjoint/metrics.csv`
- `results/artist_disjoint/split_comparison.csv`
- `results/artist_disjoint/split_comparison_summary.csv`
- `results/undersampling/metrics.csv`

## 9. Figures

The figures used in the paper are provided as PDF and PNG files under `results/figures/`:

- `fig01_benchmark`: Comparison of Accuracy and Macro F1 across the Baseline, Artist Added, and Undersampling models.
- `fig08_class_recall`: Class-wise Recall for the Baseline and Undersampling models.

Additional figures include confusion matrices, learning curves, a precision–recall curve, repeated split results, and the era distribution.

## 10. Limitations

Adding artist information also increased the number of model parameters, so the improvement cannot be attributed solely to artist identity. The Artist-Disjoint experiments used different test songs, and the repeated comparison was limited to five seeds. Undersampling reduced both class imbalance and training data size and changed the standardization parameters. The 2020s cover five years, while the other eras cover ten years each. The dataset covers Billboard Year-End chart songs rather than all music.

## 11. Links

- **Project repository:** https://github.com/microcarrot968-boop/PA04
- **Original dataset:** https://www.kaggle.com/datasets/matthewwilliams333/billboard-hot-100-year-end-tracks-2000-2024
