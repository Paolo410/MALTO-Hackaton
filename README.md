# MALTO Recruitment Hackathon — Human vs. AI Text Classification

> **Competition result: Macro F1 = 0.916** on the Kaggle public leaderboard  
> Task: 6-class classification of text samples as Human-written (class 0) or AI-generated (classes 1–5)

---

## Problem Overview

The dataset contains **3,000 text samples** with severe class imbalance: class 0 (human) holds ~63% of samples while some AI classes have as few as 80 examples (a 20:1 ratio). The evaluation metric is **Macro F1**, which treats all classes equally regardless of size — this makes the imbalance a first-class engineering problem, not just a nuisance.

```
Class 0 (Human):  1520 samples  ████████████████████████████████████████████████████████████████████████████
Class 1 (AI):       80 samples  ████
Class 2 (AI):      160 samples  ████████
Class 3 (AI):       80 samples  ████
Class 4 (AI):      240 samples  ████████████
Class 5 (AI):      320 samples  ████████████████
```

---

## Architecture

The pipeline is built entirely with scikit-learn primitives, making it serialisable, reproducible, and easy to audit. Four feature streams are combined via `FeatureUnion` and fed to a LightGBM classifier.

```
Raw text (single TEXT column)
        │
        ├── ScalarTextFeatureExtractor  ──►  ~33 dims   (handcrafted stylometry)
        │
        ├── PretrainedEmbedder (GloVe)  ──►  100 dims   (semantic context)
        │
        ├── TfidfImpChiSelector         ──►  ~120 dims  (word n-grams, ImpCHI)
        │
        └── TfidfCharSelector           ──►  ~120 dims  (char n-grams, ImpCHI)
                                              │
                                        FeatureUnion
                                              │
                                     ~373 dense features
                                              │
                                    LGBMClassifier
                                  (class_weight='balanced')
                                              │
                                      Final prediction
```

---

## Feature Engineering

### 1. Scalar Stylometric Features (`ScalarTextFeatureExtractor`)

Handcrafted signals that distinguish human writing from AI output without any learned parameters:

**Structural features:** character length, word count, sentence count, words-per-sentence ratio, log-scaled length.

**Lexical richness:** Type-Token Ratio (TTR) — the fraction of unique words over total words. Human text tends to have higher variance in TTR across sentences; AI text is more uniform.

**Sentence-length variance:** the standard deviation of word counts across sentences. AI-generated text is notably more regular; human writing has more erratic rhythm.

**Capitalisation patterns:** ratio of uppercase characters, count of ALL-CAPS words. Student essays often contain emotional capitalisation or shouting (`"I WAS ESPECIALLY DELIGHTED"`).

**Typo / error signals:** triple-consonant runs (e.g. `aboliished`), repeated-character runs (`!!!`), alphanumeric character mixing — patterns characteristic of human typos.

**AI discourse markers:** 14 binary flags for formal connectives that are overrepresented in LLM output: `furthermore`, `moreover`, `in conclusion`, `notably`, `this underscores`, `ultimately`, etc. These appear rarely in casual human writing.

### 2. Pretrained Word Embeddings (`PretrainedEmbedder`)

Loads `glove-wiki-gigaword-100` (100-dimensional GloVe vectors trained on Wikipedia + Gigaword) via `gensim.downloader`. Each document is encoded as the **mean of its token vectors** (mean-pooling). The model is downloaded once and cached in `~/gensim-data`.

Training from scratch on only 2,400 samples would memorise corpus noise rather than learn transferable semantics. Using pretrained vectors contributes zero variance to the final model — all its information is frozen from pre-training.

### 3. Word-level TF-IDF with ImpCHI (`TfidfImpChiSelector`)

Standard TF-IDF with unigrams and bigrams, followed by **Improved Chi-squared (ImpCHI)** feature selection.

The key insight: classical Chi-squared selects features by their global discriminative power, which in an imbalanced dataset means it selects features correlated with the *majority class* (class 0). ImpCHI fixes this by selecting the top-K features **per class independently** and taking the union. This guarantees that the distinctive vocabulary of minority classes (e.g. class 1 with only 80 samples) is not drowned out by the majority.

> Reference: Bahassine et al., *Feature selection using an improved chi-square for Arabic text classification*, J. King Saud University — CIS, 2018.

### 4. Character-level TF-IDF with ImpCHI (`TfidfCharSelector`)

The same ImpCHI selection applied to character n-grams (3–5-grams) with `analyzer='char_wb'` (word-boundary padding). Character n-grams capture sub-word stylometric patterns invisible to word tokenisers:

- AI text has smooth, predictable character transitions (e.g. `"rmore"` from `furthermore`)
- Human typos generate rare sequences (e.g. `"liish"` from `aboliished`)
- Formal suffixes like `"tion."`, `"ver, "` are strong AI indicators at the character level

---

## Classifier: LightGBM

A single `LGBMClassifier` with `class_weight='balanced'` to compensate for the 20:1 class imbalance. Parameters are deliberately conservative to prevent memorisation on a 2,400-sample corpus:

| Parameter | Value | Rationale |
|---|---|---|
| `max_depth` | 3–5 (tuned) | Very shallow trees, no room to memorise |
| `num_leaves` | 10–20 (tuned) | Far below sklearn default of 31 |
| `min_child_samples` | 20–60 (tuned) | Forces large, generalisable leaves |
| `reg_alpha` | 0.1–1.0 | L1 regularisation |
| `reg_lambda` | 1.0–10.0 | L2 regularisation |

The pipeline also registers `"random_forest"`, `"linear_svc"`, `"xgboost"` and `"ensemble"` in the `MODEL_REGISTRY` — switching is one line in `CONFIG`.

---

## Hyperparameter Optimisation

Bayesian optimisation via **Optuna TPE** (Tree-structured Parzen Estimator) over 7 trials with 5-fold stratified cross-validation, maximising Macro F1.

The search space is intentionally narrow — designed to regularise, not to maximise capacity:

| Parameter | Range | Rationale |
|---|---|---|
| `max_depth` (trees) | 3–5 | Prevents memorisation |
| `num_leaves` (LGBM) | 10–20 | Far below sklearn default of 31 |
| `min_child_samples` | 20–60 | Forces large, generalisable leaves |
| `k_per_label` (ImpCHI word) | 10–30 | Only strongest per-class features |
| `k_per_label_char` (ImpCHI char) | 10–30 | Only strongest char patterns |

The GloVe embedder is **frozen** before the Optuna loop starts and shared across all trials, saving substantial compute.

---

## Pseudo-Labeling

After training on the full labelled set, `predict_proba` is run on the 600 unlabelled test samples. Any sample with a maximum class probability ≥ 0.90 is assigned its predicted label as a **pseudo-label** and added to the training set.

A fresh pipeline is then retrained from scratch on this augmented dataset (TF-IDF and ImpCHI re-fitted on the larger corpus). This is safe at high thresholds: empirically, over 90% of pseudo-labels at 0.90 confidence are correct.

---

## Results

| Split | Macro F1 |
|---|---|
| Cross-validation (5-fold, training split) | ~0.91 |
| Hold-out test set (15% stratified split) | ~0.90 |
| **Kaggle public leaderboard** | **0.916** |

---

## Post-competition improvements

After the competition closed, two further changes were tested:

**Soft-voting ensemble** (`active_model = "ensemble"`): replacing the single LightGBM with a `VotingClassifier(voting='soft')` combining LGBM + XGBoost + CalibratedLinearSVC. The three members fail on different examples (trees struggle on sparse TF-IDF; LinearSVC excels there), so their averaged probabilities are better calibrated than any individual model.

**Sentence Transformer embeddings**: replacing GloVe mean-pooling with `all-MiniLM-L6-v2` from `sentence-transformers`. Unlike GloVe (static per-word vectors), MiniLM encodes the full sentence context — distinguishing `"Furthermore, this demonstrates..."` from a human typo with the same tokens. The model is frozen during inference (no fine-tuning), so it adds zero variance despite its 22M parameters.

| Version | Embedding | Active model | Macro F1 |
|---|---|---|---|
| Competition submission | GloVe 100-dim | `lgbm` | **0.916** |
| Post-competition | Sentence Transformer MiniLM | `ensemble` | **0.925** |

---

## Repository Structure

```
├── pipeline.py          # Full self-contained pipeline
├── dataset/
│   ├── train.csv        # 2,400 labelled samples (TEXT, LABEL)
│   └── test.csv         # 600 unlabelled samples (TEXT)
├── submission.csv       # Final predictions (0.916)
└── README.md
```

---

## Quickstart

```bash
pip install lightgbm xgboost scikit-learn optuna gensim pandas numpy
python pipeline.py
```

The script trains the full pipeline, prints a classification report on the hold-out set, applies pseudo-labeling, and writes `submission.csv`.

Key settings in `CONFIG` and at the top of the script:

```python
CONFIG = {
    "active_model": "lgbm",    # "lgbm" | "xgboost" | "linear_svc" | "ensemble"
    "n_trials": 7,             # Optuna trials (increase for better tuning)
    "cv_folds": 5,
    "glove_model_name": "glove-wiki-gigaword-100",
    ...
}
PSEUDO_LABEL_THRESHOLD = 0.90  # set to 1.01 to disable pseudo-labeling
```

---

## Dependencies

```
lightgbm>=4.0
xgboost>=1.7
scikit-learn>=1.3
optuna>=3.0
gensim>=4.3
pandas>=2.0
numpy>=1.24
```

---

## Key Design Decisions for the Interview

**Why not a fine-tuned BERT end-to-end?** With only 2,400 samples, fine-tuning a full transformer risks catastrophic overfitting. Using frozen pretrained embeddings as features for a gradient-boosted classifier extracts the semantic power of transformers while keeping the learnable parameter count minimal.

**Why ImpCHI instead of standard Chi-squared?** On imbalanced data, global Chi-squared selects features that discriminate the majority class — here, human text. ImpCHI independently selects the top features *per class*, guaranteeing that rare AI variants (80 samples) have their distinctive vocabulary represented in the final feature space.

**Why pseudo-labeling?** The test set is 600 samples — 25% of the training set size. High-confidence pseudo-labels at ≥0.90 threshold add effectively noise-free signal and expand the vocabulary seen by TF-IDF, improving generalisation at negligible cost.

**Why character n-grams alongside word n-grams?** Word tokenisers miss sub-word patterns. A character trigram like `"iis"` or `"liis"` is a direct fingerprint of human typing errors; the AI-marker trigram `"rmo"` (from `furthermore`) is invisible at the word level. The two feature spaces are complementary and their union is more robust than either alone.