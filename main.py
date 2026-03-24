"""
pipeline.py — MALTO Recruitment Hackathon
==========================================
Human vs. AI-Generated Text Classification (6-class, imbalanced)

Architecture: Hybrid ImpCHI + Word2Vec + Scalar Features → LightGBM
Inspired by: "Optimizing News Categorization on Imbalanced Data:
              A Hybrid Feature and Gradient Boosting Approach"
              (Busonera & Malugani, Politecnico di Torino)

Author: Paolo Malugani — adapted for MALTO single-text-column format
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import string
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# ── Reproducibility seeds must be set BEFORE any library import that uses them
os.environ["PYTHONHASHSEED"] = "42"

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

import gensim.downloader as gensim_dl
from gensim.utils import simple_preprocess

from lightgbm import LGBMClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report, f1_score

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCALAR FEATURE EXTRACTOR
#    Derives handcrafted numeric signals that distinguish human vs. AI writing.
#    Key insight: human text has higher error rates, more variance in sentence
#    length, and lower "AI tells" (formal transitions, uniform structure).
# ══════════════════════════════════════════════════════════════════════════════

class ScalarTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numeric (scalar) features from a raw text column.

    Designed to capture stylometric signals that correlate with whether a text
    was written by a human or an AI model:

    - Structural: length, word count, sentence count, paragraph hints
    - Lexical: type-token ratio (diversity), avg word length
    - Error signals: doubled chars, excessive punctuation runs, ALLCAPS words
    - AI "tells": frequency of formal discourse markers (Furthermore, Indeed…)
    - Punctuation & spacing patterns

    Parameters
    ----------
    text_col : str
        Name of the text column in the input DataFrame. Default ``"TEXT"``.
    """

    # High-confidence AI discourse markers (formal connectives rarely used by
    # casual human writers, especially student essayists with typos)
    _AI_MARKERS: List[str] = [
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bin conclusion\b",
        r"\bnotably\b",
        r"\bit is worth noting\b",
        r"\bit is important to note\b",
        r"\bin summary\b",
        r"\bto summarize\b",
        r"\bin essence\b",
        r"\bthis underscores\b",
        r"\bthis highlights\b",
        r"\bultimately\b",
        r"\bthis demonstrates\b",
        r"\boverall\b",
    ]

    def __init__(self, text_col: str = "TEXT") -> None:
        self.text_col = text_col

    # ------------------------------------------------------------------
    # sklearn API — fit is a no-op (all features are unsupervised stats)
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "ScalarTextFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a 2-D float array of shape (n_samples, n_features).

        All features are computed on the raw text string, preserving original
        casing and spacing so that error-signal features remain intact.
        """
        texts: pd.Series = X[self.text_col].fillna("").astype(str)
        frames: List[pd.Series] = []

        # ── 1. Basic length / count features ─────────────────────────────────
        frames.append(texts.str.len().rename("char_len"))
        word_counts = texts.str.split().str.len().fillna(0)
        frames.append(word_counts.rename("word_count"))

        # Sentence count: split on [.!?] followed by whitespace or end-of-str
        sent_counts = texts.apply(
            lambda t: max(1, len(re.split(r"[.!?]+\s+|[.!?]+$", t.strip())))
        )
        frames.append(sent_counts.rename("sentence_count"))
        frames.append((word_counts / sent_counts).rename("words_per_sentence"))

        # ── 2. Lexical richness ───────────────────────────────────────────────
        def _ttr(text: str) -> float:
            """Type-Token Ratio: unique_words / total_words (capped at 1.0)."""
            tokens = text.lower().split()
            if not tokens:
                return 0.0
            return min(len(set(tokens)) / len(tokens), 1.0)

        frames.append(texts.apply(_ttr).rename("type_token_ratio"))

        def _avg_word_len(text: str) -> float:
            words = text.split()
            if not words:
                return 0.0
            return np.mean([len(w) for w in words])

        frames.append(texts.apply(_avg_word_len).rename("avg_word_len"))

        # ── 3. Sentence-length variance (AI text is more uniform) ─────────────
        def _sent_len_std(text: str) -> float:
            sents = re.split(r"[.!?]+\s+|[.!?]+$", text.strip())
            lens = [len(s.split()) for s in sents if s.strip()]
            return float(np.std(lens)) if len(lens) > 1 else 0.0

        frames.append(texts.apply(_sent_len_std).rename("sent_len_std"))

        # ── 4. Punctuation & special-char ratios ──────────────────────────────
        char_len = texts.str.len().replace(0, 1)  # avoid /0

        punct_count = texts.apply(
            lambda t: sum(1 for c in t if c in string.punctuation)
        )
        frames.append((punct_count / char_len).rename("punct_ratio"))

        comma_count = texts.str.count(",")
        frames.append((comma_count / (word_counts + 1)).rename("comma_per_word"))

        # Exclamation and question marks (human emotion signals)
        frames.append(texts.str.count(r"!").rename("exclamation_count"))
        frames.append(texts.str.count(r"\?").rename("question_count"))

        # ── 5. Capitalisation patterns ────────────────────────────────────────
        # ALL-CAPS words (e.g. "I WAS ESPECIALLY DELIGHTED" in label-1 sample)
        frames.append(
            texts.apply(
                lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1)
            ).rename("allcaps_word_count")
        )

        # Ratio of uppercase characters overall
        frames.append(
            texts.apply(
                lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
            ).rename("uppercase_char_ratio")
        )

        # ── 6. Typo / error signals (human fingerprints) ─────────────────────
        # Doubled consonants that look like typos (e.g. "aboliished", "iis")
        frames.append(
            texts.str.count(r"([b-df-hj-np-tv-z])\1{2,}").rename("triple_consonant_count")
        )

        # Runs of repeated non-space characters ≥ 3 (e.g. "!!!!", "???")
        frames.append(
            texts.str.count(r"(.)\1{2,}").rename("char_repeat_run_count")
        )

        # Words with internal digit-letter mix or abnormal apostrophe (typos)
        frames.append(
            texts.apply(
                lambda t: sum(
                    1 for w in t.split()
                    if re.search(r"[a-z][0-9]|[0-9][a-z]", w.lower())
                )
            ).rename("alphanum_mix_count")
        )

        # ── 7. AI discourse markers ───────────────────────────────────────────
        ai_marker_total = sum(
            texts.str.count(pattern, flags=re.IGNORECASE)
            for pattern in self._AI_MARKERS
        )
        frames.append(ai_marker_total.rename("ai_marker_count"))

        # Each marker gets its own binary presence flag for granularity
        for pattern in self._AI_MARKERS:
            name = re.sub(r"\\b|[^a-z_]", "", pattern).strip("_")
            frames.append(
                texts.str.contains(pattern, flags=re.IGNORECASE, regex=True)
                .astype(int)
                .rename(f"marker_{name}")
            )

        # ── 8. Digit / number count ───────────────────────────────────────────
        frames.append(texts.str.count(r"\d+").rename("digit_group_count"))

        # ── 9. Log-scaled length (compresses the heavy tail) ─────────────────
        frames.append(np.log1p(texts.str.len()).rename("log_char_len"))

        # ── Assemble ──────────────────────────────────────────────────────────
        result = pd.concat(frames, axis=1).fillna(0).astype(float)
        return result.values  # shape: (n_samples, n_scalar_features)

    def get_feature_names_out(self) -> List[str]:
        """Returns feature names for inspection / downstream use."""
        dummy = pd.DataFrame({self.text_col: ["placeholder text."]})
        n = self.transform(dummy).shape[1]
        # Re-derive names by running on a 2-element frame (cheap)
        df2 = pd.DataFrame({self.text_col: ["placeholder.", "second."]})
        frames = []
        texts = df2[self.text_col]
        frames.append(texts.str.len().rename("char_len"))
        # … (abbreviated; names match exactly the ones set above)
        # For external tools, we expose a best-effort list:
        names = [
            "char_len", "word_count", "sentence_count", "words_per_sentence",
            "type_token_ratio", "avg_word_len", "sent_len_std",
            "punct_ratio", "comma_per_word", "exclamation_count", "question_count",
            "allcaps_word_count", "uppercase_char_ratio",
            "triple_consonant_count", "char_repeat_run_count", "alphanum_mix_count",
            "ai_marker_count",
        ] + [
            f"marker_{re.sub(r'[^a-z_]', '', re.sub(r'\\b', '', p)).strip('_')}"
            for p in self._AI_MARKERS
        ] + ["digit_group_count", "log_char_len"]
        return names[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRETRAINED EMBEDDER (GloVe / Word2Vec via gensim.downloader)
#
#    WHY pretrained instead of training from scratch:
#    On ~3k texts a from-scratch Word2Vec memorises corpus-specific noise
#    rather than learning transferable semantics. A model pre-trained on
#    billions of tokens (Wikipedia + Gigaword) carries genuine world knowledge
#    that generalises far better to unseen test data.
#
#    Available models (set via CONFIG["glove_model_name"]):
#      "glove-wiki-gigaword-50"   ->  50-dim,  ~70  MB  (fastest)
#      "glove-wiki-gigaword-100"  -> 100-dim,  ~130 MB  <- default
#      "glove-twitter-25"         ->  25-dim,  ~50  MB  (Twitter-domain)
#      "word2vec-google-news-300" -> 300-dim,  ~1.6 GB  (high quality, heavy)
#
#    The model is downloaded ONCE and cached in ~/gensim-data.
#    Subsequent runs load from cache instantly -- no re-download.
# ══════════════════════════════════════════════════════════════════════════════

class PretrainedEmbedder(BaseEstimator, TransformerMixin):
    """
    Document embedder via mean-pooling over a pretrained GloVe/W2V vocabulary.

    Parameters
    ----------
    model_name : str
        gensim.downloader key for the pretrained model.
        Default ``"glove-wiki-gigaword-100"`` (100-dim, ~130 MB).
    text_col : str
        Name of the text column in the input DataFrame. Default ``"TEXT"``.
    """

    def __init__(
        self,
        model_name: str = "glove-wiki-gigaword-100",
        text_col: str = "TEXT",
    ) -> None:
        self.model_name = model_name
        self.text_col   = text_col

        # Set after fit()
        self._keyed_vectors = None   # gensim KeyedVectors object
        self._vocab: set    = set()
        self._vector_size: int = 0

    # ── Helpers ────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(series: pd.Series) -> List[List[str]]:
        return [simple_preprocess(t) for t in series.fillna("").astype(str)]

    def _doc_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean-pools pretrained vectors for all in-vocabulary tokens."""
        valid = [t for t in tokens if t in self._vocab]
        if valid:
            return np.mean(self._keyed_vectors[valid], axis=0)
        return np.zeros(self._vector_size)

    # ── sklearn API ───────────────────────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y=None) -> "PretrainedEmbedder":
        """
        Loads the pretrained model from gensim cache (downloads on first call).
        No training on X -- fit() is a data-independent load step.
        """
        if self._keyed_vectors is None:
            print(f"[Embedder] Loading '{self.model_name}' via gensim.downloader...")
            print(f"[Embedder] (First run downloads to ~/gensim-data; cached afterwards)")
            self._keyed_vectors = gensim_dl.load(self.model_name)
            self._vocab         = set(self._keyed_vectors.index_to_key)
            self._vector_size   = self._keyed_vectors.vector_size
            print(f"[Embedder] Loaded. Vocab: {len(self._vocab):,} tokens, dim={self._vector_size}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._keyed_vectors is None:
            raise RuntimeError("Call fit() before transform().")
        tokenized = self._tokenize(X[self.text_col].fillna("").astype(str))
        return np.vstack([self._doc_vector(toks) for toks in tokenized])

    def get_feature_names_out(self) -> List[str]:
        return [f"glove_{i}" for i in range(self._vector_size)]


# Alias for backward compatibility
Word2VecEmbedder = PretrainedEmbedder

# ══════════════════════════════════════════════════════════════════════════════
# 3. TF-IDF + IMPCHI SELECTOR
#    Improved Chi-squared (ImpCHI) feature selection:
#    Instead of selecting the global top-K features (biased toward majority
#    classes), we independently rank the top K features *per label* and take
#    their union. This ensures minority-class vocabulary is not suppressed.
#    Reference: Bahassine et al. (2018), J. King Saud Univ. - CIS.
# ══════════════════════════════════════════════════════════════════════════════

class TfidfImpChiSelector(BaseEstimator, TransformerMixin):
    """
    TF-IDF vectorisation followed by per-label ImpCHI feature selection.

    Parameters
    ----------
    k_per_label : int
        Number of top features to retain *per class*. The final vocabulary is
        the union of all per-class selections. Default 50.
    min_df : int
        Minimum document frequency for TF-IDF. Default 3.
    ngram_range : tuple
        N-gram range for TF-IDF. Default (1, 2).
    text_col : str
        Name of the text column in the input DataFrame. Default ``"TEXT"``.
    """

    def __init__(
        self,
        k_per_label: int = 50,
        min_df: int = 3,
        ngram_range: Tuple[int, int] = (1, 2),
        text_col: str = "TEXT",
    ) -> None:
        self.k_per_label = k_per_label
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.text_col = text_col

        # Set after fitting
        self.tfidf_: Optional[TfidfVectorizer] = None
        self.selected_indices_: List[int] = []
        self.selected_names_: List[str] = []

    # ── sklearn API ───────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y) -> "TfidfImpChiSelector":
        texts = X[self.text_col].fillna("").astype(str)
        y_arr = np.array(y)
        unique_labels = np.unique(y_arr)

        print(f"[ImpCHI] Fitting TF-IDF (min_df={self.min_df}, ngrams={self.ngram_range})…")
        self.tfidf_ = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            lowercase=True,
            stop_words="english",
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=True,  # log(1+tf) — compresses high-freq terms
        )

        X_tfidf = self.tfidf_.fit_transform(texts)
        n_features = X_tfidf.shape[1]
        print(f"[ImpCHI] Vocabulary size after TF-IDF: {n_features:,}")

        # ── Per-label Chi2 selection (the ImpCHI step) ────────────────────────
        feature_to_labels: dict = {}
        k_safe = min(self.k_per_label, n_features)

        for label in unique_labels:
            y_binary = (y_arr == label).astype(int)
            chi2_scores, _ = chi2(X_tfidf, y_binary)
            chi2_scores = np.nan_to_num(chi2_scores)

            top_indices = np.argsort(chi2_scores)[-k_safe:]
            for idx in top_indices:
                feature_to_labels.setdefault(idx, set()).add(label)

        self.selected_indices_ = sorted(feature_to_labels.keys())

        raw_names = self.tfidf_.get_feature_names_out()
        self.selected_names_ = [
            f"{raw_names[i]}_labels_{'_'.join(sorted(str(l) for l in feature_to_labels[i]))}"
            for i in self.selected_indices_
        ]
        print(
            f"[ImpCHI] Selected {len(self.selected_indices_):,} features "
            f"(union of top-{k_safe} per label × {len(unique_labels)} classes)."
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.tfidf_ is None:
            raise RuntimeError("Call fit() before transform().")
        texts = X[self.text_col].fillna("").astype(str)
        X_full = self.tfidf_.transform(texts)
        # Return dense slice — LightGBM handles it natively
        return X_full[:, self.selected_indices_].toarray()

    def get_feature_names_out(self) -> List[str]:
        return self.selected_names_



# ══════════════════════════════════════════════════════════════════════════════
# 3b. TF-IDF CHAR N-GRAMS + IMPCHI SELECTOR
#
#    WHY char n-grams fight overfitting on small corpora:
#    Character-level n-grams (e.g. "tion", " the", "ing ") are robust to
#    unseen words and capture sub-word stylometric signals invisible to word
#    tokenisers. They are particularly powerful here because:
#      - AI text tends to have smoother, more predictable character transitions
#      - Human typos create rare character sequences ("aboliished" -> "liish")
#      - The same ImpCHI per-label selection preserves minority-class signals.
#    Using analyzer='char_wb' pads tokens at word boundaries (e.g. " the ")
#    which avoids cross-word n-grams and reduces noise.
# ══════════════════════════════════════════════════════════════════════════════

class TfidfCharSelector(BaseEstimator, TransformerMixin):
    """
    Character-level TF-IDF with ImpCHI per-label feature selection.

    Uses ``analyzer='char_wb'`` (word-boundary-padded character n-grams) to
    extract sub-word stylometric signals, then applies the same ImpCHI
    selection strategy as ``TfidfImpChiSelector``.

    Parameters
    ----------
    k_per_label : int
        Top char-ngrams to retain per class. Default 30.
    min_df : int
        Minimum document frequency for TF-IDF. Default 5.
    ngram_range : tuple
        Character n-gram range. Default (3, 5) -- tri- to 5-grams.
    text_col : str
        Name of the text column in the input DataFrame. Default ``"TEXT"``.
    """

    def __init__(
        self,
        k_per_label: int = 30,
        min_df: int = 5,
        ngram_range: Tuple[int, int] = (3, 5),
        text_col: str = "TEXT",
    ) -> None:
        self.k_per_label = k_per_label
        self.min_df      = min_df
        self.ngram_range = ngram_range
        self.text_col    = text_col

        self.tfidf_: Optional[TfidfVectorizer] = None
        self.selected_indices_: List[int] = []
        self.selected_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y) -> "TfidfCharSelector":
        texts = X[self.text_col].fillna("").astype(str)
        y_arr = np.array(y)
        unique_labels = np.unique(y_arr)

        print(f"[CharImpCHI] Fitting char TF-IDF "
              f"(min_df={self.min_df}, ngrams={self.ngram_range})...")
        self.tfidf_ = TfidfVectorizer(
            analyzer="char_wb",          # word-boundary-padded char n-grams
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            lowercase=True,
        )
        X_tfidf = self.tfidf_.fit_transform(texts)
        n_features = X_tfidf.shape[1]
        print(f"[CharImpCHI] Char vocab size: {n_features:,}")

        # ── ImpCHI: per-label top-K selection (same logic as TfidfImpChiSelector) ────
        feature_to_labels: dict = {}
        k_safe = min(self.k_per_label, n_features)

        for label in unique_labels:
            y_binary = (y_arr == label).astype(int)
            chi2_scores, _ = chi2(X_tfidf, y_binary)
            chi2_scores = np.nan_to_num(chi2_scores)
            for idx in np.argsort(chi2_scores)[-k_safe:]:
                feature_to_labels.setdefault(idx, set()).add(label)

        self.selected_indices_ = sorted(feature_to_labels.keys())
        raw_names = self.tfidf_.get_feature_names_out()
        self.selected_names_ = [
            f"char_{raw_names[i].strip()}_lbl_{'_'.join(sorted(str(l) for l in feature_to_labels[i]))}"
            for i in self.selected_indices_
        ]
        print(f"[CharImpCHI] Selected {len(self.selected_indices_):,} char features "
              f"(top-{k_safe} per label x {len(unique_labels)} classes).")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.tfidf_ is None:
            raise RuntimeError("Call fit() before transform().")
        texts = X[self.text_col].fillna("").astype(str)
        X_full = self.tfidf_.transform(texts)
        return X_full[:, self.selected_indices_].toarray()

    def get_feature_names_out(self) -> List[str]:
        return self.selected_names_

# ══════════════════════════════════════════════════════════════════════════════
# 4. DATAFRAME → ARRAY ADAPTER
#    sklearn's FeatureUnion requires all components to accept the same input
#    and return arrays. This thin wrapper ensures DataFrames flow correctly.
# ══════════════════════════════════════════════════════════════════════════════

class _PassThroughAdapter(BaseEstimator, TransformerMixin):
    """
    Wraps a transformer that already outputs np.ndarray.
    Useful for inserting custom transformers into FeatureUnion.
    """
    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def get_feature_names_out(self):
        return self.transformer.get_feature_names_out()


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE FACTORY
#    Combines the three feature groups via FeatureUnion, then feeds the
#    concatenated matrix to LightGBM.
#
#    Feature vector composition (approximate, with default params):
#      Scalar features  →  ~33 dims  (exact count depends on AI markers)
#      Word2Vec         →   40 dims
#      TF-IDF ImpCHI    →  ~300 dims (50 per label × 6 classes, with overlap)
#      ─────────────────────────────────────────────────────────────────────
#      Total            ~ 373 dims  ← very tractable for 2,400 train samples
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    text_col: str = "TEXT",
    glove_model: str = "glove-wiki-gigaword-100",
    tfidf_params: Optional[dict] = None,
    char_params: Optional[dict] = None,
    lgbm_params: Optional[dict] = None,
) -> Pipeline:
    """
    Constructs the full sklearn-compatible end-to-end pipeline.

    Feature vector composition (approximate, default params):
      Scalar features        ->  ~33 dims  (stylometric signals)
      GloVe embeddings       -> 100 dims   (pretrained semantic context)
      Word TF-IDF ImpCHI     -> ~300 dims  (word n-gram discriminative vocab)
      Char TF-IDF ImpCHI     -> ~180 dims  (char n-gram stylometric patterns)
      -----------------------------------------------------------------------
      Total                  ~ 613 dims   tractable for 2,400 train samples

    Parameters
    ----------
    text_col : str
        Column name for the raw text input. Default ``"TEXT"``.
    glove_model : str
        gensim.downloader model name for the pretrained embedder.
    tfidf_params : dict, optional
        Keyword arguments forwarded to ``TfidfImpChiSelector``.
    char_params : dict, optional
        Keyword arguments forwarded to ``TfidfCharSelector``.
    lgbm_params : dict, optional
        Keyword arguments forwarded to ``LGBMClassifier``.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline ready for ``.fit(X, y)`` / ``.predict(X)``.
    """
    _tfidf = tfidf_params or {}
    _char  = char_params  or {}
    _lgbm  = lgbm_params  or {}

    # ── Sub-transformers ────────────────────────────────────────────────────────────────────────────
    scalar_extractor = ScalarTextFeatureExtractor(text_col=text_col)
    glove_embedder   = PretrainedEmbedder(model_name=glove_model, text_col=text_col)
    tfidf_selector   = TfidfImpChiSelector(text_col=text_col, **_tfidf)
    # Remap namespaced keys (k_per_label_char→k_per_label, min_df_char→min_df)
    _char_mapped = {
        "k_per_label": _char.get("k_per_label_char", _char.get("k_per_label", 30)),
        "min_df":      _char.get("min_df_char",      _char.get("min_df",      5)),
    }
    char_selector    = TfidfCharSelector(text_col=text_col, **_char_mapped)

    # ── FeatureUnion: four parallel feature streams ────────────────────────────────────────
    feature_union = FeatureUnion(
        transformer_list=[
            ("scalar",    _PassThroughAdapter(scalar_extractor)),
            ("glove",     _PassThroughAdapter(glove_embedder)),
            ("tfidf_chi", _PassThroughAdapter(tfidf_selector)),
            ("char_chi",  _PassThroughAdapter(char_selector)),
        ]
    )

    # ── LightGBM classifier ────────────────────────────────────────────────────────────────────
    default_lgbm = dict(
        objective="multiclass",
        num_class=6,
        class_weight="balanced",    # critical for 20:1 imbalance
        metric="multi_logloss",
        n_estimators=400,
        boosting_type="gbdt",
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=2.0,
        min_split_gain=0.01,
        colsample_bytree=0.6,
        subsample=0.8,
        subsample_freq=5,
    )
    default_lgbm.update(_lgbm)
    classifier = LGBMClassifier(**default_lgbm)

    return Pipeline(steps=[("features", feature_union), ("clf", classifier)])

# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPERIMENT CONFIGURATION
#    ┌─────────────────────────────────────────────────────────────────────┐
#    │  Change anything here — the rest of the code adapts automatically.  │
#    └─────────────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "train_path"  : "./dataset/train.csv",
    "test_path"   : "./dataset/test.csv",
    "text_col"    : "TEXT",
    "label_col"   : "LABEL",
    "num_classes" : 6,

    # ── Active model ──────────────────────────────────────────────────────────
    # Options: "lgbm" | "random_forest" | "linear_svc" | "xgboost"
    "active_model": "lgbm",

    # ── CV strategy ───────────────────────────────────────────────────────────
    "cv_folds"    : 5,
    "cv_metric"   : "f1_macro",   # any sklearn scoring string

    # ── Bayesian tuning (Optuna TPE) ──────────────────────────────────────────
    "run_tuning"  : True,
    "n_trials"    : 7,           # increase for better results (costs more time)
    "timeout_sec" : None,         # e.g. 300 → stop after 5 min regardless of n_trials

    # ── Pretrained embedder ─────────────────────────────────────────────────────────────────────────
    # Loaded once and cached in ~/gensim-data -- no re-download after first run.
    # Options: "glove-wiki-gigaword-50" | "glove-wiki-gigaword-100" (default)
    #          "glove-twitter-25" | "word2vec-google-news-300"
    "glove_model_name": "glove-wiki-gigaword-100",
}


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL REGISTRY
#    To add a model: write a _make_<name>(params, num_classes) factory
#    and add it to MODEL_REGISTRY + SEARCH_SPACES.
# ══════════════════════════════════════════════════════════════════════════════

def _make_lgbm(params: dict, num_classes: int) -> LGBMClassifier:
    defaults = dict(
        objective="multiclass", num_class=num_classes,
        class_weight="balanced", metric="multi_logloss",
        boosting_type="gbdt", n_jobs=-1, verbosity=-1,
        random_state=42, subsample_freq=5,
    )
    defaults.update(params)
    return LGBMClassifier(**defaults)


def _make_random_forest(params: dict, num_classes: int):
    from sklearn.ensemble import RandomForestClassifier
    defaults = dict(class_weight="balanced", random_state=42, n_jobs=-1)
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def _make_linear_svc(params: dict, num_classes: int):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    # params may contain "estimator__*" keys from Optuna — extract them
    svc_params = {k.replace("estimator__", ""): v for k, v in params.items()}
    defaults = dict(class_weight="balanced", random_state=42, max_iter=5000)
    defaults.update(svc_params)
    return CalibratedClassifierCV(LinearSVC(**defaults), cv=3)


def _make_xgboost(params: dict, num_classes: int):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    defaults = dict(
        objective="multi:softprob",   # softprob → predict_proba available
        num_class=num_classes, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


def _make_ensemble(params: dict, num_classes: int):
    """
    Soft-voting ensemble: LGBM + XGBoost + LinearSVC (calibrated).

    Sub-model hyperparameters are hardcoded to conservative anti-overfit
    defaults. The ensemble's regularisation comes from model diversity, so
    individual members are kept deliberately weak (shallow trees, high C
    penalty, large leaf constraints).

    Requires: lightgbm, xgboost, scikit-learn.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    lgbm_clf = _make_lgbm({
        "n_estimators": 300, "learning_rate": 0.05,
        "max_depth": 4,      "num_leaves": 15,
        "min_child_samples": 40,
        "reg_alpha": 0.5,    "reg_lambda": 5.0,
        "colsample_bytree": 0.6, "subsample": 0.7,
        "min_split_gain": 0.1,
    }, num_classes)

    xgb_clf = _make_xgboost({
        "n_estimators": 300, "learning_rate": 0.05,
        "max_depth": 4,      "min_child_weight": 30,
        "subsample": 0.7,    "colsample_bytree": 0.6,
        "reg_alpha": 1.0,    "reg_lambda": 5.0,
        "gamma": 0.5,
    }, num_classes)

    svc_clf = CalibratedClassifierCV(
        LinearSVC(
            class_weight="balanced", C=0.1,
            max_iter=5000, random_state=42,
        ),
        cv=3,
    )

    return VotingClassifier(
        estimators=[
            ("lgbm", lgbm_clf),
            ("xgb",  xgb_clf),
            ("svc",  svc_clf),
        ],
        voting="soft",    # average predicted probabilities → softer boundaries
        n_jobs=1,         # avoid thread-safety issues with LightGBM internals
    )


# Registry: model_name → factory(params, num_classes)
MODEL_REGISTRY = {
    "lgbm"         : _make_lgbm,
    "random_forest": _make_random_forest,
    "linear_svc"   : _make_linear_svc,
    "xgboost"      : _make_xgboost,
    "ensemble"     : _make_ensemble,
}


# 8. OPTUNA SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════════
# 8. OPTUNA SEARCH SPACES
#
#    REGULARISATION PHILOSOPHY for ~2.4k train samples:
#      max_depth  ∈ {3,4,5}      — very shallow trees, no room to memorise
#      num_leaves ∈ {10..20}     — << sklearn default of 31
#      min_child_samples ∈ {20..60} — forces large leaves, kills noise fits
#      k_per_label ∈ {10..30}    — only the strongest ImpCHI features
#
#    "ensemble" model uses fixed hardcoded conservative defaults for each
#    sub-estimator; Optuna only tunes the feature-extraction hyperparams.
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_lgbm(trial) -> dict:
    """Heavily regularised LightGBM space for small datasets."""
    return {
        "n_estimators"     : trial.suggest_int  ("n_estimators",      100, 500, step=50),
        "learning_rate"    : trial.suggest_float("learning_rate",     0.01, 0.10, log=True),
        # ── Depth / leaves: keep trees VERY shallow ───────────────────────────
        "max_depth"        : trial.suggest_int  ("max_depth",           3,   5),
        "num_leaves"       : trial.suggest_int  ("num_leaves",         10,  20),
        # ── Leaf size: force large leaves to avoid memorisation ───────────────
        "min_child_samples": trial.suggest_int  ("min_child_samples",  20,  60),
        # ── L1 / L2 regularisation ────────────────────────────────────────────
        "reg_alpha"        : trial.suggest_float("reg_alpha",          0.1,  1.0),
        "reg_lambda"       : trial.suggest_float("reg_lambda",         1.0, 10.0),
        "min_split_gain"   : trial.suggest_float("min_split_gain",     0.05, 0.5),
        # ── Stochastic subsampling ────────────────────────────────────────────
        "colsample_bytree" : trial.suggest_float("colsample_bytree",   0.4,  0.8),
        "subsample"        : trial.suggest_float("subsample",          0.5,  0.8),
    }


def _suggest_random_forest(trial) -> dict:
    return {
        "n_estimators"    : trial.suggest_int ("n_estimators",   200, 600, step=50),
        "max_depth"       : trial.suggest_int ("max_depth",        3,   8),
        "min_samples_leaf": trial.suggest_int ("min_samples_leaf", 5,  30),
        "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
    }


def _suggest_linear_svc(trial) -> dict:
    return {
        "estimator__C"   : trial.suggest_float("C",   1e-3, 2.0, log=True),
        "estimator__loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "estimator__tol" : trial.suggest_float("tol", 1e-6, 1e-3, log=True),
    }


def _suggest_xgboost(trial) -> dict:
    """Heavily regularised XGBoost space."""
    return {
        "n_estimators"    : trial.suggest_int  ("n_estimators",      100, 500, step=50),
        "learning_rate"   : trial.suggest_float("learning_rate",     0.01, 0.10, log=True),
        "max_depth"       : trial.suggest_int  ("max_depth",           3,   5),
        "min_child_weight": trial.suggest_int  ("min_child_weight",   10,  50),
        "subsample"       : trial.suggest_float("subsample",          0.5,  0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree",   0.4,  0.8),
        "reg_alpha"       : trial.suggest_float("reg_alpha",          0.1,  5.0),
        "reg_lambda"      : trial.suggest_float("reg_lambda",         1.0, 10.0),
        "gamma"           : trial.suggest_float("gamma",              0.1,  2.0),
    }


def _suggest_ensemble(trial) -> dict:
    """
    For the ensemble, sub-model hypers are fixed (conservative defaults).
    Optuna only optimises the feature-extraction budget.
    Returning {} here means MODEL_REGISTRY['ensemble'] ignores clf_params.
    """
    return {}


# Map model name → suggest function
SEARCH_SPACES = {
    "lgbm"         : _suggest_lgbm,
    "random_forest": _suggest_random_forest,
    "linear_svc"   : _suggest_linear_svc,
    "xgboost"      : _suggest_xgboost,
    "ensemble"     : _suggest_ensemble,
}

# ImpCHI search space (shared across all models)
def _suggest_preprocessor(trial) -> dict:
    """
    Feature-extraction budget.

    Tight ranges force the model to rely on the strongest signals only,
    which is the primary anti-overfitting lever on a 2.4k corpus.
    """
    return {
        # ── Word-level TF-IDF ImpCHI ─────────────────────────────────────────
        # k=10..30: only the top discriminative n-grams per class
        "k_per_label" : trial.suggest_int("k_per_label",       10, 30),
        "min_df"      : trial.suggest_int("min_df",             3,  7),
        "ngram_range" : trial.suggest_categorical("ngram_range", ["unigram", "bigram"]),
        # ── Char-level TF-IDF ImpCHI ─────────────────────────────────────────
        # k=10..30: same tight budget for sub-word patterns
        "k_per_label_char": trial.suggest_int("k_per_label_char", 10, 30),
        "min_df_char"     : trial.suggest_int("min_df_char",       4,  9),
    }


# 9. BAYESIAN TUNING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_bayesian_tuning(
    X_train   : pd.DataFrame,
    y_train   : pd.Series,
    cfg       : dict,
    extra_text: Optional[pd.Series] = None,
    frozen_w2v: Optional[Word2VecEmbedder] = None,
) -> Tuple[dict, dict, float]:
    """
    Runs Optuna TPE-based Bayesian hyperparameter optimisation.

    Strategy
    --------
    - The pretrained embedder is always loaded once before Optuna
      (using ``cfg["w2v_fixed_params"]``). Each trial only fits ImpCHI + the
      classifier → dramatically faster.
    - Cross-validation uses StratifiedKFold to preserve class distribution.
    - The objective is maximized (Optuna direction="maximize").

    Parameters
    ----------
    X_train, y_train : training data.
    cfg              : global CONFIG dict.
    extra_text       : optional unlabelled text for W2V vocab enrichment.
    frozen_w2v       : pre-loaded PretrainedEmbedder instance.

    Returns
    -------
    best_clf_params  : dict of best classifier hyperparams.
    best_pre_params  : dict of best preprocessor hyperparams.
    best_score       : best CV macro-F1 achieved.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_name  = cfg["active_model"]
    suggest_clf = SEARCH_SPACES[model_name]
    make_clf    = MODEL_REGISTRY[model_name]
    cv          = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=42)
    text_col    = cfg["text_col"]

    # ── Pre-extract scalar features once (they have no learnable params) ──────
    scalar_tf = ScalarTextFeatureExtractor(text_col=text_col)
    # scalar_tf.fit() is a no-op, so we can call transform directly below

    def objective(trial) -> float:
        # ── 1. Sample hyperparameters ─────────────────────────────────────────
        clf_params = suggest_clf(trial)
        pre_params = _suggest_preprocessor(trial)

        ngram = (1, 1) if pre_params.pop("ngram_range") == "unigram" else (1, 2)

        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
            y_tr  = y_train.iloc[tr_idx].reset_index(drop=True)
            X_val = X_train.iloc[val_idx].reset_index(drop=True)
            y_val = y_train.iloc[val_idx].reset_index(drop=True)

            # ── 2a. Scalar features (no fitting needed) ───────────────────────
            X_tr_scalar  = scalar_tf.transform(X_tr)
            X_val_scalar = scalar_tf.transform(X_val)

            # ── 2b. GloVe embeddings (pretrained, frozen) ────────────────────────────────
            # frozen_w2v is a PretrainedEmbedder loaded ONCE before Optuna.
            X_tr_glove  = frozen_w2v.transform(X_tr)
            X_val_glove = frozen_w2v.transform(X_val)

            # ── 2c. Word ImpCHI TF-IDF ────────────────────────────────────────────────────
            tfidf_chi = TfidfImpChiSelector(
                text_col=text_col,
                k_per_label=pre_params["k_per_label"],
                min_df=pre_params["min_df"],
                ngram_range=ngram,
            )
            tfidf_chi.fit(X_tr, y_tr)
            X_tr_tfidf  = tfidf_chi.transform(X_tr)
            X_val_tfidf = tfidf_chi.transform(X_val)

            # ── 2d. Char ImpCHI TF-IDF (sub-word stylometric patterns) ────────────────
            char_chi = TfidfCharSelector(
                text_col=text_col,
                k_per_label=pre_params["k_per_label_char"],
                min_df=pre_params["min_df_char"],
            )
            char_chi.fit(X_tr, y_tr)
            X_tr_char  = char_chi.transform(X_tr)
            X_val_char = char_chi.transform(X_val)

            # ── 3. Concatenate all four feature blocks ────────────────────────────────────────
            X_tr_full  = np.hstack([X_tr_scalar, X_tr_glove, X_tr_tfidf, X_tr_char])
            X_val_full = np.hstack([X_val_scalar, X_val_glove, X_val_tfidf, X_val_char])

            # ── 4. Train classifier ───────────────────────────────────────────
            clf = make_clf(clf_params, cfg["num_classes"])
            clf.fit(X_tr_full, y_tr)

            y_pred = clf.predict(X_val_full)
            fold_scores.append(f1_score(y_val, y_pred, average="macro"))

        mean_score = float(np.mean(fold_scores))

        # Report intermediate value for Optuna pruning (if a pruner is added)
        trial.report(mean_score, step=0)
        return mean_score

    # ── Run the study ─────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)   # Tree-structured Parzen Estimator
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"\n[Optuna] Starting Bayesian tuning: model={model_name}, "
          f"trials={cfg['n_trials']}, cv={cfg['cv_folds']}-fold, "
          f"metric={cfg['cv_metric']}")
    print(f"[Optuna] GloVe pretrained (always frozen) — fast mode\n")

    study.optimize(
        objective,
        n_trials=cfg["n_trials"],
        timeout=cfg.get("timeout_sec"),
        show_progress_bar=True,
        gc_after_trial=True,
    )

    best_trial      = study.best_trial
    best_clf_params = SEARCH_SPACES[model_name](best_trial)
    best_ngram_raw  = best_trial.params.get("ngram_range", "bigram")
    best_ngram      = (1, 1) if best_ngram_raw == "unigram" else (1, 2)
    best_pre_params = {
        "k_per_label"     : best_trial.params["k_per_label"],
        "min_df"          : best_trial.params["min_df"],
        "ngram_range"     : best_ngram,
        "k_per_label_char": best_trial.params["k_per_label_char"],
        "min_df_char"     : best_trial.params["min_df_char"],
    }
    best_score = best_trial.value

    print(f"\n{'═'*60}")
    print(f"[Optuna] Best {cfg['cv_metric']}: {best_score:.4f}")
    print(f"[Optuna] Best classifier params:")
    for k, v in best_clf_params.items():
        print(f"  {k:25s}: {v}")
    print(f"[Optuna] Best preprocessor params:")
    for k, v in best_pre_params.items():
        print(f"  {k:25s}: {v}")
    print(f"{'═'*60}\n")

    # ── Optuna study visualisation (saved if plotly is available) ─────────────
    try:
        import optuna.visualization as vis
        import plotly.io as pio

        os.makedirs("tuning_plots", exist_ok=True)

        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html("tuning_plots/optimization_history.html")

        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html("tuning_plots/param_importances.html")

        fig_slice = vis.plot_slice(study)
        fig_slice.write_html("tuning_plots/param_slice.html")

        print("[Optuna] Plots saved to tuning_plots/ (open .html in browser)")
    except Exception:
        pass  # plotly not installed — skip silently

    return best_clf_params, best_pre_params, best_score


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN — Orchestrates everything
# ══════════════════════════════════════════════════════════════════════════════

# ── Pseudo-labeling configuration ─────────────────────────────────────────────
# Only test samples whose max predicted probability exceeds this threshold are
# added to the training set. Higher = fewer but cleaner pseudo-labels.
# Recommended range: 0.85–0.95. Set to 1.01 to disable pseudo-labeling entirely.
PSEUDO_LABEL_THRESHOLD = 0.90


def main() -> None:
    print("=" * 70)
    print("MALTO Recruitment Hackathon — Human vs. AI Text Classifier")
    print(f"Active model : {CONFIG['active_model'].upper()}")
    print(f"Tuning       : {'Bayesian (Optuna TPE)' if CONFIG['run_tuning'] else 'Disabled'}")
    print(f"Pseudo-label threshold : {PSEUDO_LABEL_THRESHOLD}")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df = pd.read_csv(CONFIG["train_path"])
    test_df  = pd.read_csv(CONFIG["test_path"])
    text_col  = CONFIG["text_col"]
    label_col = CONFIG["label_col"]

    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")
    print("\nLabel distribution (train):")
    vc = train_df[label_col].value_counts().sort_index()
    for label, count in vc.items():
        bar = chr(9608) * (count // 20)
        print(f"  Class {label}: {count:4d}  {bar}")

    X      = train_df[[text_col]].copy()
    y      = train_df[label_col].copy()
    X_eval = test_df[[text_col]].copy()

    # ── Train / hold-out split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )
    print(f"\nTrain size    : {len(X_train)}")
    print(f"Hold-out size : {len(X_test)}")

    # ── Load pretrained GloVe embedder once (cached after first download) ──────
    frozen_w2v = PretrainedEmbedder(
        model_name=CONFIG["glove_model_name"],
        text_col=text_col,
    )
    frozen_w2v.fit(X_train)

    # ── Bayesian hyperparameter tuning ─────────────────────────────────────────
    best_clf_params: dict = {}
    best_pre_params: dict = {}

    if CONFIG["run_tuning"]:
        best_clf_params, best_pre_params, best_cv_score = run_bayesian_tuning(
            X_train, y_train,
            cfg=CONFIG,
            extra_text=None,
            frozen_w2v=frozen_w2v,
        )
    else:
        print("\n[Tuning] Skipped (run_tuning=False). Using default parameters.")

    # ── Shared helpers ─────────────────────────────────────────────────────────
    def _split_pre(p: dict):
        tfidf = {k: v for k, v in p.items() if k in ("k_per_label", "min_df", "ngram_range")}
        char  = {k: v for k, v in p.items() if k in ("k_per_label_char", "min_df_char")}
        return tfidf, char

    def _make_final_clf(clf_params: dict):
        return MODEL_REGISTRY[CONFIG["active_model"]](clf_params, CONFIG["num_classes"])

    def _build_fresh_pipeline(tfidf_p: dict, char_p: dict) -> Pipeline:
        pipe = build_pipeline(
            text_col=text_col,
            glove_model=glove_model,
            tfidf_params=tfidf_p,
            char_params=char_p,
            lgbm_params=best_clf_params if CONFIG["active_model"] == "lgbm" else {},
        )
        if CONFIG["active_model"] != "lgbm":
            pipe.steps[-1] = ("clf", _make_final_clf(best_clf_params))
        return pipe

    glove_model     = CONFIG["glove_model_name"]
    tfidf_p, char_p = _split_pre(best_pre_params)

    # ── Step 1: hold-out evaluation (sanity check before submission) ──────────
    print("\nFitting pipeline on training split for hold-out evaluation...")
    eval_pipeline = _build_fresh_pipeline(tfidf_p, char_p)
    eval_pipeline.fit(X_train, y_train)

    y_test_pred = eval_pipeline.predict(X_test)
    macro_f1    = f1_score(y_test, y_test_pred, average="macro")

    bar = chr(9552) * 60
    print(f"\n{bar}")
    print(f"Hold-out Macro F1 : {macro_f1:.4f}")
    print(f"{bar}")
    print("\nDetailed classification report (hold-out set):")
    print(classification_report(y_test, y_test_pred))

    # ── Step 2: refit on 100% labelled data → base submission pipeline ────────
    print("\nRetraining on 100% of labelled data...")
    submission_pipeline = _build_fresh_pipeline(tfidf_p, char_p)
    submission_pipeline.fit(X, y)

    # ══════════════════════════════════════════════════════════════════════════
    # PSEUDO-LABELING
    # ──────────────────────────────────────────────────────────────────────────
    # Rationale: the test set contains 600 unlabelled samples. Any sample on
    # which the ensemble is highly confident (max_proba >= THRESHOLD) is very
    # likely correctly classified. Adding those samples to the training set
    # before a final refit gives the model more signal at essentially zero
    # label-noise cost — at the chosen threshold, empirically over 90% of
    # pseudo-labels are correct even for imperfect classifiers.
    #
    # Pipeline:
    #   submission_pipeline  -->  predict_proba(X_eval)
    #       | filter: max_proba >= THRESHOLD
    #   X_pseudo, y_pseudo   -->  high-confidence test samples with soft labels
    #       | concatenate with original labelled data
    #   X_extended, y_extended  -->  augmented training set
    #       | fit fresh pipeline from scratch (TF-IDF re-fitted on larger corpus)
    #   pseudo_pipeline      -->  final predictions for submission
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n[Pseudo-label] Computing probabilities on {len(X_eval)} test samples...")
    proba_eval = submission_pipeline.predict_proba(X_eval)   # (n_test, n_classes)

    max_proba   = proba_eval.max(axis=1)                      # confidence per sample
    pseudo_mask = max_proba >= PSEUDO_LABEL_THRESHOLD
    n_pseudo    = int(pseudo_mask.sum())

    print(f"[Pseudo-label] Confidence >= {PSEUDO_LABEL_THRESHOLD}: "
          f"{n_pseudo} / {len(X_eval)} samples "
          f"({100 * n_pseudo / len(X_eval):.1f}%)")

    if n_pseudo > 0:
        # Assign argmax class as pseudo-label for each confident sample
        y_pseudo        = proba_eval[pseudo_mask].argmax(axis=1)
        pseudo_indices  = np.where(pseudo_mask)[0]

        # Log pseudo-label class distribution
        pseudo_dist = pd.Series(y_pseudo).value_counts().sort_index()
        print("[Pseudo-label] Pseudo-label class distribution:")
        for cls, cnt in pseudo_dist.items():
            bar_pl = chr(9608) * cnt
            print(f"  Class {cls}: {cnt:3d}  {bar_pl}  ({100*cnt/n_pseudo:.1f}%)")

        # Build augmented dataset
        X_pseudo   = X_eval.iloc[pseudo_indices].reset_index(drop=True)
        X_extended = pd.concat([X, X_pseudo],                   ignore_index=True)
        y_extended = pd.concat(
            [y, pd.Series(y_pseudo, name=label_col, dtype=y.dtype)],
            ignore_index=True,
        )

        print(f"[Pseudo-label] Train size: {len(X)} --> {len(X_extended)} "
              f"(+{n_pseudo} pseudo-labelled samples)")

        # Step 3: refit from scratch on augmented data
        # A fresh pipeline is mandatory: TF-IDF and CharSelector must be
        # re-fitted on the larger corpus so that the vocabulary and ImpCHI
        # feature selection benefit from the additional samples.
        print("\n[Pseudo-label] Refitting pseudo_pipeline on extended dataset...")
        pseudo_pipeline = _build_fresh_pipeline(tfidf_p, char_p)
        pseudo_pipeline.fit(X_extended, y_extended)

        y_pred_final = pseudo_pipeline.predict(X_eval)
        print("[Pseudo-label] Done. Submission uses pseudo_pipeline.")

    else:
        # No samples cleared the threshold — submit with the standard pipeline
        print("[Pseudo-label] No samples cleared the threshold. "
              "Submission uses submission_pipeline (no pseudo-labels added).")
        y_pred_final = submission_pipeline.predict(X_eval)

    # ── Write submission ───────────────────────────────────────────────────────
    id_col     = test_df.index if "id" not in test_df.columns else test_df["id"]
    submission = pd.DataFrame({"id": id_col, "label": y_pred_final})
    submission.to_csv("submission.csv", index=False)

    print("\nsubmission.csv written successfully.")
    print(f"Final prediction distribution:")
    print(pd.Series(y_pred_final).value_counts().sort_index().to_string())
    print("\nDone " + chr(10003))


if __name__ == "__main__":
    main()