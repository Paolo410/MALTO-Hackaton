import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from typing import List, Tuple, Optional

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