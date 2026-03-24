import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from typing import List, Tuple, Optional

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