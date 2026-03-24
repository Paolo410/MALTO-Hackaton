import pandas as pd
import numpy as np
import gensim.downloader as gensim_dl
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

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