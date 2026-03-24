import pandas as pd
import numpy as np
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


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