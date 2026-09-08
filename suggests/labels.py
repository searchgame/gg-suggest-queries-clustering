"""Cluster naming: c-TF-IDF first, an optional LLM pass on top.

c-TF-IDF replaces the old raw-frequency counts. It also defuses a trap the seed
grid creates: once you seed with "comment", a plain word count puts *comment*
at the top of every French cluster. Weighting each cluster's terms against the
whole corpus down-weights anything that appears everywhere, so labelling stops
depending on a perfect stopword list.
"""

from __future__ import annotations

import re

import numpy as np

from . import ollama
from .modifiers import stopwords
from .text import tokenize

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "nl": "Dutch",
    "it": "Italian",
    "es": "Spanish",
}

_MAX_LABEL_WORDS = 8
_QUOTES = "\"'“”«»‘’"


def ctfidf_labels(
    queries: list[str],
    labels: np.ndarray,
    language: str,
    *,
    top_k: int = 4,
) -> dict[int, str]:
    """Return ``{cluster_id: "term, term, term"}``."""
    from sklearn.feature_extraction.text import CountVectorizer

    cluster_ids = sorted({int(c) for c in labels if c >= 0})
    if not cluster_ids:
        return {}

    documents = [
        " ".join(q for q, c in zip(queries, labels) if int(c) == cid) for cid in cluster_ids
    ]

    stopset = stopwords(language)

    def analyzer(document: str) -> list[str]:
        # Stopwords are filtered here rather than via CountVectorizer's
        # stop_words argument, which warns about inconsistency whenever a
        # custom tokenizer is also supplied.
        tokens = [t for t in tokenize(document) if len(t) > 1 and t not in stopset]
        bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    vectorizer = CountVectorizer(analyzer=analyzer, min_df=1)
    try:
        counts = vectorizer.fit_transform(documents).toarray().astype(float)
    except ValueError:
        # Every document was stopwords only.
        return {cid: "" for cid in cluster_ids}

    vocabulary = np.array(vectorizer.get_feature_names_out())

    # c-TF-IDF: term frequency within the cluster, weighted against how widely
    # the term is used across all clusters.
    per_cluster_total = counts.sum(axis=1, keepdims=True)
    per_cluster_total[per_cluster_total == 0] = 1.0
    tf = counts / per_cluster_total

    per_term_total = counts.sum(axis=0)
    per_term_total[per_term_total == 0] = 1.0
    average_words = counts.sum() / max(len(documents), 1)
    idf = np.log1p(average_words / per_term_total)

    scores = tf * idf

    out: dict[int, str] = {}
    for row, cid in enumerate(cluster_ids):
        ranked = vocabulary[np.argsort(scores[row])[::-1]]
        out[cid] = ", ".join(_pick_terms(ranked, top_k))
    return out


def _pick_terms(ranked: np.ndarray, top_k: int) -> list[str]:
    """Take the best terms, dropping unigrams already covered by a bigram."""
    chosen: list[str] = []
    for term in ranked[: top_k * 4]:
        if any(term != other and term in other.split() for other in chosen):
            continue
        chosen = [c for c in chosen if not (c != term and c in term.split())]
        chosen.append(str(term))
        if len(chosen) >= top_k:
            break
    return chosen


def centroid_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
) -> dict[int, list[int]]:
    """Row indices of the k queries closest to each cluster centroid.

    Vectorized, replacing the old per-row ``iterrows`` loop.
    """
    out: dict[int, list[int]] = {}
    for cid in sorted({int(c) for c in labels if c >= 0}):
        members = np.flatnonzero(labels == cid)
        if members.size == 0:
            continue
        vectors = embeddings[members]
        centroid = vectors.mean(axis=0)
        order = np.linalg.norm(vectors - centroid, axis=1).argsort()
        out[cid] = [int(members[i]) for i in order[:k]]
    return out


def llm_label(
    terms: str,
    examples: list[str],
    language: str,
    model: str,
    host: str,
    *,
    temperature: float = 0.2,
) -> str | None:
    """Ask a local model for a short label. ``None`` means fall back.

    Naming is a garnish, never a dependency: every failure path here returns
    None so the caller keeps its c-TF-IDF label.
    """
    if not model:
        return None

    language_name = LANGUAGE_NAMES.get(language, language)
    sample = "\n".join(f"- {q}" for q in examples[:10])
    prompt = (
        "You name clusters of Google search queries.\n\n"
        f"Language: {language_name}\n"
        f"Key terms: {terms}\n"
        f"Example queries:\n{sample}\n\n"
        f"Reply with ONLY a short topic label of 3 to 6 words, written in "
        f"{language_name}. No quotes, no punctuation at the end, no explanation."
    )

    try:
        raw = ollama.generate(
            prompt, model, host, temperature=temperature, num_predict=40
        )
    except ollama.OllamaError:
        return None

    return _clean_label(raw)


def _clean_label(raw: str) -> str | None:
    """Salvage a usable label from a small model's output, or give up.

    llama3.2 at 3B will sometimes answer with a sentence or a preamble; better
    to reject that and keep the c-TF-IDF label than to show it.
    """
    lines = raw.strip().splitlines()
    if not lines:
        return None

    label = lines[0].strip().strip(_QUOTES).strip()
    label = re.sub(r"^(label|topic|cluster)\s*[:\-]\s*", "", label, flags=re.I)
    label = label.strip(_QUOTES + " .,;:").strip()

    if not label or len(label.split()) > _MAX_LABEL_WORDS:
        return None
    return label
