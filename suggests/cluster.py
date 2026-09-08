"""Clustering, in the space it should have been running in all along.

HDBSCAN on raw 1024-d embeddings is why the old app dumped so much into the
outlier bucket: density estimates degrade badly at that dimensionality. Here
UMAP reduces to ~10 components under a cosine metric first, and HDBSCAN runs
euclidean on top of that. A *separate* 2-d projection drives the map.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .config import ClusterConfig, VizConfig

# UMAP warns that a fixed random_state disables its parallelism. That is the
# trade we want - a map that stops moving between runs - and irrelevant at a
# few hundred points.
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    dbcv: float | None = None
    n_reassigned: int = 0
    message: str | None = None

    @property
    def n_clusters(self) -> int:
        return int((np.unique(self.labels) >= 0).sum())

    @property
    def n_outliers(self) -> int:
        return int((self.labels < 0).sum())


def reduce_dimensions(
    embeddings: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    random_state: int = 42,
    metric: str = "cosine",
) -> np.ndarray:
    """UMAP with the guards the old code was missing.

    ``n_neighbors=15`` against 12 points misbehaves, and UMAP requires
    ``n_components < n_samples``. Both are clamped rather than left to blow up.
    """
    import umap

    n = len(embeddings)
    if n < 5:
        return np.asarray(embeddings, dtype=np.float32)

    n_neighbors = max(2, min(n_neighbors, n - 1))
    n_components = max(2, min(n_components, n - 2))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def cluster_embeddings(embeddings: np.ndarray, cfg: ClusterConfig) -> ClusterResult:
    """Reduce, cluster, optionally rescue outliers, and score the result."""
    import hdbscan

    n = len(embeddings)
    if n < max(4, cfg.min_cluster_size * 2):
        return ClusterResult(
            labels=np.full(n, -1, dtype=int),
            probabilities=np.zeros(n, dtype=float),
            outlier_scores=np.zeros(n, dtype=float),
            message=(
                f"Only {n} queries: too few to cluster at a minimum size of "
                f"{cfg.min_cluster_size}. Widen the seed grid or lower the minimum."
            ),
        )

    reduced = reduce_dimensions(
        embeddings,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=0.0,
        random_state=cfg.random_state,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
        cluster_selection_method=cfg.selection_method,
        cluster_selection_epsilon=float(cfg.selection_epsilon),
        prediction_data=True,
        gen_min_span_tree=True,  # required for relative_validity_
    )
    labels = clusterer.fit_predict(reduced).astype(int)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    outlier_scores = np.asarray(
        getattr(clusterer, "outlier_scores_", np.zeros(n)), dtype=float
    )
    outlier_scores = np.nan_to_num(outlier_scores, nan=0.0)

    n_reassigned = 0
    if cfg.soft_assign and (labels >= 0).any():
        labels, probabilities, n_reassigned = _soft_assign(
            clusterer, labels, probabilities, cfg.soft_threshold
        )

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
        dbcv=_dbcv(clusterer),
        n_reassigned=n_reassigned,
    )


def _soft_assign(
    clusterer,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rescue outliers using the prediction data we already paid to compute.

    The old app set ``prediction_data=True`` and never read it.
    """
    import hdbscan

    try:
        memberships = hdbscan.all_points_membership_vectors(clusterer)
    except (ValueError, AttributeError, IndexError):
        return labels, probabilities, 0

    memberships = np.atleast_2d(np.asarray(memberships, dtype=float))
    if memberships.shape[0] != len(labels) or memberships.size == 0:
        return labels, probabilities, 0

    labels = labels.copy()
    probabilities = probabilities.copy()
    outliers = np.flatnonzero(labels < 0)
    if outliers.size == 0:
        return labels, probabilities, 0

    best = memberships[outliers].argmax(axis=1)
    strength = memberships[outliers].max(axis=1)
    accepted = strength >= threshold

    labels[outliers[accepted]] = best[accepted]
    probabilities[outliers[accepted]] = strength[accepted]
    return labels, probabilities, int(accepted.sum())


def _dbcv(clusterer) -> float | None:
    """Density-based cluster validity, or None when it is not meaningful."""
    try:
        score = float(clusterer.relative_validity_)
    except (AttributeError, ValueError, TypeError):
        return None
    return score if np.isfinite(score) else None


def project(embeddings: np.ndarray, cfg: VizConfig) -> np.ndarray:
    """The 2-d map. Seeded, so it stops moving between reruns."""
    n = len(embeddings)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n < 5:
        coords = np.zeros((n, 2), dtype=np.float32)
        coords[:, 0] = np.arange(n, dtype=np.float32)
        return coords

    return reduce_dimensions(
        embeddings,
        n_components=2,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        random_state=cfg.random_state,
    )
