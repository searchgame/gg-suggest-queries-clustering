"""One Ollama client for all three endpoints the app uses.

``labels.py`` calls :func:`generate` from here rather than opening its own HTTP
connection, so host configuration and error handling live in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import requests

from .config import DEFAULT_OLLAMA_HOST

EMBED_TIMEOUT = 300
GENERATE_TIMEOUT = 90
TAGS_TIMEOUT = 5


class OllamaError(RuntimeError):
    """Ollama is unreachable, or answered with something unusable."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    capabilities: tuple[str, ...]

    @property
    def can_embed(self) -> bool:
        return "embedding" in self.capabilities

    @property
    def can_generate(self) -> bool:
        return "completion" in self.capabilities


def ping(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """True if an Ollama server answers at ``host``."""
    try:
        requests.get(f"{host.rstrip('/')}/api/tags", timeout=TAGS_TIMEOUT).raise_for_status()
    except requests.RequestException:
        return False
    return True


def list_models(host: str = DEFAULT_OLLAMA_HOST) -> list[ModelInfo]:
    """Every installed model, tagged with what it can actually do.

    ``/api/show`` reports a ``capabilities`` array on current Ollama builds.
    Older ones omit it, so fall back to the naming convention - which is only a
    heuristic, and the reason both model fields stay editable in the UI.
    """
    base = host.rstrip("/")
    try:
        response = requests.get(f"{base}/api/tags", timeout=TAGS_TIMEOUT)
        response.raise_for_status()
        names = [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise OllamaError(f"Could not reach Ollama at {host}: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise OllamaError(f"Unexpected /api/tags response from {host}: {exc}") from exc

    return sorted(
        (ModelInfo(name, _capabilities(base, name)) for name in names),
        key=lambda m: m.name,
    )


def _capabilities(base: str, name: str) -> tuple[str, ...]:
    try:
        response = requests.post(f"{base}/api/show", json={"model": name}, timeout=TAGS_TIMEOUT)
        response.raise_for_status()
        caps = response.json().get("capabilities")
        if caps:
            return tuple(caps)
    except (requests.RequestException, ValueError):
        pass
    # Heuristic fallback for builds that do not report capabilities.
    return ("embedding",) if "embed" in name.lower() else ("completion",)


def embedding_models(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    return [m.name for m in list_models(host) if m.can_embed]


def completion_models(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    return [m.name for m in list_models(host) if m.can_generate]


def embed(
    texts: Sequence[str],
    model: str,
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed texts, returning an L2-normalized ``(n, d)`` float32 matrix.

    Normalizing here is what makes euclidean distance rank-equivalent to cosine
    for everything downstream, and it is the fix for the old metric mismatch.
    Callers can rely on it; nothing else re-normalizes.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    url = f"{host.rstrip('/')}/api/embed"
    chunks: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        try:
            response = requests.post(
                url, json={"model": model, "input": batch}, timeout=EMBED_TIMEOUT
            )
            response.raise_for_status()
            vectors = response.json()["embeddings"]
        except requests.RequestException as exc:
            raise OllamaError(f"Embedding request failed: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise OllamaError(f"Malformed embedding response: {exc}") from exc

        if len(vectors) != len(batch):
            raise OllamaError(
                f"Ollama returned {len(vectors)} vectors for {len(batch)} inputs."
            )
        chunks.append(np.asarray(vectors, dtype=np.float32))

        if progress is not None:
            progress(min(start + batch_size, len(texts)), len(texts))

    matrix = np.vstack(chunks)
    return l2_normalize(matrix)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, np.where(norms == 0.0, 1.0, norms), out=matrix)
    return matrix


def generate(
    prompt: str,
    model: str,
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    temperature: float = 0.2,
    num_predict: int = 40,
) -> str:
    """Single-shot completion. Raises :class:`OllamaError` on any failure."""
    try:
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            timeout=GENERATE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise OllamaError(f"Generation request failed: {exc}") from exc
    except ValueError as exc:
        raise OllamaError(f"Malformed generation response: {exc}") from exc
