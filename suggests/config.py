"""Typed settings and the country/language tables.

Country and language stay decoupled on purpose: ``gl=CH`` legitimately pairs
with de, fr or it, and ``gl=BE`` with nl, fr or de. The presets below suggest a
default per country; they never force one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Google's `gl` parameter takes ISO 3166-1 alpha-2. The old app sent "UK",
# which is not a valid code - the United Kingdom is "GB".
COUNTRIES: dict[str, str] = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "FR": "France",
    "BE": "Belgium",
    "CH": "Switzerland",
    "DE": "Germany",
    "AT": "Austria",
    "NL": "Netherlands",
    "IT": "Italy",
    "ES": "Spain",
}

LANGUAGES: dict[str, str] = {
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "nl": "Nederlands",
    "it": "Italiano",
    "es": "Español",
}

# First entry is the default offered for that country; the rest are the other
# languages that country plausibly searches in.
COUNTRY_LANGUAGES: dict[str, tuple[str, ...]] = {
    "US": ("en", "es"),
    "GB": ("en",),
    "CA": ("en", "fr"),
    "FR": ("fr", "en"),
    "BE": ("nl", "fr", "de"),
    "CH": ("de", "fr", "it"),
    "DE": ("de", "en"),
    "AT": ("de",),
    "NL": ("nl", "en"),
    "IT": ("it", "en"),
    "ES": ("es", "en"),
}


def languages_for(country: str) -> tuple[str, ...]:
    """Suggested languages for a country, best guess first."""
    preset = COUNTRY_LANGUAGES.get(country, ())
    rest = tuple(code for code in LANGUAGES if code not in preset)
    return preset + rest


@dataclass(frozen=True)
class FetchConfig:
    """Everything that changes which suggestions come back."""

    country: str = "US"
    language: str = "en"
    categories: tuple[str, ...] = ("questions",)
    letter_positions: tuple[str, ...] = ("suffix",)
    include_digits: bool = False
    include_unaccented: bool = False
    max_workers: int = 8


@dataclass(frozen=True)
class ClusterConfig:
    """HDBSCAN plus the reduction it runs on top of."""

    min_cluster_size: int = 3
    min_samples: int = 2
    selection_method: str = "eom"
    selection_epsilon: float = 0.0
    n_components: int = 10
    n_neighbors: int = 15
    soft_assign: bool = False
    soft_threshold: float = 0.10
    random_state: int = 42


@dataclass(frozen=True)
class VizConfig:
    """The separate 2-d projection used for the map."""

    n_neighbors: int = 15
    min_dist: float = 0.10
    random_state: int = 42


@dataclass(frozen=True)
class LabelConfig:
    """Cluster naming. c-TF-IDF always runs; the LLM layers on top."""

    top_k: int = 4
    n_representatives: int = 3
    use_llm: bool = False
    inference_model: str = ""
    temperature: float = 0.2


@dataclass
class RunParams:
    """A snapshot of everything the Run button froze."""

    query: str
    fetch: FetchConfig
    embedding_model: str
    host: str = DEFAULT_OLLAMA_HOST
    extras: dict = field(default_factory=dict)
