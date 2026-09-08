"""Language modifier registry and the seed grid built from it.

The registry is data (``data/modifiers.json``), not code, because the term
lists change far more often than the pipeline does. It is validated at load so
a bad edit fails loudly at startup rather than silently producing a thin grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .text import collapse_ws, dedupe, fold_key, nfc, strip_accents

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "modifiers.json"

VALID_POSITIONS = ("prefix", "suffix", "both")
DIGITS = "0123456789"


class RegistryError(RuntimeError):
    """The modifier registry is missing or malformed."""


@dataclass(frozen=True)
class Seed:
    """One query to send to Google, with the modifier that produced it."""

    text: str
    category: str
    position: str


@lru_cache(maxsize=1)
def load_registry(path: str | Path = REGISTRY_PATH) -> dict:
    """Load and validate the registry.

    The explicit ``encoding="utf-8"`` matters: Python defaults to the platform
    encoding, which is cp1252 on Windows and would crash on ö.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"Modifier registry not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Modifier registry is not valid JSON: {exc}") from exc

    registry = {code: spec for code, spec in raw.items() if not code.startswith("_")}
    if not registry:
        raise RegistryError("Modifier registry contains no languages.")

    for code, spec in registry.items():
        if not isinstance(spec.get("alphabet"), str) or not spec["alphabet"]:
            raise RegistryError(f"[{code}] 'alphabet' must be a non-empty string.")
        if not isinstance(spec.get("stopwords"), list):
            raise RegistryError(f"[{code}] 'stopwords' must be a list.")
        categories = spec.get("categories")
        if not isinstance(categories, dict) or not categories:
            raise RegistryError(f"[{code}] 'categories' must be a non-empty object.")
        for name, cat in categories.items():
            if cat.get("pos") not in VALID_POSITIONS:
                raise RegistryError(
                    f"[{code}.{name}] 'pos' must be one of {VALID_POSITIONS}, "
                    f"got {cat.get('pos')!r}."
                )
            if not isinstance(cat.get("terms"), list) or not cat["terms"]:
                raise RegistryError(f"[{code}.{name}] 'terms' must be a non-empty list.")

    return registry


def available_languages() -> list[str]:
    return sorted(load_registry())


def category_names(language: str) -> list[str]:
    return list(_language(language)["categories"])


def alphabet(language: str) -> str:
    return _language(language)["alphabet"]


def stopwords(language: str) -> frozenset[str]:
    """Base stopwords unioned with this language's own modifier terms.

    Seeding with "comment" would otherwise put *comment* at the top of every
    French cluster label. c-TF-IDF already down-weights terms that appear
    everywhere; this is the belt to that pair of braces.
    """
    spec = _language(language)
    words = {fold_key(w) for w in spec["stopwords"]}
    for cat in spec["categories"].values():
        for term in cat["terms"]:
            words.update(fold_key(term).split())
    return frozenset(words)


def _language(language: str) -> dict:
    registry = load_registry()
    try:
        return registry[language]
    except KeyError:
        raise RegistryError(
            f"Unknown language {language!r}. Available: {', '.join(sorted(registry))}"
        ) from None


def build_seed_grid(
    query: str,
    language: str,
    *,
    categories: tuple[str, ...] | list[str] = (),
    letter_positions: tuple[str, ...] | list[str] = ("suffix",),
    include_digits: bool = False,
    include_unaccented: bool = False,
) -> list[Seed]:
    """Expand one query into the full grid of seeds to fetch.

    Position is honoured per category, so question words land before the seed
    ("wie größe") and never after it, which would be dead grammar in all six
    languages.
    """
    base = collapse_ws(nfc(query))
    if not base:
        return []

    variants = [base]
    if include_unaccented:
        plain = strip_accents(base)
        if fold_key(plain) != fold_key(base):
            variants.append(plain)

    spec = _language(language)
    seeds: list[Seed] = []

    for variant in variants:
        seeds.append(Seed(variant, "base", "none"))

        for position in letter_positions:
            for char in alphabet(language):
                seeds.append(Seed(_join(variant, char, position), "letters", position))

        if include_digits:
            for position in ("prefix", "suffix"):
                for digit in DIGITS:
                    seeds.append(Seed(_join(variant, digit, position), "digits", position))

        for name in categories:
            cat = spec["categories"].get(name)
            if cat is None:
                continue
            positions = ("prefix", "suffix") if cat["pos"] == "both" else (cat["pos"],)
            for term in cat["terms"]:
                for position in positions:
                    seeds.append(Seed(_join(variant, term, position), name, position))

    return list(dedupe(seeds, key=lambda s: fold_key(s.text)))


def _join(base: str, modifier: str, position: str) -> str:
    return f"{modifier} {base}" if position == "prefix" else f"{base} {modifier}"


def estimate_requests(query: str, language: str, **kwargs) -> int:
    """Seed count for the UI's pre-flight estimate."""
    return len(build_seed_grid(query or "x", language, **kwargs))
