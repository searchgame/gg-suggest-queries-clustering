"""Unicode-safe text handling.

The rule the rest of the app depends on: **normalize for keys, never for
display or for embeddings.** ``nfc()`` is the only transformation ever applied
to text that gets stored, shown or sent to the embedding model. Everything
else here produces a throwaway comparison key.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

# Letters in any script - so é, ö, ß and ñ all survive - with internal hyphens
# kept ("porte-clés" stays one token). Digits and underscore are excluded,
# which a bare \w would admit.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

# Every apostrophe variant, including the typographic one Google returns.
# Splitting here turns French elisions into two tokens ("l'eau" -> "l", "eau");
# the leftover "l" is then caught by the stopword list.
_APOSTROPHE_RE = re.compile(r"['’ʼ՚]")

_WS_RE = re.compile(r"\s+")

_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def nfc(text: str) -> str:
    """Canonically compose text on the way in.

    macOS hands out decomposed strings and Google returns both forms, so
    ``"é"`` (U+00E9) and ``"e" + U+0301`` arrive as different strings that
    render identically. Without this they survive deduplication as two rows.
    """
    return unicodedata.normalize("NFC", text)


def collapse_ws(text: str) -> str:
    """Collapse runs of whitespace and trim."""
    return _WS_RE.sub(" ", text).strip()


def fold_key(text: str) -> str:
    """Comparison key: NFC, casefolded, whitespace-collapsed.

    ``casefold()`` rather than ``lower()`` because it maps ß to ss, so
    ``"Straße"`` and ``"STRASSE"`` produce one key instead of two. German
    search queries mix both spellings constantly.

    The result is for grouping only - never display it.
    """
    return collapse_ws(nfc(text).casefold())


def tokenize(text: str) -> list[str]:
    """Split into casefolded word tokens, preserving every letter.

    Note that casefolding means ``"Größe"`` tokenizes to ``["grösse"]``: the ß
    is folded to ss rather than dropped. That is the intended unification. What
    matters is that no character is silently *deleted*, which is what the old
    ASCII-only ``[a-z]+`` pattern did (``"größe"`` became ``["gr", "e"]``).
    """
    stripped = _APOSTROPHE_RE.sub(" ", nfc(text))
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(stripped)]


def strip_accents(text: str) -> str:
    """Remove combining diacritics: ``"café"`` -> ``"cafe"``.

    Only used for the opt-in unaccented seed variant. It is deliberately not
    part of ``fold_key``: in French ``cote``, ``côté`` and ``coté`` are three
    different words, so accent-blind deduplication would destroy real results.

    ß is not a combining sequence and passes through unchanged.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without)


def dedupe(
    items: Iterable[T],
    key: Callable[[T], str] = None,  # type: ignore[assignment]
) -> Iterator[T]:
    """Yield items in order, skipping ones whose key was already seen."""
    key_fn: Callable[[T], str] = key or (lambda item: fold_key(str(item)))
    seen: set[str] = set()
    for item in items:
        k = key_fn(item)
        if k in seen:
            continue
        seen.add(k)
        yield item


def safe_filename(stem: str, ext: str, max_len: int = 80) -> str:
    """Build a download filename that survives a trip through a browser.

    Accented characters are kept - they are valid in filenames and the export
    is meant to be readable - but path separators and control characters are
    stripped and the stem is length-capped.
    """
    cleaned = collapse_ws(_UNSAFE_FILENAME_RE.sub(" ", nfc(stem)))
    cleaned = cleaned.replace(" ", "-").strip("-.") or "export"
    return f"{cleaned[:max_len]}.{ext.lstrip('.')}"
