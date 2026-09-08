"""Parallel Google Suggest scraping with provenance and failure accounting."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .modifiers import Seed
from .text import fold_key, nfc

ENDPOINT = "https://suggestqueries.google.com/complete/search"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# (connect, read). The old app set no timeout at all, so a single hung socket
# hung the whole page.
TIMEOUT = (3.05, 10.0)

_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>")


@dataclass
class SeedResult:
    seed: Seed
    suggestions: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class FetchReport:
    """Suggestions plus enough diagnostics to tell a block from a thin niche."""

    frame: pd.DataFrame
    n_seeds: int = 0
    n_failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.frame.empty

    @property
    def failure_rate(self) -> float:
        return self.n_failed / self.n_seeds if self.n_seeds else 0.0


def make_session(pool_size: int = 16) -> requests.Session:
    """Session with retry/backoff, so a 429 costs a pause rather than a gap."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=pool_size, pool_connections=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/xml, text/xml, */*"})
    return session


def parse_suggestions(content: bytes) -> list[str]:
    """Parse the toolbar XML payload.

    ElementTree accepts bytes and honours the document's own encoding
    declaration, which is why the raw ``response.content`` goes in here. The
    old code decoded with ``errors="ignore"`` first - that flag is what was
    silently deleting accented characters.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        # Malformed or mislabelled payload: decode leniently, drop the (now
        # inaccurate) declaration, and try once more.
        text = _XML_DECL_RE.sub("", content.decode("cp1252", errors="replace"), count=1)
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

    return [nfc(data) for el in root.iter("suggestion") if (data := el.get("data"))]


def fetch_seed(session: requests.Session, seed: Seed, country: str, language: str) -> SeedResult:
    params = {"output": "toolbar", "gl": country, "hl": language, "q": seed.text}
    try:
        response = session.get(ENDPOINT, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return SeedResult(seed, error=f"{type(exc).__name__}: {exc}")

    if response.status_code != 200:
        return SeedResult(seed, error=f"HTTP {response.status_code}")

    return SeedResult(seed, suggestions=parse_suggestions(response.content))


def fetch_suggestions(
    seeds: Sequence[Seed],
    country: str,
    language: str,
    *,
    max_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> FetchReport:
    """Fetch every seed in parallel and fold the results into one frame.

    A suggestion reachable from several seeds keeps *all* of its source
    categories, not just the first one - that provenance is what lets the UI
    facet clusters by intent later.
    """
    if not seeds:
        return FetchReport(frame=_empty_frame())

    workers = max(1, min(max_workers, len(seeds)))
    results: list[SeedResult] = []

    with make_session(pool_size=workers * 2) as session:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_seed, session, s, country, language) for s in seeds]
            for done, future in enumerate(futures, start=1):
                results.append(future.result())
                if progress is not None:
                    progress(done, len(futures))

    return _fold(results)


def _fold(results: Iterable[SeedResult]) -> FetchReport:
    rows: dict[str, dict] = {}
    n_seeds = 0
    errors: list[str] = []

    for result in results:
        n_seeds += 1
        if result.error:
            errors.append(f"{result.seed.text!r}: {result.error}")
            continue
        for suggestion in result.suggestions:
            key = fold_key(suggestion)
            if not key:
                continue
            row = rows.get(key)
            if row is None:
                # First sighting wins the surface form, accents and all.
                rows[key] = {
                    "query": suggestion,
                    "key": key,
                    "categories": {result.seed.category},
                    "n_seeds": 1,
                }
            else:
                row["categories"].add(result.seed.category)
                row["n_seeds"] += 1

    if not rows:
        return FetchReport(_empty_frame(), n_seeds=n_seeds, n_failed=len(errors), errors=errors[:10])

    frame = pd.DataFrame(
        [
            {
                "query": row["query"],
                "key": row["key"],
                "categories": sorted(row["categories"]),
                "n_seeds": row["n_seeds"],
            }
            for row in rows.values()
        ]
    )
    return FetchReport(frame, n_seeds=n_seeds, n_failed=len(errors), errors=errors[:10])


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({"query": [], "key": [], "categories": [], "n_seeds": []})
