![Tool Logo](assets/logo-wide.png)

# Google Suggest Queries Clustering (with local Ollama models)

Expands a seed query across a language-aware seed grid, fetches Google Suggests, embeds the suggestions locally with Ollama, and groups them by similar subtopic using UMAP & HDBSCAN.

![Tool Overview Screenshot](assets/gg-suggests-tool-overview.png)

**Table of contents:**
- [Notes](#notes)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Adding a language](#adding-a-language)
- [App structure](#app-structure)

## Notes

> [!NOTE]
> * The original purpose of this tool is to experiment with relatively small, locally stored large language models (like <2GB footprint), on real consumer hardware (my machine is a Macbook 2020 with M1 chip & 16GB RAM, 250GB storage). You can, of course, get better results with larger models than the ones I give as examples below.

## Prerequisites

Have Ollama installed with at least one embedding model (for clustering), and optionally one inference model (for cluster naming). If you don't, install Ollama and then:

```bash
ollama pull [EMBEDDING_MODEL_NAME]   # embedding model for clustering, for example 'embeddinggemma:300m'
ollama pull [INFERENCE_MODEL_NAME]   # optional, inference model for LLM cluster naming, for example 'llama3.2'
```

See available models in the Ollama library: https://ollama.com/library

## Installation

In your project directory:

```bash
pip install -r requirements.txt

streamlit run app.py
```

## Usage

1. Pick a **country** (`GL`) and **language** (`hl`) in the sidebar. They are
   deliberately independent. For example, `CH` (Switzerland) users searches in `de` (german), `fr` (french) *and* `it` (italian).
2. Tick the seed categories you want. The caption under the query box shows how
   many requests the run will make before you commit to it.
3. Enter a query and press **Run**.

Everything below the Sources block (clustering, visualization, labelling) is
live. Changing it re-renders without re-fetching.

### The Diagnostics tab

Seeds fired, failure count and rate, DBCV score, rescued outliers, per-stage
timings, cluster size distribution, and the provenance breakdown showing which
modifier category each suggestion came from. This is what makes the tuning
sliders meaningful — DBCV is the number to optimize against.

## Adding a language

Everything lives in `data/modifiers.json`; no Python change is needed. Add a
key and the app picks it up:

```json
"pt": {
  "alphabet": "abcdefghijklmnopqrstuvwxyzáéíóúâê",
  "stopwords": ["o", "a", "de", "do", "da", "em", "para", "..."],
  "categories": {
    "questions":    {"pos": "prefix", "terms": ["como", "por que", "quando", "onde"]},
    "prepositions": {"pos": "both",   "terms": ["para", "com", "sem", "perto de"]},
    "comparatives": {"pos": "suffix", "terms": ["ou", "vs", "melhor que"]},
    "commercial":   {"pos": "suffix", "terms": ["preço", "barato", "comprar"]}
  }
}
```

Three rules the registry enforces:

- **`alphabet` holds word-*initial* letters only.** A letter seed means "the
  next word starts with this", so German gets ä/ö/ü but never ß, which begins
  no German word.
- **`pos` is grammar, not preference.** Question words go before the seed in
  every language here; `{query} wie` is dead grammar and wastes a request.
- **Don't repeat modifier terms in `stopwords`.** They are unioned in
  automatically, so seeding with *comment* can't put "comment" atop every
  French cluster label.

Then add the code to `LANGUAGES` in `suggests/config.py` so it appears in the
dropdown, and to `LANGUAGE_NAMES` in `suggests/labels.py` so the LLM naming
prompt can name the language.

The file must stay UTF-8. It is read with an explicit `encoding="utf-8"`, since
Python otherwise defaults to the platform encoding and would fail on `ö`.

## App structure

```
app.py              Streamlit UI + the cache layer, and nothing else
data/modifiers.json per-language seed and stopword registry
suggests/
  config.py         dataclasses, country/language tables and presets
  text.py           NFC, casefold keys, the Unicode tokenizer
  modifiers.py      registry loader, build_seed_grid()
  fetch.py          parallel scraping, provenance, failure accounting
  ollama.py         one client: /api/tags, /api/embed, /api/generate
  cluster.py        UMAP → HDBSCAN, soft-assign, DBCV
  labels.py         c-TF-IDF, centroids, optional LLM naming
  export.py         utf-8-sig CSV / XLSX
```

Nothing under `suggests/` imports Streamlit, so the pipeline is driveable from
a script or a test.

### The one rule to preserve

**Normalize for keys, never for display or for embeddings.** `nfc()` is the
only transformation applied to stored text; `fold_key()` and `tokenize()`
produce throwaway comparison keys. The `query` column keeps its original
accented surface form, and that form is what reaches the embedding model —
accents carry meaning, and stripping them degrades the vectors.

Accent-stripping is available (`strip_accents`) but is opt-in and used only for
the alternative seed variant. It is deliberately *not* part of `fold_key`: in
French `cote`, `côté` and `coté` are three different words.
