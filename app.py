"""Google Suggest clustering - Streamlit front end.

This module owns the UI and the cache layer, and nothing else. All the work
lives in ``suggests/``, which never imports Streamlit.

The caching is the point of the layout below: four independently-keyed layers
mean that nudging a clustering slider recomputes only the last two stages, with
no network call and no re-embedding.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from suggests import cluster as clu
from suggests import config, export, fetch, labels as lbl, modifiers, ollama
from suggests.config import ClusterConfig, LabelConfig, VizConfig
from suggests.text import safe_filename

st.set_page_config(page_title="Google Suggest Queries Clustering Tool", page_icon="🫧", layout="wide")

# st.cache_data cannot hash a NumPy array on its own. Registering it once here
# is cleaner than threading digest strings through every cached signature.
ARRAY_HASH = {
    np.ndarray: lambda a: hashlib.blake2b(
        np.ascontiguousarray(a).tobytes(), digest_size=8
    ).digest()
}


# ── cache layer ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def probe_ollama(host: str) -> tuple[list[str], list[str], str | None]:
    """(embedding models, completion models, error)."""
    try:
        models = ollama.list_models(host)
    except ollama.OllamaError as exc:
        return [], [], str(exc)
    return (
        [m.name for m in models if m.can_embed],
        [m.name for m in models if m.can_generate],
        None,
    )


@st.cache_data(show_spinner=False)
def cached_fetch(
    query: str,
    country: str,
    language: str,
    categories: tuple[str, ...],
    letter_positions: tuple[str, ...],
    include_digits: bool,
    include_unaccented: bool,
    max_workers: int,
    _progress=None,
) -> fetch.FetchReport:
    seeds = modifiers.build_seed_grid(
        query,
        language,
        categories=categories,
        letter_positions=letter_positions,
        include_digits=include_digits,
        include_unaccented=include_unaccented,
    )
    return fetch.fetch_suggestions(
        seeds, country, language, max_workers=max_workers, progress=_progress
    )


@st.cache_data(show_spinner=False)
def cached_embed(texts: tuple[str, ...], model: str, host: str, _progress=None) -> np.ndarray:
    return ollama.embed(list(texts), model, host, progress=_progress)


@st.cache_data(show_spinner=False, hash_funcs=ARRAY_HASH)
def cached_cluster(embeddings: np.ndarray, cfg: ClusterConfig) -> clu.ClusterResult:
    return clu.cluster_embeddings(embeddings, cfg)


@st.cache_data(show_spinner=False, hash_funcs=ARRAY_HASH)
def cached_project(embeddings: np.ndarray, cfg: VizConfig) -> np.ndarray:
    return clu.project(embeddings, cfg)


@st.cache_data(show_spinner=False, hash_funcs=ARRAY_HASH)
def cached_ctfidf(
    queries: tuple[str, ...], cluster_labels: np.ndarray, language: str, top_k: int
) -> dict[int, str]:
    return lbl.ctfidf_labels(list(queries), cluster_labels, language, top_k=top_k)


@st.cache_data(show_spinner=False)
def cached_llm_labels(
    fingerprint: tuple[tuple[int, str, tuple[str, ...]], ...],
    language: str,
    model: str,
    host: str,
    temperature: float,
) -> dict[int, str]:
    """Label every cluster with the inference model, skipping what it fumbles."""
    out: dict[int, str] = {}
    for cid, terms, examples in fingerprint:
        label = lbl.llm_label(
            terms, list(examples), language, model, host, temperature=temperature
        )
        if label:
            out[cid] = label
    return out


# ── sidebar ────────────────────────────────────────────────────────────────

st.sidebar.image("assets/logo.png", width=288)

st.sidebar.header("Ollama")
host = st.sidebar.text_input("Host", value=config.DEFAULT_OLLAMA_HOST)
embed_models, gen_models, ollama_error = probe_ollama(host)

if ollama_error:
    st.sidebar.error(f"Not reachable. {ollama_error}")
else:
    st.sidebar.caption(
        f"{len(embed_models)} embedding · {len(gen_models)} inference model(s) available"
    )


def _model_picker(label: str, options: list[str], preferred: str, key: str) -> str:
    """Dropdown filtered by capability, with a free-text escape hatch.

    Capability detection is a heuristic on older Ollama builds, so nothing here
    prevents typing a model the probe failed to classify.
    """
    if not options:
        return st.sidebar.text_input(label, value=preferred, key=f"{key}_text")
    index = options.index(preferred) if preferred in options else 0
    choice = st.sidebar.selectbox(label, options + ["Other…"], index=index, key=key)
    if choice == "Other…":
        return st.sidebar.text_input(f"{label} (custom)", value=preferred, key=f"{key}_text")
    return choice


embedding_model = _model_picker(
    "Embedding Model", embed_models, "qwen3-embedding:0.6b", "embed_model"
)

st.sidebar.header("Sources")
country = st.sidebar.selectbox(
    "Country (gl)",
    list(config.COUNTRIES),
    format_func=lambda c: f"{c} — {config.COUNTRIES[c]}",
)
language = st.sidebar.selectbox(
    "Language (hl)",
    config.languages_for(country),
    format_func=lambda code: f"{code} — {config.LANGUAGES.get(code, code)}",
)

available_categories = modifiers.category_names(language)
selected_categories = tuple(
    name
    for name in available_categories
    if st.sidebar.checkbox(
        name.capitalize(), value=(name == "questions"), key=f"cat_{name}"
    )
)

letter_positions = tuple(
    position
    for position, default in (("suffix", True), ("prefix", False))
    if st.sidebar.checkbox(f"Letters ({position})", value=default, key=f"letters_{position}")
)
include_digits = st.sidebar.checkbox("Digits 0–9", value=False)
include_unaccented = st.sidebar.checkbox(
    "Also seed an unaccented variant",
    value=False,
    help="Google Suggest is largely diacritic-insensitive, so this roughly doubles "
    "the request count for a modest coverage gain.",
)
max_workers = st.sidebar.slider(
    "Parallel requests", 2, 12, 8, help="Lower this if Google starts throttling."
)

with st.sidebar.expander("Clustering"):
    cluster_cfg = ClusterConfig(
        min_cluster_size=st.slider("Minimum cluster size", 2, 25, 3),
        min_samples=st.slider("Minimum samples", 1, 10, 2),
        selection_method=st.radio("Selection method", ["eom", "leaf"], horizontal=True),
        selection_epsilon=st.slider("Selection epsilon", 0.0, 1.0, 0.0, 0.01),
        n_components=st.slider(
            "Cluster-space dimensions", 2, 30, 10,
            help="HDBSCAN runs here, not on the raw embeddings.",
        ),
        n_neighbors=st.slider("Cluster-space neighbours", 2, 50, 15),
        soft_assign=st.checkbox("Rescue outliers (soft assign)", value=False),
        soft_threshold=st.slider("Rescue threshold", 0.0, 1.0, 0.10, 0.01),
    )

with st.sidebar.expander("Visualization"):
    viz_cfg = VizConfig(
        n_neighbors=st.slider("Map neighbours", 2, 50, 15),
        min_dist=st.slider("Map minimum distance", 0.0, 0.99, 0.10),
    )

with st.sidebar.expander("Labelling"):
    use_llm = st.checkbox("Name clusters with an LLM", value=False)
    inference_model = ""
    if use_llm:
        inference_model = _model_picker(
            "Inference Model", gen_models, "llama3.2:latest", "infer_model"
        )
    label_cfg = LabelConfig(
        top_k=st.slider("Terms per label", 2, 8, 4),
        n_representatives=st.slider("Representatives per cluster", 1, 10, 3),
        use_llm=use_llm,
        inference_model=inference_model,
    )


# ── query ──────────────────────────────────────────────────────────────────

st.title("Google Suggest Clustering Tool")

st.markdown(
    "Expands a seed query across a language-aware seed grid, embeds the suggestions locally with Ollama, and groups them by simmilar topic. Made by [Victor Gras](https://victorgras.com/).")

query = st.text_input("Base search query", placeholder="e.g. recette crêpes")

n_seeds = (
    modifiers.estimate_requests(
        query,
        language,
        categories=selected_categories,
        letter_positions=letter_positions,
        include_digits=include_digits,
        include_unaccented=include_unaccented,
    )
    if query.strip()
    else 0
)
if n_seeds:
    st.caption(f"This run will make **{n_seeds}** requests to Google Suggest.")

if st.button("Run", type="primary", disabled=not query.strip()):
    st.session_state["run"] = {
        "query": query.strip(),
        "country": country,
        "language": language,
        "categories": selected_categories,
        "letter_positions": letter_positions,
        "include_digits": include_digits,
        "include_unaccented": include_unaccented,
        "max_workers": max_workers,
        "embedding_model": embedding_model,
        "host": host,
    }

run = st.session_state.get("run")
if not run:
    st.info("Enter a query and press Run to begin.")
    st.stop()


# ── pipeline ───────────────────────────────────────────────────────────────

timings: dict[str, float] = {}

started = time.perf_counter()
with st.spinner(f"Fetching {n_seeds or '…'} seeds from Google Suggest…"):
    report = cached_fetch(
        run["query"],
        run["country"],
        run["language"],
        run["categories"],
        run["letter_positions"],
        run["include_digits"],
        run["include_unaccented"],
        run["max_workers"],
    )
timings["fetch"] = time.perf_counter() - started

if not report.ok:
    st.error(
        f"No suggestions came back from {report.n_seeds} seeds "
        f"({report.n_failed} failed). Try another query, or check the country "
        f"and language pairing."
    )
    if report.errors:
        st.code("\n".join(report.errors))
    st.stop()

if report.n_failed:
    st.warning(
        f"{report.n_failed} of {report.n_seeds} seed requests failed "
        f"({report.failure_rate:.0%}). Results may be incomplete — lower the "
        f"parallel request count if this persists."
    )

frame = report.frame.copy()
st.success(
    f"{len(frame)} unique suggestions from {report.n_seeds} seeds "
    f"({run['language']} / {run['country']})."
)

started = time.perf_counter()
try:
    with st.spinner(f"Embedding {len(frame)} queries with {run['embedding_model']}…"):
        embeddings = cached_embed(
            tuple(frame["query"]), run["embedding_model"], run["host"]
        )
except ollama.OllamaError as exc:
    st.error(f"Embedding failed: {exc}")
    st.stop()
timings["embed"] = time.perf_counter() - started

started = time.perf_counter()
with st.spinner("Clustering…"):
    result = cached_cluster(embeddings, cluster_cfg)
timings["cluster"] = time.perf_counter() - started

if result.message:
    st.warning(result.message)

started = time.perf_counter()
with st.spinner("Projecting the map…"):
    coords = cached_project(embeddings, viz_cfg)
timings["project"] = time.perf_counter() - started

frame["cluster"] = result.labels
frame["cluster_probability"] = result.probabilities
frame["outlier_score"] = result.outlier_scores
frame["x"] = coords[:, 0]
frame["y"] = coords[:, 1]
frame["primary_category"] = frame["categories"].apply(
    lambda cats: next((c for c in cats if c != "base"), "base")
)

started = time.perf_counter()
term_labels = cached_ctfidf(
    tuple(frame["query"]), result.labels, run["language"], label_cfg.top_k
)
representatives = lbl.centroid_representatives(
    embeddings, result.labels, k=label_cfg.n_representatives
)

cluster_labels = dict(term_labels)
llm_used = False
if label_cfg.use_llm and label_cfg.inference_model and term_labels:
    fingerprint = tuple(
        (
            cid,
            term_labels.get(cid, ""),
            tuple(frame["query"].iloc[i] for i in representatives.get(cid, [])[:10]),
        )
        for cid in sorted(term_labels)
    )
    with st.spinner(f"Naming clusters with {label_cfg.inference_model}…"):
        llm_labels = cached_llm_labels(
            fingerprint,
            run["language"],
            label_cfg.inference_model,
            run["host"],
            label_cfg.temperature,
        )
    cluster_labels.update(llm_labels)
    llm_used = bool(llm_labels)
    if not llm_labels:
        st.info(
            f"{label_cfg.inference_model} returned no usable labels — "
            "showing key terms instead."
        )
timings["label"] = time.perf_counter() - started

frame["cluster_label"] = frame["cluster"].map(
    lambda cid: cluster_labels.get(int(cid), "") if cid >= 0 else "Outliers"
)
representative_rows = {i for rows in representatives.values() for i in rows}
frame["is_representative"] = [i in representative_rows for i in range(len(frame))]


# ── results ────────────────────────────────────────────────────────────────

tab_map, tab_clusters, tab_queries, tab_diag = st.tabs(
    ["Cluster map", "Clusters", "All queries", "Diagnostics"]
)

with tab_map:
    color_by = st.radio(
        "Colour by", ["Cluster", "Source category"], horizontal=True, key="color_by"
    )
    if color_by == "Cluster":
        frame["_colour"] = frame.apply(
            lambda r: f"{int(r['cluster'])} · {r['cluster_label']}"
            if r["cluster"] >= 0
            else "Outliers",
            axis=1,
        )
        order = sorted(
            (c for c in frame["_colour"].unique() if c != "Outliers"),
            key=lambda s: int(s.split(" · ")[0]),
        ) + (["Outliers"] if (frame["cluster"] < 0).any() else [])
    else:
        frame["_colour"] = frame["primary_category"]
        order = sorted(frame["_colour"].unique())

    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="_colour",
        hover_name="query",
        hover_data={"x": False, "y": False, "_colour": False, "cluster_probability": ":.2f"},
        category_orders={"_colour": order},
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=620,
        opacity=0.85,
    )
    sizes = (8 - frame["outlier_score"] * 4).clip(4, 8)
    figure.update_traces(marker=dict(size=sizes))
    figure.update_layout(
        legend_title=color_by,
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(figure, use_container_width=True)

    n_outliers = int((frame["cluster"] < 0).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries", len(frame))
    col2.metric("Clusters", result.n_clusters)
    col3.metric("Outliers", f"{n_outliers} ({n_outliers / len(frame):.0%})")
    col4.metric("DBCV", f"{result.dbcv:.3f}" if result.dbcv is not None else "n/a")

with tab_clusters:
    if result.n_clusters == 0:
        st.info("No clusters found. Try lowering the minimum cluster size.")
    else:
        sizes_by_cluster = frame[frame["cluster"] >= 0]["cluster"].value_counts()
        for cid in sizes_by_cluster.index:
            members = frame[frame["cluster"] == cid]
            label = cluster_labels.get(int(cid)) or "unlabelled"
            confidence = members["cluster_probability"].mean()
            with st.expander(
                f"Cluster {int(cid)} · {label} — {len(members)} queries, "
                f"{confidence:.2f} confidence"
            ):
                rep_indices = representatives.get(int(cid), [])
                if rep_indices:
                    st.markdown("**Most representative**")
                    for rank, row_index in enumerate(rep_indices, start=1):
                        st.markdown(f"{rank}. {frame['query'].iloc[row_index]}")
                    st.divider()
                if llm_used and int(cid) in term_labels:
                    st.caption(f"Key terms: {term_labels[int(cid)]}")
                st.markdown("**All queries**")
                for _, member in members.sort_values("query", key=lambda s: s.str.len()).iterrows():
                    marker = " ⭐" if member["is_representative"] else ""
                    st.write(f"- {member['query']}{marker}")

        outlier_frame = frame[frame["cluster"] < 0]
        if not outlier_frame.empty:
            with st.expander(f"Outliers — {len(outlier_frame)} queries"):
                for value in outlier_frame["query"]:
                    st.write(f"- {value}")

with tab_queries:
    display = frame[
        ["query", "cluster", "cluster_label", "primary_category", "cluster_probability", "n_seeds"]
    ].sort_values(["cluster", "cluster_probability"], ascending=[True, False])
    st.dataframe(display, use_container_width=True, hide_index=True)

    export_frame = export.build_export_frame(frame)
    stem = f"suggest-clusters-{run['query']}"
    st.download_button(
        "Download CSV",
        export.to_csv_bytes(export_frame),
        safe_filename(stem, "csv"),
        "text/csv",
    )
    xlsx = export.to_xlsx_bytes(export_frame)
    if xlsx is not None:
        st.download_button(
            "Download XLSX",
            xlsx,
            safe_filename(stem, "xlsx"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Install `openpyxl` to enable the XLSX export.")

with tab_diag:
    col1, col2, col3 = st.columns(3)
    col1.metric("Seeds fired", report.n_seeds)
    col2.metric("Failed", f"{report.n_failed} ({report.failure_rate:.0%})")
    col3.metric("Rescued outliers", result.n_reassigned)

    st.markdown("**Stage timings this render** — cached stages return instantly.")
    st.dataframe(
        pd.DataFrame(
            {"stage": list(timings), "seconds": [round(v, 3) for v in timings.values()]}
        ),
        use_container_width=True,
        hide_index=True,
    )

    if result.n_clusters:
        st.markdown("**Cluster sizes**")
        st.bar_chart(
            frame[frame["cluster"] >= 0]["cluster"].value_counts().sort_index(),
            use_container_width=True,
        )

    st.markdown("**Seed provenance**")
    st.dataframe(
        frame["primary_category"].value_counts().rename_axis("category").reset_index(name="queries"),
        use_container_width=True,
        hide_index=True,
    )

    if report.errors:
        st.markdown("**Sample failures**")
        st.code("\n".join(report.errors))
