"""Export writers.

CSV goes out as utf-8-sig. Plain utf-8 makes Excel on Windows render "café" as
"cafÃ©", which for a tool aimed at French and German markets is a daily
irritation rather than an edge case.
"""

from __future__ import annotations

import io

import pandas as pd

EXPORT_COLUMNS = [
    "query",
    "cluster",
    "cluster_label",
    "is_representative",
    "cluster_probability",
    "outlier_score",
    "source_categories",
    "n_seeds",
    "x",
    "y",
]


def build_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten the working frame into something a spreadsheet can hold."""
    out = frame.copy()

    if "categories" in out.columns:
        out["source_categories"] = out["categories"].apply(
            lambda cats: "|".join(cats) if isinstance(cats, (list, tuple, set)) else ""
        )

    for column in EXPORT_COLUMNS:
        if column not in out.columns:
            out[column] = ""

    out = out[EXPORT_COLUMNS].sort_values(
        ["cluster", "cluster_probability"], ascending=[True, False]
    )
    return out.round({"cluster_probability": 4, "outlier_score": 4, "x": 4, "y": 4})


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 with a BOM, so Excel reads the accents correctly."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(frame: pd.DataFrame) -> bytes | None:
    """XLSX if openpyxl is installed, otherwise None."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return None

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="clusters")
    return buffer.getvalue()
