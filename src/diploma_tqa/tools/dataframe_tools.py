import re
from difflib import SequenceMatcher

import pandas as pd


def normalize_text(text: str) -> str:

    # normalize text for easier column name and question comparison

    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"<[^<>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    # return similarity score (0, 1) between 2 strings
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def find_columns(df: pd.DataFrame, query: str, top_k: int = 5) -> str:
    # Find dataframe columns that are related to the query

    query_norm = normalize_text(query)
    query_tokens = set(query_norm.split())

    scored = []

    for column in df.columns:
        col_norm = normalize_text(column)
        col_tokens = set(col_norm.split())

        score = 0.0

        # Strong match: query appears inside column name
        if query_norm and query_norm in col_norm:
            score += 3.0
        # Strong match: column name appears inside query
        if col_norm and col_norm in query_norm:
            score += 2.0

        # add 1.5 points for shared words
        score += 1.5 * len(query_tokens & col_tokens)
        # add small score for general string similarity
        score += 0.5 * similarity(query_norm, col_norm)

        scored.append((score, column))

    # show best matching columns first
    scored.sort(key=lambda item: item[0], reverse=True)

    lines = [f"find_columns({query!r}) results:"]
    for score, column in scored[:top_k]:
        lines.append(f"- {column} (score={score:.2f})")

    return "\n".join(lines)


def profile_column(df: pd.DataFrame, column: str, max_values: int = 8) -> str:
    # Show useful information about one dataframe column

    if column not in df.columns:
        return f"profile_column({column!r}) error: column does not exist."

    series = df[column]
    non_null = series.dropna()

    lines = [
        f"profile_column({column!r}) result:",
        f"- dtype: {series.dtype}",
        f"- non-null: {series.notna().sum()}/{len(series)}",
        f"- unique: {series.nunique(dropna=True)}",
    ]
    # for numeric columns, compute simple summary stats
    if pd.api.types.is_numeric_dtype(series):
        lines.extend(
            [
                f"- min: {series.min()}",
                f"- max: {series.max()}",
                f"- mean: {series.mean()}",
            ]
        )
    # for non-numeric columns, find examples and frequent values
    else:
        sample_values = non_null.head(max_values).tolist()
        lines.append(f"- sample values: {sample_values}")

        try:
            top_values = non_null.value_counts().head(max_values)
            lines.append("- top values:")
            for value, count in top_values.items():
                lines.append(f"  - {repr(value)}: {count}")
        except Exception:
            pass

    return "\n".join(lines)