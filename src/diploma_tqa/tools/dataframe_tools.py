import re
from difflib import SequenceMatcher

import pandas as pd


ABSTRACT_VALUE_QUERY_TERMS = [
    "largest", "smallest", "highest", "lowest",
    "maximum", "minimum", "max", "min",
    "top", "first", "last",
    "most common", "least common", "most frequent", "least frequent",
    "average", "mean", "count", "total", "sum",
    "greater than", "less than", "more than", "fewer than",
    "above", "below",
]

def looks_like_abstract_value_query(query: str) -> bool:
    q = normalize_text(query)
    return any(term in q for term in ABSTRACT_VALUE_QUERY_TERMS)

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

def find_values(
    df: pd.DataFrame,
    column: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.65,
) -> str:
    # Search inside a dataframe column for values similar to the query.
    # Returns unique candidate values with scores and counts.

    if column not in df.columns:
        return f"find_values({column!r}, {query!r}) error: column does not exist."

    query_norm = normalize_text(query)

    if not query_norm:
        return f"find_values({column!r}, {query!r}) error: empty query."

    if looks_like_abstract_value_query(query):
        return (
            f"find_values(column={column!r}, query={query!r}) skipped: "
            "query looks like an operation, comparison, or aggregation, "
            "not a literal cell value."
        )

    series = df[column].dropna()

    query_tokens = set(query_norm.split())
    best_by_value = {}

    for idx, value in series.items():
        value_str = str(value)
        value_norm = normalize_text(value_str)

        if not value_norm:
            continue

        value_tokens = set(value_norm.split())
        score = 0.0

        # Strong signal: exact normalized match.
        if query_norm == value_norm:
            score += 5.0

        # Strong signal: substring match.
        if query_norm and query_norm in value_norm:
            score += 3.0

        if value_norm and value_norm in query_norm:
            score += 2.0

        # Medium signal: shared words.
        score += 1.5 * len(query_tokens & value_tokens)

        # Weak signal: fuzzy string similarity.
        score += similarity(query_norm, value_norm)

        if score >= min_score:
            rec = best_by_value.get(value_str)
            if rec is None:
                best_by_value[value_str] = {
                    "score": score,
                    "first_row": idx,
                    "count": 1,
                }
            else:
                rec["score"] = max(rec["score"], score)
                rec["count"] += 1

    scored = [
        (rec["score"], value_str, rec["first_row"], rec["count"])
        for value_str, rec in best_by_value.items()
    ]

    scored.sort(key=lambda item: (item[0], item[3]), reverse=True)

    lines = [f"find_values(column={column!r}, query={query!r}) results:"]

    if not scored:
        lines.append("- No high-confidence value matches found.")
        return "\n".join(lines)

    for score, value, first_row, count in scored[:top_k]:
        lines.append(
            f"- value {value!r} (score={score:.2f}, count={count}, first_row={first_row})"
        )

    return "\n".join(lines)