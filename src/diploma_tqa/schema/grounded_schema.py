from __future__ import annotations

"""Deterministic, value-aware schema evidence for table question answering.

This module deliberately does *not* predict an operation, aggregation, or logical
plan.  It only exposes facts that can be checked against the dataframe:

* exact column names and physical/annotated types;
* compact per-column profiles;
* lexical evidence between the question and column names; and
* entity/value links between question spans and real cell values.

The design is conservative because a small code model is easily anchored by an
incorrect intermediate plan.  Weak evidence is omitted instead of being shown as
"likely" truth.
"""

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd

from diploma_tqa.schema.schema_linker import (
    extract_type_annotation,
    strip_type_annotation,
)


# Words that should not make a column/value look relevant by themselves.
_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "in", "into", "is", "it", "its", "list", "of", "on", "or", "our",
    "please", "return", "show", "than", "that", "the", "their", "there",
    "these", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "with", "would", "you",
}

_OPERATION_WORDS = {
    "all", "average", "bottom", "combined", "count", "different", "distinct",
    "fewest", "first", "greatest", "highest", "largest", "last", "least",
    "longest", "lowest", "maximum", "mean", "median", "minimum", "most",
    "number", "oldest", "smallest", "sum", "top", "total", "unique",
    "youngest",
}

_TITLE_WORDS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "lady"}

_NUMERIC_ANSWER_TYPES = {"number", "list[number]"}
_CATEGORY_ANSWER_TYPES = {"category", "list[category]"}

_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:%|\b)")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class ValueLink:
    column: str
    question_span: str
    value: str
    score: float
    match_type: str


@dataclass(frozen=True)
class RankedColumn:
    column: str
    name_score: float
    value_score: float
    type_bonus: float
    total_score: float
    reasons: tuple[str, ...]


@dataclass
class _ColumnMeta:
    column: str
    base_name: str
    annotation: str | None
    dtype: str
    kind: str
    non_null: int
    unique: int
    null_fraction: float


@dataclass
class _ColumnProfile:
    meta: _ColumnMeta
    sample_values: list[Any]
    top_values: list[dict[str, Any]]
    numeric_stats: dict[str, Any] | None


@dataclass
class _ColumnValueIndex:
    values: list[str]
    exact: dict[str, list[str]]
    token_to_values: dict[str, list[str]]
    numeric_exact: dict[str, list[str]]


def _ascii_fold(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    return value.encode("ascii", "ignore").decode("ascii")


def normalize_text(text: Any) -> str:
    """Normalize text while splitting snake_case and camelCase identifiers."""

    value = _ascii_fold(text)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = value.replace("_", " ")
    value = re.sub(r"<[^<>]+>", " ", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _stem_token(token: str) -> str:
    """A deliberately small normalizer for plural/inflection mismatches."""

    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


def content_tokens(text: Any, *, drop_operations: bool = False) -> list[str]:
    tokens: list[str] = []
    for token in normalize_text(text).split():
        token = _stem_token(token)
        if len(token) < 2 or token in _STOPWORDS:
            continue
        if drop_operations and token in _OPERATION_WORDS:
            continue
        tokens.append(token)
    return tokens


def _token_f1(left: Iterable[str], right: Iterable[str]) -> tuple[float, float, float]:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0, 0.0, 0.0
    overlap = len(left_set & right_set)
    precision = overlap / len(right_set)
    recall = overlap / len(left_set)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _safe_scalar(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _column_kind(column: str, series: pd.Series) -> str:
    annotation = (extract_type_annotation(column) or "").lower()

    if "list[number" in annotation:
        return "list[number]"
    if "list[category" in annotation:
        return "list[category]"
    if "number" in annotation:
        return "number"
    if "boolean" in annotation or "bool" in annotation:
        return "boolean"
    if "date" in annotation or "time" in annotation:
        return "date"
    if "category" in annotation:
        return "category"
    if "text" in annotation:
        return "text"
    if "url" in annotation:
        return "url"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "text"


def score_column_name(question: str, column: str) -> tuple[float, list[str]]:
    """Score a question-column pair using only inspectable lexical evidence."""

    q_norm = normalize_text(question)
    base = strip_type_annotation(column)
    c_norm = normalize_text(base)
    q_tokens = content_tokens(question)
    c_tokens = content_tokens(base)

    score = 0.0
    reasons: list[str] = []

    if c_norm and re.search(rf"\b{re.escape(c_norm)}\b", q_norm):
        score += 8.0
        reasons.append("exact column phrase in question")

    precision, recall, f1 = _token_f1(c_tokens, q_tokens)
    score += 4.0 * f1 + 2.0 * recall

    if c_tokens and set(c_tokens).issubset(set(q_tokens)):
        score += 2.0
        reasons.append("all column tokens occur in question")
    elif f1 >= 0.66:
        reasons.append("strong token overlap")
    elif f1 > 0:
        reasons.append("partial token overlap")

    # Handle review/reviewer and similar local morphology without embeddings.
    partial_hits = 0
    for c_token in set(c_tokens):
        for q_token in set(q_tokens):
            if min(len(c_token), len(q_token)) >= 4 and (
                c_token in q_token or q_token in c_token
            ):
                partial_hits += 1
                break
    if partial_hits:
        score += min(1.5, 0.75 * partial_hits)
        reasons.append("partial token match")

    if c_norm and q_norm:
        score += 0.75 * SequenceMatcher(None, c_norm, q_norm).ratio()

    return score, reasons


def _question_spans(question: str) -> list[tuple[str, str]]:
    """Return conservative candidate entity/literal spans from a question."""

    spans: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(raw: str, source: str) -> None:
        raw = raw.strip(" \t\n\r.,;:!?()[]{}")
        norm = normalize_text(raw)
        if not norm or norm in seen:
            return
        tokens = content_tokens(raw, drop_operations=True)
        if not tokens and not _NUMBER_RE.fullmatch(raw.strip()):
            return
        if all(token in _OPERATION_WORDS for token in tokens):
            return
        seen.add(norm)
        spans.append((raw, source))

    for match in _QUOTED_RE.finditer(question):
        add(match.group(1) or match.group(2), "quoted")

    for match in _NUMBER_RE.finditer(question):
        # A number following top/bottom/first/etc. is usually an output-list size,
        # not a cell literal.  Linking it to every numeric column containing that
        # value creates extremely persuasive but false schema evidence.
        prefix_tokens = normalize_text(question[:match.start()]).split()[-3:]
        if any(
            token in {
                "top", "bottom", "first", "last", "list", "name",
                "highest", "lowest", "largest", "smallest",
                "youngest", "oldest",
            }
            for token in prefix_tokens
        ):
            continue
        add(match.group(0), "numeric")

    # Capitalized multi-token expressions often contain named entities.  Skip the
    # first question token because it is capitalized merely by sentence position.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.'-]*", question)
    current: list[str] = []
    for index, word in enumerate(words):
        is_capitalized = index > 0 and word[:1].isupper() and not word.isupper()
        is_acronym = len(word) >= 2 and word.isupper()
        if is_capitalized or is_acronym:
            current.append(word)
        else:
            if current:
                add(" ".join(current), "capitalized")
                current = []
    if current:
        add(" ".join(current), "capitalized")

    # Contiguous n-grams provide recall for lowercase categories such as
    # "research and development".  Only keep n-grams containing a content word.
    raw_tokens = _WORD_RE.findall(question)
    max_n = min(5, len(raw_tokens))
    for n in range(max_n, 0, -1):
        for start in range(0, len(raw_tokens) - n + 1):
            span = " ".join(raw_tokens[start:start + n])
            tokens = content_tokens(span, drop_operations=True)
            if not tokens:
                continue
            # Single-token spans must be informative enough to avoid matching
            # values such as "a", "in", or an operation word.
            if n == 1 and (len(tokens[0]) < 4 or tokens[0] in _OPERATION_WORDS):
                continue
            add(span, "ngram")

    # Prefer explicit and longer spans during matching.
    source_priority = {"quoted": 0, "numeric": 1, "capitalized": 2, "ngram": 3}
    spans.sort(key=lambda item: (source_priority[item[1]], -len(normalize_text(item[0]))))
    return spans[:80]


def _value_match_score(span: str, value: str, source: str) -> tuple[float, str] | None:
    s_norm = normalize_text(span)
    v_norm = normalize_text(value)
    if not s_norm or not v_norm:
        return None

    s_tokens = [t for t in content_tokens(span, drop_operations=True) if t not in _TITLE_WORDS]
    v_tokens = [t for t in content_tokens(value, drop_operations=False) if t not in _TITLE_WORDS]

    if s_norm == v_norm:
        return (12.0 if source == "quoted" else 11.0), "exact normalized value"

    if len(v_norm) >= 3 and re.search(rf"\b{re.escape(v_norm)}\b", s_norm):
        return 10.0, "table value occurs in question span"

    if len(s_norm) >= 3 and re.search(rf"\b{re.escape(s_norm)}\b", v_norm):
        return 9.0, "question span occurs in table value"

    if not s_tokens or not v_tokens:
        return None

    shared = set(s_tokens) & set(v_tokens)
    if not shared:
        return None

    # Coverage of the shorter side is useful for aliases such as
    # "Mr Harari" -> "Yuval Noah Harari" after title-word removal.
    short_coverage = len(shared) / min(len(set(s_tokens)), len(set(v_tokens)))
    _, _, f1 = _token_f1(s_tokens, v_tokens)

    if short_coverage == 1.0 and any(len(token) >= 4 for token in shared):
        score = 7.0 + min(1.5, 0.5 * len(shared))
        if source == "quoted":
            score += 0.5
        return score, "token-contained entity match"

    if f1 >= 0.66 and any(len(token) >= 4 for token in shared):
        score = 5.0 + 2.0 * f1
        return score, "strong entity token overlap"

    # Fuzzy matching is used only after a real token overlap, preventing the low
    # thresholds in the original tool from returning almost arbitrary values.
    ratio = SequenceMatcher(None, s_norm, v_norm).ratio()
    if ratio >= 0.82 and max(len(s_norm), len(v_norm)) >= 5:
        return 4.5 + ratio, "fuzzy value match"

    return None


def _numeric_key(text: Any) -> str | None:
    raw = str(text).strip().replace("\u00a0", " ")
    raw = re.sub(r"[^0-9,.$%+\-]", "", raw).replace("$", "")
    if not raw:
        return None
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%")
    if raw.count(",") == 1 and "." not in raw:
        left, right = raw.split(",", 1)
        raw = f"{left}.{right}" if len(right) <= 3 else left + right
    else:
        raw = raw.replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    if is_percent:
        # Keep the literal percentage representation.  We do not assume whether a
        # table stores 25% as 25 or 0.25; exact text still remains available.
        return f"percent:{number:.12g}"
    return f"number:{number:.12g}"


class GroundedSchemaIndex:
    """Per-dataframe cache used by all questions for a dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_index_values: int = 12000,
        max_value_length: int = 120,
    ) -> None:
        self.df = df
        self.max_index_values = max(100, int(max_index_values))
        self.max_value_length = max(20, int(max_value_length))
        self._meta: dict[str, _ColumnMeta] = {}
        self._profiles: dict[str, _ColumnProfile] = {}
        self._value_indexes: dict[str, _ColumnValueIndex] = {}

        for column in df.columns:
            series = df[column]
            non_null = int(series.notna().sum())
            try:
                unique = int(series.nunique(dropna=True))
            except Exception:
                unique = -1
            self._meta[column] = _ColumnMeta(
                column=column,
                base_name=strip_type_annotation(column),
                annotation=extract_type_annotation(column),
                dtype=str(series.dtype),
                kind=_column_kind(column, series),
                non_null=non_null,
                unique=unique,
                null_fraction=(1.0 - non_null / len(series)) if len(series) else 0.0,
            )

    def _profile(self, column: str) -> _ColumnProfile:
        cached = self._profiles.get(column)
        if cached is not None:
            return cached

        series = self.df[column]
        non_null = series.dropna()
        meta = self._meta[column]

        sample_values: list[Any] = []
        seen: set[str] = set()
        for value in non_null.head(40).tolist():
            safe = _safe_scalar(value)
            key = repr(safe)
            if key in seen:
                continue
            seen.add(key)
            sample_values.append(safe)
            if len(sample_values) >= 4:
                break

        top_values: list[dict[str, Any]] = []
        if meta.kind not in {"number", "date"} and meta.unique != 0:
            try:
                counts = non_null.value_counts(dropna=True).head(4)
                # On high-cardinality identifiers/names, value_counts() often
                # returns four arbitrary singleton values. Calling those
                # "frequent" is misleading and adds prompt noise; fall back to
                # neutral samples unless at least one value actually repeats.
                if not counts.empty and int(counts.iloc[0]) > 1:
                    for value, count in counts.items():
                        top_values.append({
                            "value": _safe_scalar(value),
                            "count": int(count),
                        })
            except Exception:
                pass

        numeric_stats: dict[str, Any] | None = None
        if meta.kind == "number":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                numeric_stats = {
                    "min": _safe_scalar(numeric.min()),
                    "max": _safe_scalar(numeric.max()),
                    "median": _safe_scalar(numeric.median()),
                }

        profile = _ColumnProfile(
            meta=meta,
            sample_values=sample_values,
            top_values=top_values,
            numeric_stats=numeric_stats,
        )
        self._profiles[column] = profile
        return profile

    def _index_values(self, column: str) -> _ColumnValueIndex:
        cached = self._value_indexes.get(column)
        if cached is not None:
            return cached

        meta = self._meta[column]
        empty = _ColumnValueIndex(values=[], exact={}, token_to_values={}, numeric_exact={})
        if meta.kind in {"list[number]", "list[category]", "url"}:
            self._value_indexes[column] = empty
            return empty

        series = self.df[column].dropna()
        if series.empty:
            self._value_indexes[column] = empty
            return empty

        # Numeric literals are cheap to link, but indexing every continuous value
        # is not useful.  Keep unique numeric values only for low-cardinality data.
        if meta.kind == "number" and meta.unique > 500:
            self._value_indexes[column] = empty
            return empty

        try:
            unique_values = pd.unique(series.map(str))
        except Exception:
            unique_values = pd.unique(series.astype(str))

        values = [
            str(value)
            for value in unique_values
            if 0 < len(str(value)) <= self.max_value_length
        ]

        if len(values) > self.max_index_values:
            # Preserve frequent values and a deterministic sample spread across
            # the complete column. Taking only the first values systematically
            # missed rare entities that occurred late in a table.
            half = self.max_index_values // 2
            sample_count = max(1, half)
            if sample_count == 1:
                spread_values = [values[0]]
            else:
                step = (len(values) - 1) / (sample_count - 1)
                spread_values = [
                    values[min(len(values) - 1, round(position * step))]
                    for position in range(sample_count)
                ]
            try:
                frequent = [
                    str(value)
                    for value in series.astype(str).value_counts().head(
                        self.max_index_values - half
                    ).index.tolist()
                ]
            except Exception:
                frequent = []
            values = list(dict.fromkeys(frequent + spread_values))[:self.max_index_values]

        exact: dict[str, list[str]] = {}
        token_to_values: dict[str, list[str]] = {}
        numeric_exact: dict[str, list[str]] = {}
        for value in values:
            norm = normalize_text(value)
            if norm:
                exact.setdefault(norm, []).append(value)

            numeric = _numeric_key(value)
            if numeric is not None:
                numeric_exact.setdefault(numeric, []).append(value)

            # Token lookup makes entity linking sublinear in column cardinality.
            # Cap postings for extremely common tokens; exact lookup still works.
            tokens = {
                token
                for token in content_tokens(value)
                if len(token) >= 4 and token not in _TITLE_WORDS
            }
            for token in list(tokens)[:12]:
                posting = token_to_values.setdefault(token, [])
                if len(posting) < 250:
                    posting.append(value)

        index = _ColumnValueIndex(
            values=values,
            exact=exact,
            token_to_values=token_to_values,
            numeric_exact=numeric_exact,
        )
        self._value_indexes[column] = index
        return index

    @staticmethod
    def _candidate_values_for_span(
        index: _ColumnValueIndex,
        span: str,
        source: str,
        *,
        max_candidates: int = 120,
    ) -> list[str]:
        if not index.values:
            return []

        norm = normalize_text(span)
        candidates: list[str] = list(index.exact.get(norm, []))

        if source == "numeric":
            numeric = _numeric_key(span)
            if numeric is not None:
                candidates.extend(index.numeric_exact.get(numeric, []))
            return list(dict.fromkeys(candidates))[:max_candidates]

        span_tokens = [
            token
            for token in content_tokens(span, drop_operations=True)
            if len(token) >= 4 and token not in _TITLE_WORDS
        ]
        unique_tokens = list(dict.fromkeys(span_tokens))
        postings = [
            index.token_to_values[token]
            for token in unique_tokens
            if token in index.token_to_values
        ]
        if postings and len(postings) == len(unique_tokens):
            if len(unique_tokens) == 1:
                # A lone common token such as "customer" or "united" is not an
                # entity link.  Exact normalized lookup above remains available.
                if len(postings[0]) <= 20:
                    candidates.extend(postings[0])
            else:
                # Multi-token aliases must match every informative token.  Union
                # matching made broad question n-grams link to arbitrary values.
                common = set(postings[0])
                for posting in postings[1:]:
                    common.intersection_update(posting)
                candidates.extend(
                    sorted(common, key=lambda value: abs(len(value) - len(span)))
                )

        return list(dict.fromkeys(candidates))[:max_candidates]

    def _value_search_columns(self, ranked_by_name: list[RankedColumn]) -> list[str]:
        total_columns = len(self.df.columns)
        if total_columns <= 40:
            return [
                column
                for column, meta in self._meta.items()
                if meta.kind in {"category", "text", "boolean", "date", "number"}
            ]

        # Very wide tables are where unconstrained value scans become expensive.
        # Search top lexical columns plus low-cardinality categorical columns.
        selected = [item.column for item in ranked_by_name[:20]]
        for column, meta in self._meta.items():
            if meta.kind in {"category", "boolean", "date"} and 0 <= meta.unique <= 200:
                selected.append(column)
        return list(dict.fromkeys(selected))

    def _find_value_links(
        self,
        question: str,
        ranked_by_name: list[RankedColumn],
        *,
        max_links: int = 8,
    ) -> list[ValueLink]:
        spans = _question_spans(question)
        if not spans:
            return []

        links: list[ValueLink] = []
        name_scores = {item.column: item.name_score for item in ranked_by_name}
        for column in self._value_search_columns(ranked_by_name):
            best_for_column: dict[str, ValueLink] = {}
            index = self._index_values(column)
            if not index.values:
                continue

            for span, source in spans:
                # Bare numeric literals are highly non-unique.  Require the
                # question to also name the column before treating a number as
                # schema evidence.  Categorical entity links do not need this.
                if source == "numeric" and name_scores.get(column, 0.0) < 1.5:
                    continue

                for value in self._candidate_values_for_span(index, span, source):
                    scored = _value_match_score(span, value, source)
                    if scored is None:
                        continue
                    score, match_type = scored
                    if score < 6.25:
                        continue
                    candidate = ValueLink(
                        column=column,
                        question_span=span,
                        value=value,
                        score=score,
                        match_type=match_type,
                    )
                    key = normalize_text(value)
                    previous = best_for_column.get(key)
                    if previous is None or candidate.score > previous.score:
                        best_for_column[key] = candidate

            links.extend(
                sorted(best_for_column.values(), key=lambda item: item.score, reverse=True)[:3]
            )

        links.sort(key=lambda item: (item.score, len(item.value)), reverse=True)

        # Remove exact duplicate links while retaining alternative columns for
        # genuine ambiguity.
        deduped: list[ValueLink] = []
        seen: set[tuple[str, str, str]] = set()
        for link in links:
            key = (
                normalize_text(link.column),
                normalize_text(link.question_span),
                normalize_text(link.value),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(link)
            if len(deduped) >= max_links:
                break
        return deduped

    def rank_columns(
        self,
        question: str,
        answer_type: str,
    ) -> tuple[list[RankedColumn], list[ValueLink]]:
        preliminary: list[RankedColumn] = []
        answer_type = str(answer_type or "unknown")

        for column in self.df.columns:
            name_score, reasons = score_column_name(question, column)
            kind = self._meta[column].kind
            # The official answer type constrains the returned Python value, but
            # it does not reliably identify the computation column. For example,
            # a numeric answer may count rows in a categorical column, and a
            # category answer may come from a numeric-coded ID. Keep the field for
            # diagnostics but do not inject an unverified type prior here.
            type_bonus = 0.0

            preliminary.append(RankedColumn(
                column=column,
                name_score=name_score,
                value_score=0.0,
                type_bonus=type_bonus,
                total_score=name_score + type_bonus,
                reasons=tuple(reasons),
            ))

        preliminary.sort(key=lambda item: (item.total_score, item.column), reverse=True)
        value_links = self._find_value_links(question, preliminary)
        best_value_score: dict[str, float] = {}
        for link in value_links:
            best_value_score[link.column] = max(
                best_value_score.get(link.column, 0.0),
                link.score,
            )

        ranked: list[RankedColumn] = []
        for item in preliminary:
            value_score = best_value_score.get(item.column, 0.0)
            reasons = list(item.reasons)
            if value_score:
                reasons.append("question span linked to real table value")
            # A verified value link should dominate a weak name match, but it must
            # not become a guessed operation or role.
            total = item.name_score + item.type_bonus + 0.75 * value_score
            ranked.append(RankedColumn(
                column=item.column,
                name_score=item.name_score,
                value_score=value_score,
                type_bonus=item.type_bonus,
                total_score=total,
                reasons=tuple(reasons),
            ))

        ranked.sort(key=lambda item: (item.total_score, item.column), reverse=True)
        return ranked, value_links

    def build_evidence(
        self,
        question: str,
        answer_type: str,
        *,
        max_columns: int = 6,
    ) -> dict[str, Any]:
        ranked, value_links = self.rank_columns(question, answer_type)
        max_columns = max(1, int(max_columns))

        selected: list[str] = []
        # Never drop a column with strong, table-verified entity evidence.
        for link in value_links:
            if link.score >= 7.0 and link.column not in selected:
                selected.append(link.column)

        # Do not pad the evidence block with arbitrary top-k columns.  Only show
        # columns that have an inspectable lexical signal or a verified value link;
        # the complete schema is already present in the base prompt.
        for item in ranked:
            has_lexical_evidence = (
                item.name_score >= 1.5
                or "exact column phrase in question" in item.reasons
                or "strong token overlap" in item.reasons
                or "all column tokens occur in question" in item.reasons
            )
            has_value_evidence = item.value_score >= 6.25
            if (has_lexical_evidence or has_value_evidence) and item.column not in selected:
                selected.append(item.column)
            if len(selected) >= max_columns:
                break
        selected = selected[:max_columns]

        top_score = ranked[0].total_score if ranked else 0.0
        second_score = ranked[1].total_score if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        has_strong_value_link = any(link.score >= 7.0 for link in value_links)
        has_exact_name = bool(ranked and "exact column phrase in question" in ranked[0].reasons)

        # Conservative activation: for narrow tables the base prompt already shows
        # all columns, so weak extra hints add noise without improving visibility.
        active = bool(
            has_strong_value_link
            or has_exact_name
            or (
                len(self.df.columns) > 8
                and top_score >= 2.25
                and (margin >= 0.25 or top_score >= 4.0)
            )
        )

        if has_strong_value_link or has_exact_name or top_score >= 5.5:
            confidence = "high"
        elif active:
            confidence = "medium"
        else:
            confidence = "low"

        ranked_map = {item.column: item for item in ranked}
        column_evidence: list[dict[str, Any]] = []
        for column in selected:
            profile = self._profile(column)
            rank = ranked_map[column]
            column_evidence.append({
                "exact_name": column,
                "base_name": profile.meta.base_name,
                "annotation": profile.meta.annotation,
                "dtype": profile.meta.dtype,
                "kind": profile.meta.kind,
                "non_null": profile.meta.non_null,
                "unique": profile.meta.unique,
                "null_fraction": round(profile.meta.null_fraction, 4),
                "sample_values": profile.sample_values,
                "top_values": profile.top_values,
                "numeric_stats": profile.numeric_stats,
                "name_score": round(rank.name_score, 3),
                "value_score": round(rank.value_score, 3),
                "combined_score": round(rank.total_score, 3),
                "evidence_reasons": list(rank.reasons),
            })

        return {
            "active": active,
            "confidence": confidence,
            "answer_type": answer_type,
            "num_table_columns": len(self.df.columns),
            "selected_columns": selected,
            "columns": column_evidence,
            "value_links": [asdict(link) for link in value_links],
            "top_score": round(top_score, 3),
            "score_margin": round(margin, 3),
            "policy": (
                "facts_only_no_operation_prediction; weak evidence is omitted; "
                "full dataframe schema remains authoritative"
            ),
        }


def _compact_value(value: Any, limit: int = 70) -> str:
    text = repr(_safe_scalar(value))
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def format_grounded_schema_evidence(evidence: dict[str, Any]) -> str:
    """Format evidence compactly for the code-generation prompt."""

    if not evidence or not evidence.get("active"):
        return ""

    lines = [
        "Grounded schema evidence (verified table facts, not a reasoning plan):",
        "- The original question and expected answer type remain authoritative.",
        "- Do not infer an operation, aggregation, or filter solely from this section.",
        "- Use exact dataframe column names and exact linked values when relevant.",
        "Selected columns:",
    ]

    for item in evidence.get("columns", []):
        exact_name = item["exact_name"]
        summary = (
            f'- {exact_name!r}: kind={item["kind"]}, dtype={item["dtype"]}, '
            f'non_null={item["non_null"]}, unique={item["unique"]}'
        )
        lines.append(summary)

        top_values = item.get("top_values") or []
        sample_values = item.get("sample_values") or []
        numeric_stats = item.get("numeric_stats")

        if top_values:
            rendered = ", ".join(
                f'{_compact_value(entry["value"])} (n={entry["count"]})'
                for entry in top_values[:4]
            )
            lines.append(f"  frequent values: {rendered}")
        elif sample_values:
            rendered = ", ".join(_compact_value(value) for value in sample_values[:4])
            lines.append(f"  sample values: {rendered}")

        if numeric_stats:
            lines.append(
                "  global numeric profile (not necessarily the answer): "
                f"min={_compact_value(numeric_stats.get('min'))}, "
                f"max={_compact_value(numeric_stats.get('max'))}, "
                f"median={_compact_value(numeric_stats.get('median'))}"
            )

    links = evidence.get("value_links") or []
    if links:
        lines.append("Question-to-value links found in the dataframe:")
        for link in links[:8]:
            lines.append(
                f'- question span {link["question_span"]!r} -> '
                f'column {link["column"]!r}, exact table value {link["value"]!r} '
                f'({link["match_type"]})'
            )

    lines.append(
        "If this evidence conflicts with the full schema or the question, ignore it; "
        "do not substitute a merely similar column or value."
    )
    return "\n".join(lines)


def build_grounded_schema_evidence(
    question: str,
    answer_type: str,
    df: pd.DataFrame,
    *,
    max_columns: int = 6,
    max_index_values: int = 12000,
) -> dict[str, Any]:
    """Convenience function for one-off use.

    For benchmark runs, instantiate :class:`GroundedSchemaIndex` once per dataset
    and reuse it for all questions to avoid repeated profiling/indexing.
    """

    index = GroundedSchemaIndex(df, max_index_values=max_index_values)
    return index.build_evidence(
        question=question,
        answer_type=answer_type,
        max_columns=max_columns,
    )
