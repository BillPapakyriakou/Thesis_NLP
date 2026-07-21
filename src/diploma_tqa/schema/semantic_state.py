from __future__ import annotations

import json
from typing import Any

import pandas as pd

from diploma_tqa.schema.schema_linker import (
    extract_type_annotation,
    find_relevant_columns,
    strip_type_annotation,
)


ALLOWED_INTENTS = {
    "lookup",
    "filter",
    "count",
    "aggregate",
    "rank",
    "compare",
    "difference",
    "ratio",
    "unique_values",
    "existence",
    "unknown",
}

ALLOWED_AGGREGATIONS = {
    "none",
    "sum",
    "mean",
    "count",
    "min",
    "max",
    "median",
    "nunique",
    "mode",
}

ALLOWED_DIRECTIONS = {
    "none",
    "highest",
    "lowest",
}

ALLOWED_ANSWER_KINDS = {
    "scalar_number",
    "scalar_label",
    "boolean",
    "list_numbers",
    "list_labels",
    "unknown",
}

ALLOWED_CERTAINTY = {
    "high",
    "medium",
    "ambiguous",
}

ALLOWED_FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
}

# Backward-compatible aliases make the validator tolerant when the model emits
# one of the old execution-plan labels despite the new prompt.
OLD_OPERATION_TO_INTENT = {
    "lookup": "lookup",
    "filter": "filter",
    "count": "count",
    "aggregate": "aggregate",
    "grouped_aggregate": "aggregate",
    "argmax": "rank",
    "argmin": "rank",
    "grouped_argmax": "rank",
    "grouped_argmin": "rank",
    "comparison": "compare",
    "difference": "difference",
    "ratio": "ratio",
    "top_k": "rank",
    "unique_values": "unique_values",
    "existence": "existence",
    "unknown": "unknown",
    # A common old-schema mistake in the logs.
    "mode": "aggregate",
}

OLD_OPERATION_TO_DIRECTION = {
    "argmax": "highest",
    "grouped_argmax": "highest",
    "top_k": "highest",
    "argmin": "lowest",
    "grouped_argmin": "lowest",
}


def dataframe_schema_summary(
    df: pd.DataFrame,
    candidate_columns: list[str],
) -> str:
    """Create a compact, copy-safe schema description for the semantic LLM."""

    lines: list[str] = []

    for column in candidate_columns:
        if column not in df.columns:
            continue

        base_name = strip_type_annotation(column)
        annotation = extract_type_annotation(column)
        dtype = str(df[column].dtype)

        details = [
            json.dumps(column, ensure_ascii=False),
            f"dtype={dtype}",
        ]

        if base_name != column:
            details.append(
                f"base_name={json.dumps(base_name, ensure_ascii=False)}"
            )

        if annotation:
            details.append(f"annotation={annotation}")

        lines.append("- " + " | ".join(details))

    return "\n".join(lines)


def make_semantic_state_prompt(
    question: str,
    df: pd.DataFrame,
    candidate_columns: list[str],
) -> str:
    candidate_schema_summary = dataframe_schema_summary(
        df=df,
        candidate_columns=candidate_columns,
    )

    full_column_names = json.dumps(
        list(df.columns),
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are constructing a minimal semantic interpretation for a tabular
question-answering system.

The interpretation will guide Pandas code generation. Describe the meaning of
the question, not a Pandas execution plan. Be conservative. Do not invent
columns, filters, literal values, aggregations, or relationships.

ORIGINAL QUESTION:
{question}

LIKELY RELEVANT COLUMNS:
{candidate_schema_summary}

ALL EXACT DATAFRAME COLUMN NAMES:
{full_column_names}

Before producing JSON, determine internally in this order:
1. Which explicit filters are requested?
2. Which column supplies the returned value, if any?
3. Which column contains values to aggregate, count, compare, or rank?
4. Which column defines groups, if rows must be grouped?
5. What semantic intent, aggregation, ranking direction, and output type apply?

Return exactly one JSON object with this structure:

{{
  "intent": "lookup | filter | count | aggregate | rank | compare |
             difference | ratio | unique_values | existence | unknown",
  "answer_column": "exact dataframe column name or null",
  "value_column": "exact dataframe column name or null",
  "group_column": "exact dataframe column name or null",
  "aggregation": "none | sum | mean | count | min | max | median |
                  nunique | mode",
  "direction": "none | highest | lowest",
  "top_k": "positive integer or null",
  "filters": [
    {{
      "column": "exact dataframe column name",
      "operator": "eq | ne | gt | gte | lt | lte | contains | in",
      "value": "literal value explicitly supported by the question"
    }}
  ],
  "answer_kind": "scalar_number | scalar_label | boolean |
                  list_numbers | list_labels | unknown",
  "certainty": "high | medium | ambiguous",
  "ambiguities": [
    "short description of a genuine unresolved ambiguity"
  ]
}}

Field meanings:
- answer_column: the column whose values are returned. Use null when the answer
  is only a computed number or boolean.
- value_column: the column whose values are aggregated, counted, compared, or
  ranked. It may equal answer_column.
- group_column: the entity/category column used to group rows. For a question
  such as "Which department has the highest average salary?", Department is
  both answer_column and group_column.
- intent="rank": select the highest/lowest row or group. Use direction to say
  which one. Grouping is represented only by group_column.
- intent="aggregate": compute an aggregate without selecting a winning row or
  group. "Most common" is aggregation="mode", not an intent named "mode".
- intent="count" with aggregation="nunique" means count distinct values.
- intent="unique_values" means return the actual distinct values, not their
  count.

Rules:
1. Copy column names exactly from ALL EXACT DATAFRAME COLUMN NAMES. Never output
   prefixes such as exact_name= or base_name=.
2. LIKELY RELEVANT COLUMNS are suggestions, not restrictions.
3. The original question is the source of truth.
4. Do not infer filters that are not stated or clearly implied.
5. Do not replace explicit literals with observed maxima, minima, or nearby
   values.
6. "total", "sum", or "combined" -> aggregation="sum".
7. "average", "mean", or "on average" -> aggregation="mean".
8. "how many" or "number of" -> intent="count". Use aggregation="count"
   for rows/items and aggregation="nunique" for distinct values.
9. "most common" or "most frequent" -> intent="aggregate",
   aggregation="mode".
10. "highest", "largest", "longest", or "top" -> intent="rank",
    direction="highest". Use top_k only when a number of results is requested.
11. "lowest", "smallest", or "bottom" -> intent="rank",
    direction="lowest".
12. A row-level rank has group_column=null. A grouped rank has a non-null
    group_column and an explicit aggregation when rows must be combined.
13. Use aggregation="none" for direct lookup, filtering, existence, and raw
    row-level ranking.
14. When uncertain, set certainty="ambiguous" and describe the ambiguity.
15. Output JSON only. No Markdown and no explanation.

Example 1:
Question: Which department has the highest average monthly income?
{{
  "intent": "rank",
  "answer_column": "Department",
  "value_column": "MonthlyIncome",
  "group_column": "Department",
  "aggregation": "mean",
  "direction": "highest",
  "top_k": null,
  "filters": [],
  "answer_kind": "scalar_label",
  "certainty": "high",
  "ambiguities": []
}}

Example 2:
Question: What is the department of the employee with the highest monthly income?
{{
  "intent": "rank",
  "answer_column": "Department",
  "value_column": "MonthlyIncome",
  "group_column": null,
  "aggregation": "none",
  "direction": "highest",
  "top_k": null,
  "filters": [],
  "answer_kind": "scalar_label",
  "certainty": "high",
  "ambiguities": []
}}
""".strip()


def default_semantic_state() -> dict[str, Any]:
    return {
        "intent": "unknown",
        "answer_column": None,
        "value_column": None,
        "group_column": None,
        "aggregation": "none",
        "direction": "none",
        "top_k": None,
        "filters": [],
        "answer_kind": "unknown",
        "certainty": "ambiguous",
        "ambiguities": [
            "Semantic-state generation failed or was invalid."
        ],
    }


def _parse_old_column_roles(state: dict[str, Any]) -> dict[str, Any]:
    """Translate old column_roles into the new slots when necessary."""

    migrated: dict[str, Any] = {}
    raw_roles = state.get("column_roles")

    if not isinstance(raw_roles, list):
        return migrated

    for item in raw_roles:
        if not isinstance(item, dict):
            continue

        column = item.get("column")
        role = item.get("role")

        if not isinstance(column, str):
            continue

        if role == "measure" and "value_column" not in migrated:
            migrated["value_column"] = column
        elif role == "group" and "group_column" not in migrated:
            migrated["group_column"] = column
        elif role == "return" and "answer_column" not in migrated:
            migrated["answer_column"] = column
        elif role == "group_and_return":
            migrated.setdefault("group_column", column)
            migrated.setdefault("answer_column", column)

    return migrated


def _normalize_old_schema(state: dict[str, Any]) -> dict[str, Any]:
    """Accept old-schema output without making it the preferred format."""

    normalized = dict(state)

    for field, value in _parse_old_column_roles(state).items():
        normalized.setdefault(field, value)

    old_operation = state.get("operation_family")

    if "intent" not in normalized and old_operation in OLD_OPERATION_TO_INTENT:
        normalized["intent"] = OLD_OPERATION_TO_INTENT[old_operation]

    if (
        normalized.get("direction") in {None, "", "none"}
        and old_operation in OLD_OPERATION_TO_DIRECTION
    ):
        normalized["direction"] = OLD_OPERATION_TO_DIRECTION[old_operation]

    if old_operation == "mode" and "aggregation" not in normalized:
        normalized["aggregation"] = "mode"

    return normalized


def _unwrap_exact_name(value: str) -> str:
    text = value.strip()

    if not text.startswith("exact_name="):
        return text

    payload = text.split("=", 1)[1].strip()

    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, str) else text
    except json.JSONDecodeError:
        return payload.strip("\"'")


def _resolve_column_reference(
    value: Any,
    available_columns: list[str],
) -> str | None:
    """Resolve only exact or unambiguous column references; never fuzzy-match."""

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    text = _unwrap_exact_name(value)

    if not text or text.lower() in {"none", "null"}:
        return None

    if text in available_columns:
        return text

    casefold_matches = [
        column
        for column in available_columns
        if column.casefold() == text.casefold()
    ]

    if len(casefold_matches) == 1:
        return casefold_matches[0]

    base_name = strip_type_annotation(text)
    base_matches = [
        column
        for column in available_columns
        if strip_type_annotation(column).casefold() == base_name.casefold()
    ]

    if len(base_matches) == 1:
        return base_matches[0]

    return None


def validate_semantic_state(
    state: Any,
    available_columns: list[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Normalize and validate an LLM-produced semantic state.

    Fatal errors cause fallback. Warnings remove only the invalid field so a
    partially useful interpretation can still guide code generation.
    """

    fatal_errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(state, dict):
        return (
            default_semantic_state(),
            ["Semantic state is not a JSON object."],
            [],
        )

    state = _normalize_old_schema(state)
    validated = default_semantic_state()
    validated["ambiguities"] = []

    intent = state.get("intent", "unknown")

    if intent not in ALLOWED_INTENTS:
        warnings.append(f"Invalid intent: {intent!r}; replaced with 'unknown'.")
        intent = "unknown"

    validated["intent"] = intent

    for field_name in ("answer_column", "value_column", "group_column"):
        raw_value = state.get(field_name)
        resolved = _resolve_column_reference(raw_value, available_columns)

        if raw_value not in {None, "", "none", "null"} and resolved is None:
            warnings.append(
                f"Ignored unknown {field_name}: {raw_value!r}."
            )

        validated[field_name] = resolved

    aggregation = state.get("aggregation", "none")

    if aggregation not in ALLOWED_AGGREGATIONS:
        warnings.append(
            f"Invalid aggregation: {aggregation!r}; replaced with 'none'."
        )
        aggregation = "none"

    validated["aggregation"] = aggregation

    direction = state.get("direction", "none")

    if direction not in ALLOWED_DIRECTIONS:
        warnings.append(
            f"Invalid direction: {direction!r}; replaced with 'none'."
        )
        direction = "none"

    validated["direction"] = direction

    raw_top_k = state.get("top_k")
    top_k: int | None = None

    if raw_top_k not in {None, "", "none", "null"}:
        if isinstance(raw_top_k, bool):
            warnings.append(f"Invalid top_k: {raw_top_k!r}; ignored.")
        else:
            try:
                parsed_top_k = int(raw_top_k)
            except (TypeError, ValueError):
                warnings.append(f"Invalid top_k: {raw_top_k!r}; ignored.")
            else:
                if parsed_top_k > 0:
                    top_k = parsed_top_k
                else:
                    warnings.append(f"Invalid top_k: {raw_top_k!r}; ignored.")

    validated["top_k"] = top_k

    validated_filters: list[dict[str, Any]] = []
    raw_filters = state.get("filters", [])

    if not isinstance(raw_filters, list):
        warnings.append("filters must be a list; ignored.")
        raw_filters = []

    for item in raw_filters:
        if not isinstance(item, dict):
            warnings.append("Ignored non-object filter.")
            continue

        raw_column = item.get("column")
        column = _resolve_column_reference(raw_column, available_columns)
        operator = item.get("operator")
        value = item.get("value")

        if column is None:
            warnings.append(
                f"Ignored filter using unknown column: {raw_column!r}."
            )
            continue

        if operator not in ALLOWED_FILTER_OPERATORS:
            warnings.append(f"Ignored invalid filter operator: {operator!r}.")
            continue

        if value is None:
            warnings.append(
                f"Ignored filter without value for column {column!r}."
            )
            continue

        validated_filters.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

    validated["filters"] = validated_filters

    answer_kind = state.get("answer_kind", "unknown")

    if answer_kind not in ALLOWED_ANSWER_KINDS:
        warnings.append(
            f"Invalid answer_kind: {answer_kind!r}; replaced with 'unknown'."
        )
        answer_kind = "unknown"

    validated["answer_kind"] = answer_kind

    certainty = state.get("certainty", "ambiguous")

    if certainty not in ALLOWED_CERTAINTY:
        warnings.append(
            f"Invalid certainty: {certainty!r}; replaced with 'ambiguous'."
        )
        certainty = "ambiguous"

    validated["certainty"] = certainty

    ambiguities = state.get("ambiguities", [])

    if not isinstance(ambiguities, list):
        warnings.append("ambiguities must be a list; ignored.")
        ambiguities = []

    validated["ambiguities"] = [
        str(item).strip()
        for item in ambiguities
        if str(item).strip()
    ]

    # Only conditions that make the whole state uninformative are fatal.
    if validated["intent"] == "unknown":
        fatal_errors.append("Semantic intent is unknown.")

    has_column_signal = any(
        validated[field] is not None
        for field in ("answer_column", "value_column", "group_column")
    ) or bool(validated_filters)

    if not has_column_signal and validated["intent"] not in {"count", "existence"}:
        fatal_errors.append("No valid semantic columns or filters were identified.")

    if fatal_errors and validated["certainty"] == "high":
        validated["certainty"] = "medium"

    if validated["certainty"] == "ambiguous" and not validated["ambiguities"]:
        validated["ambiguities"].append(
            "The semantic interpretation is ambiguous."
        )

    return validated, fatal_errors, warnings


def format_semantic_state(state: dict[str, Any]) -> str:
    """Format the state for inclusion in the code-generation prompt."""

    return json.dumps(
        state,
        ensure_ascii=False,
        indent=2,
    )


def build_semantic_state(
    question: str,
    df: pd.DataFrame,
    llm: Any,
    parse_json_object: Any,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve candidates, generate a semantic interpretation, and validate it."""

    candidate_columns = find_relevant_columns(
        question=question,
        columns=list(df.columns),
        top_k=top_k,
    )

    if not candidate_columns:
        candidate_columns = list(df.columns)

    prompt = make_semantic_state_prompt(
        question=question,
        df=df,
        candidate_columns=candidate_columns,
    )

    raw_response = llm.generate(prompt)
    parsed_state = parse_json_object(raw_response)

    validated_state, validation_errors, validation_warnings = (
        validate_semantic_state(
            state=parsed_state,
            available_columns=list(df.columns),
        )
    )

    return {
        "candidate_columns": candidate_columns,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_state": parsed_state,
        "state": validated_state,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }