from __future__ import annotations

import json
from typing import Any

import pandas as pd

from diploma_tqa.schema.schema_linker import (
    extract_type_annotation,
    find_relevant_columns,
    strip_type_annotation,
)


ALLOWED_ROLES = {
    "filter",
    "measure",
    "group",
    "return",
    "group_and_return",
}

ALLOWED_OPERATIONS = {
    "lookup",
    "filter",
    "count",
    "aggregate",
    "grouped_aggregate",
    "argmax",
    "argmin",
    "grouped_argmax",
    "grouped_argmin",
    "comparison",
    "difference",
    "ratio",
    "top_k",
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




def dataframe_schema_summary(
    df: pd.DataFrame,
    candidate_columns: list[str],
) -> str:
    """
    Create a compact schema description for the semantic-state LLM call.

    It intentionally avoids sending the entire dataframe or large value lists.
    """

    lines: list[str] = []

    for column in candidate_columns:
        if column not in df.columns:
            continue

        base_name = strip_type_annotation(column)
        annotation = extract_type_annotation(column)
        dtype = str(df[column].dtype)

        details = [
            f"exact_name={json.dumps(column, ensure_ascii=False)}",
            f"base_name={json.dumps(base_name, ensure_ascii=False)}",
            f"dtype={dtype}",
        ]

        if annotation:
            details.append(f"annotation={annotation}")

        lines.append("- " + ", ".join(details))

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

    full_schema_summary = dataframe_schema_summary(
        df=df,
        candidate_columns=list(df.columns),
    )

    return f"""
You are constructing a minimal semantic interpretation for a tabular
question-answering system.

Your output will guide Pandas code generation. Be conservative. Do not invent
columns, filters, values, aggregations, or operations.

ORIGINAL QUESTION:
{question}

LIKELY RELEVANT COLUMNS:
{candidate_schema_summary}

FULL TABLE SCHEMA:
{full_schema_summary}

Return exactly one JSON object with this structure:

{{
  "column_roles": [
    {{
      "column": "exact dataframe column name",
      "role": "filter | measure | group | return | group_and_return"
    }}
  ],
  "operation_family": "lookup | filter | count | aggregate |
                       grouped_aggregate | argmax | argmin |
                       grouped_argmax | grouped_argmin | comparison |
                       difference | ratio | top_k | unique_values |
                       existence | unknown",
  "aggregation": "none | sum | mean | count | min | max |
                median | nunique | mode",                     
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

Rules:

1. Use only exact column names from FULL TABLE SCHEMA.
2. LIKELY RELEVANT COLUMNS are suggestions, not restrictions.
3. If the question clearly refers to a column outside the likely candidates,
   use the appropriate column from FULL TABLE SCHEMA.
4. The original question is the source of truth.
5. Do not replace explicit literals with observed maxima, minima, or nearby values.
6. Do not infer filters that are not stated or clearly implied.
8. "Which/what person, country, category, product..." usually asks for a label.
9. "How many" usually asks for a scalar number.
10. For "highest total X by Y":
   - Y is normally group_and_return,
   - X is normally measure,
   - operation_family is grouped_argmax.
11. For "lowest total X by Y", use grouped_argmin.
12. For "highest X" where the numeric value itself is requested, use argmax with
   answer_kind scalar_number.
13. When uncertain, use certainty="ambiguous" and record the ambiguity rather
    than inventing an interpretation.
14. Set aggregation according to the wording:
    - "total", "sum", or "combined" -> sum
    - "average", "mean", or "on average" -> mean
    - "how many", "number of rows", or "count" -> count
    - "different", "distinct", or "unique number of" -> nunique
    - "most common" or "most frequent" -> mode
    - "maximum value" -> max
    - "minimum value" -> min
    - "median" -> median   
15. Use aggregation="none" for direct lookup, filtering, existence, raw-value
    ranking, and other operations that do not aggregate rows.     
16. Output JSON only. No Markdown and no explanation.
""".strip()


def default_semantic_state() -> dict[str, Any]:
    return {
        "column_roles": [],
        "operation_family": "unknown",
        "aggregation": "none",
        "filters": [],
        "answer_kind": "unknown",
        "certainty": "ambiguous",
        "ambiguities": [
            "Semantic-state generation failed or was invalid."
        ],
    }


def validate_semantic_state(
    state: Any,
    available_columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """
    Normalize and validate an LLM-produced semantic state.

    Invalid fields are removed or replaced rather than trusted.
    """

    errors: list[str] = []
    valid_columns = set(available_columns)

    if not isinstance(state, dict):
        return default_semantic_state(), ["Semantic state is not a JSON object."]

    validated = default_semantic_state()
    validated["ambiguities"] = []

    # Validate column roles.
    validated_roles: list[dict[str, str]] = []
    seen_role_pairs: set[tuple[str, str]] = set()

    raw_roles = state.get("column_roles", [])

    if not isinstance(raw_roles, list):
        errors.append("column_roles must be a list.")
        raw_roles = []

    for item in raw_roles:
        if not isinstance(item, dict):
            errors.append("Ignored non-object column role.")
            continue

        column = item.get("column")
        role = item.get("role")

        if column not in valid_columns:
            errors.append(f"Ignored unknown role column: {column!r}")
            continue

        if role not in ALLOWED_ROLES:
            errors.append(f"Ignored invalid role: {role!r}")
            continue

        pair = (column, role)

        if pair not in seen_role_pairs:
            seen_role_pairs.add(pair)
            validated_roles.append(
                {
                    "column": column,
                    "role": role,
                }
            )

    validated["column_roles"] = validated_roles

    # Validate operation.
    operation = state.get("operation_family", "unknown")

    if operation not in ALLOWED_OPERATIONS:
        errors.append(f"Invalid operation_family: {operation!r}")
        operation = "unknown"

    validated["operation_family"] = operation

    # Validate aggregation.
    aggregation = state.get("aggregation", "none")

    if aggregation not in ALLOWED_AGGREGATIONS:
        errors.append(f"Invalid aggregation: {aggregation!r}")
        aggregation = "none"

    validated["aggregation"] = aggregation

    # Validate consistency between operation and aggregation.
    operations_that_can_aggregate = {
        "count",
        "aggregate",
        "grouped_aggregate",
        "grouped_argmax",
        "grouped_argmin",
        "comparison",
        "difference",
        "ratio",
        "top_k",
    }

    if (
            aggregation != "none"
            and operation not in operations_that_can_aggregate
    ):
        errors.append(
            f"Aggregation {aggregation!r} may be inconsistent with "
            f"operation_family {operation!r}."
        )

    if (
            operation in {
        "aggregate",
        "grouped_aggregate",
        "grouped_argmax",
        "grouped_argmin",
    }
            and aggregation == "none"
    ):
        errors.append(
            f"operation_family {operation!r} normally requires "
            "an explicit aggregation."
        )

    measure_columns = {
        item["column"]
        for item in validated_roles
        if item["role"] == "measure"
    }

    if aggregation not in {"none", "count"} and not measure_columns:
        errors.append(
            f"Aggregation {aggregation!r} normally requires a measure column."
        )

    group_columns = {
        item["column"]
        for item in validated_roles
        if item["role"] in {"group", "group_and_return"}
    }

    if (
            operation in {
        "grouped_aggregate",
        "grouped_argmax",
        "grouped_argmin",
    }
            and not group_columns
    ):
        errors.append(
            f"operation_family {operation!r} requires a group or "
            "group_and_return column."
        )

    # Validate filters.
    validated_filters: list[dict[str, Any]] = []
    raw_filters = state.get("filters", [])

    if not isinstance(raw_filters, list):
        errors.append("filters must be a list.")
        raw_filters = []

    for item in raw_filters:
        if not isinstance(item, dict):
            errors.append("Ignored non-object filter.")
            continue

        column = item.get("column")
        operator = item.get("operator")
        value = item.get("value")

        if column not in valid_columns:
            errors.append(f"Ignored filter using unknown column: {column!r}")
            continue

        if operator not in ALLOWED_FILTER_OPERATORS:
            errors.append(f"Ignored invalid filter operator: {operator!r}")
            continue

        if value is None:
            errors.append(f"Ignored filter without value for column {column!r}.")
            continue

        validated_filters.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

    validated["filters"] = validated_filters

    # Validate answer kind.
    answer_kind = state.get("answer_kind", "unknown")

    if answer_kind not in ALLOWED_ANSWER_KINDS:
        errors.append(f"Invalid answer_kind: {answer_kind!r}")
        answer_kind = "unknown"

    validated["answer_kind"] = answer_kind

    # Validate certainty.
    certainty = state.get("certainty", "ambiguous")

    if certainty not in ALLOWED_CERTAINTY:
        errors.append(f"Invalid certainty: {certainty!r}")
        certainty = "ambiguous"

    validated["certainty"] = certainty

    # Validate ambiguities.
    ambiguities = state.get("ambiguities", [])

    if not isinstance(ambiguities, list):
        errors.append("ambiguities must be a list.")
        ambiguities = []

    validated["ambiguities"] = [
        str(item).strip()
        for item in ambiguities
        if str(item).strip()
    ]

    if errors and validated["certainty"] == "high":
        validated["certainty"] = "medium"

    if not validated_roles:
        validated["certainty"] = "ambiguous"

        if not validated["ambiguities"]:
            validated["ambiguities"].append(
                "No valid column roles were identified."
            )

    return validated, errors


def format_semantic_state(state: dict[str, Any]) -> str:
    """
    Format the state for inclusion in the code-generation prompt.
    """

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
    """
    Retrieve candidate columns, ask the LLM for a semantic state,
    and validate the result.
    """

    candidate_columns = find_relevant_columns(
        question=question,
        columns=list(df.columns),
        top_k=top_k,
    )

    # A lexical linker can miss semantic synonyms. If it finds nothing,
    # provide the full schema rather than making the LLM reason with no columns.
    if not candidate_columns:
        candidate_columns = list(df.columns)

    prompt = make_semantic_state_prompt(
        question=question,
        df=df,
        candidate_columns=candidate_columns,
    )

    raw_response = llm.generate(prompt)
    parsed_state = parse_json_object(raw_response)

    validated_state, validation_errors = validate_semantic_state(
        state=parsed_state,
        available_columns=list(df.columns),
    )

    return {
        "candidate_columns": candidate_columns,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_state": parsed_state,
        "state": validated_state,
        "validation_errors": validation_errors,
    }