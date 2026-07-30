import argparse
import ast
import json
import numbers
import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
import copy

from tqdm import tqdm
from databench_eval import Evaluator

from diploma_tqa.data.databench_loader import load_qa, load_table
from diploma_tqa.llms.ollama_client import OllamaClient
from diploma_tqa.prompts.baseline_prompts import make_baseline_prompt
from diploma_tqa.prompts.error_fix_prompts import make_error_fix_prompt
from diploma_tqa.execution.code_extract import extract_answer_body
from diploma_tqa.execution.pandas_executor import execute_answer_body

from diploma_tqa.tools.dataframe_tools import find_columns

from diploma_tqa.tools.tool_prompts import (
    make_tool_planning_prompt,
    make_react_tool_planning_prompt,
    make_post_code_react_action_prompt,
    make_post_code_react_decision_prompt,
    make_post_code_react_repair_prompt, make_semantic_react_repair_prompt, make_semantic_react_critic_prompt,
)

from diploma_tqa.tools.tool_runner import (
    parse_tool_calls,
    parse_react_plan,
    execute_tool_calls,
    format_react_plan,
    parse_json_object,
    execute_post_code_react_action, parse_semantic_critic,
)

from diploma_tqa.schema.semantic_state import (
    build_semantic_state,
    format_semantic_state, ALLOWED_OPERATIONS,
)

from diploma_tqa.schema.semantic_state import (
    ALLOWED_AGGREGATIONS,
    ALLOWED_OPERATIONS,
    build_semantic_state,
)

from diploma_tqa.schema.grounded_schema import (
    GroundedSchemaIndex,
    format_grounded_schema_evidence,
)

def make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, pd.Categorical):
        return [make_json_safe(x) for x in value.tolist()]

    if isinstance(value, pd.CategoricalDtype):
        return {
            "type": "category",
            "categories": [
                make_json_safe(x) for x in value.categories.tolist()
            ],
            "ordered": bool(value.ordered),
        }

    if isinstance(value, pd.Series):
        return [make_json_safe(x) for x in value.tolist()]

    if isinstance(value, pd.Index):
        return [make_json_safe(x) for x in value.tolist()]

    if isinstance(value, np.ndarray):
        return [make_json_safe(x) for x in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [
            {str(k): make_json_safe(v) for k, v in row.items()}
            for row in value.to_dict(orient="records")
        ]

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(x) for x in value]

    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()

    if isinstance(value, pd.Interval):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    # Last-resort representation so logging cannot destroy a completed run.
    return repr(value)

def is_execution_error(pred) -> bool:
    # check whether model prediction is an error
    return isinstance(pred, str) and (
        pred.startswith("__CODE_ERROR__")
        or pred.startswith("__TIMEOUT__")
    )
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NUMBER_TOKEN = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"


def _parse_number_token(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def expected_exact_list_length(question: str) -> int | None:
    """Return an exact requested list length only when wording is unambiguous."""

    q = str(question).lower()
    non_exact_markers = (
        "up to",
        "at most",
        "no more than",
        "if available",
        "if there are fewer",
        "if there are less",
        "or fewer",
        "or less",
    )
    if any(marker in q for marker in non_exact_markers):
        return None

    patterns = (
        rf"\b(?:top|bottom|first|last)\s+({_NUMBER_TOKEN})\b",
        rf"\b(?:the\s+)?({_NUMBER_TOKEN})\s+"
        rf"(?:largest|smallest|highest|lowest|top|bottom)\b",
        rf"\blist\s+(?:the\s+)?({_NUMBER_TOKEN})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return _parse_number_token(match.group(1))
    return None


def output_type_matches(pred, answer_type: str) -> bool:
    answer_type = str(answer_type or "unknown").strip().lower()

    if answer_type == "boolean":
        return isinstance(pred, (bool, np.bool_))

    if answer_type == "number":
        return (
            isinstance(pred, numbers.Number)
            and not isinstance(pred, (bool, np.bool_))
        )

    if answer_type == "category":
        return isinstance(pred, str) and not is_execution_error(pred)

    if answer_type == "list[number]":
        return isinstance(pred, list) and all(
            isinstance(item, numbers.Number)
            and not isinstance(item, (bool, np.bool_))
            for item in pred
        )

    if answer_type == "list[category]":
        return isinstance(pred, list) and all(
            isinstance(item, str) for item in pred
        )

    return True


def _question_requires_unique_values(question: str) -> bool:
    q = str(question).lower()
    if "non-unique" in q or "non unique" in q:
        return False
    return any(word in q for word in ("unique", "distinct", "different"))


def _contains_direct_row_index_return(
    code: str,
    answer_type: str,
    question: str = "",
) -> bool:
    """Detect a raw dataframe row index returned as a category answer.

    ``df[metric].idxmax()`` returns a row index and is suspicious when the
    question asks for a name/category. In contrast,
    ``df[category].value_counts().idxmax()`` and grouped ``idxmax`` calls
    return category/group labels and must not be flagged.
    """

    if str(answer_type).lower() != "category":
        return False

    q = str(question).lower()
    if any(phrase in q for phrase in ("index", "row number", "row id")):
        return False

    body = textwrap.dedent(str(code or "")).strip("\n")
    if not body:
        return False

    try:
        wrapped = "def _candidate(df):\n" + textwrap.indent(body, "    ")
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False

    aggregate_methods = {
        "value_counts",
        "groupby",
        "agg",
        "aggregate",
        "size",
        "count",
        "nunique",
        "sum",
        "mean",
        "median",
        "mode",
        "max",
        "min",
    }
    aggregate_series_vars: set[str] = set()
    raw_series_vars: set[str] = set()
    direct_index_vars: set[str] = set()

    def has_df_reference(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id == "df"
            for child in ast.walk(node)
        )

    def has_aggregate_operation(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute)
            and child.attr in aggregate_methods
            for child in ast.walk(node)
        )

    def assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {target.id for target in targets if isinstance(target, ast.Name)}

    # Track whether intermediate Series variables are raw row-aligned Series
    # or aggregate/category-label Series. Unknown variables are not flagged.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        names = assigned_names(node)
        if has_aggregate_operation(value):
            aggregate_series_vars.update(names)
        elif has_df_reference(value):
            raw_series_vars.update(names)

    def is_direct_index_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in {"idxmax", "idxmin"}:
            return False

        receiver = node.func.value
        if has_aggregate_operation(receiver):
            return False
        if isinstance(receiver, ast.Name):
            if receiver.id in aggregate_series_vars:
                return False
            return receiver.id in raw_series_vars
        return has_df_reference(receiver)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and is_direct_index_call(node.value):
            direct_index_vars.update(assigned_names(node))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if is_direct_index_call(node.value):
            return True
        if isinstance(node.value, ast.Name) and node.value.id in direct_index_vars:
            return True

    return False

def _rank_direction_is_suspicious(question: str, code: str, columns) -> bool:
    q = str(question).lower()
    rank_request = re.search(
        rf"\b(?:top|best|highest)\s*(?:-\s*)?"
        rf"(?:(?:{_NUMBER_TOKEN})\s+)?ranked\b",
        q,
    )
    if not rank_request:
        return False

    rank_columns = {
        str(column)
        for column in columns
        if "rank" in re.sub(r"[^a-z0-9]+", " ", str(column).lower()).split()
    }
    used_columns = extract_used_columns(code, columns)
    if not (rank_columns & used_columns):
        return False

    lowered = str(code).lower()
    return ".nlargest(" in lowered or "ascending=false" in lowered.replace(" ", "")


def _repair_introduces_operation_drift(original_code: str, candidate_code: str) -> list[str]:
    """Block broad semantic rewrites for the narrow safe repair classes."""

    original = re.sub(r"\s+", "", str(original_code or "").lower())
    candidate = re.sub(r"\s+", "", str(candidate_code or "").lower())
    markers = (
        ".groupby(",
        ".value_counts(",
        ".mode(",
    )
    return [marker for marker in markers if marker not in original and marker in candidate]

def extract_used_columns(code: str, columns) -> set[str]:
    available = {str(column) for column in columns}
    quoted = set(re.findall(r"['\"]([^'\"]+)['\"]", str(code or "")))
    return quoted & available


def candidate_contract_violations(
    *,
    question: str,
    answer_type: str,
    pred,
    code: str,
    columns,
    strict_list_length: bool = True,
) -> set[str]:
    """Objective checks only; no LLM judgement is used here."""

    violations: set[str] = set()

    if is_execution_error(pred):
        violations.add("execution_error")
        return violations

    if not output_type_matches(pred, answer_type):
        violations.add("wrong_output_type")

    requested_length = expected_exact_list_length(question)
    if requested_length is not None and isinstance(pred, list):
        if len(pred) > requested_length:
            violations.add("wrong_list_length")
        elif strict_list_length and len(pred) < requested_length:
            violations.add("wrong_list_length")

    if _question_requires_unique_values(question) and isinstance(pred, list):
        normalised = [json.dumps(make_json_safe(item), sort_keys=True, ensure_ascii=False) for item in pred]
        if len(normalised) != len(set(normalised)):
            violations.add("duplicates_when_unique_required")

    q = str(question).lower()
    compact_code = re.sub(r"\s+", "", str(code or ""))
    if "exactly" in q and (">=" in compact_code or "<=" in compact_code):
        violations.add("exactly_uses_inequality")

    if _contains_direct_row_index_return(code, answer_type, question):
        violations.add("raw_row_index_return")

    if _rank_direction_is_suspicious(question, code, columns):
        violations.add("rank_direction")

    return violations


def semantic_critic_trigger_reasons(
    *,
    question: str,
    answer_type: str,
    pred,
    code: str,
    columns,
) -> set[str]:
    """Only objective contract violations trigger the conservative critic."""

    return candidate_contract_violations(
        question=question,
        answer_type=answer_type,
        pred=pred,
        code=code,
        columns=columns,
        strict_list_length=False,
    )


def _normalise_error_types(value) -> set[str]:
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = re.split(r"[|,/]", str(value or "uncertain"))
    return {part.strip().lower() for part in parts if part.strip()}


def _evidence_is_nonempty(value) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, dict) and str(item.get("fact", "")).strip():
            return True
        if isinstance(item, str) and item.strip():
            return True
    return False


def _error_types_support_violations(
    error_types: set[str],
    violations: set[str],
) -> bool:
    blocked = {
        "uncertain",
        "wrong_column",
        "wrong_value_mapping",
        "entity_mismatch",
        "code_error",
    }
    if error_types & blocked:
        return False

    type_or_shape = {
        "wrong_output_type",
        "wrong_list_length",
        "duplicates_when_unique_required",
    }
    operation = {
        "exactly_uses_inequality",
        "rank_direction",
    }
    return_column = {"raw_row_index_return"}

    if violations & type_or_shape and not (
        {"wrong_answer_type", "wrong_operation"} & error_types
    ):
        return False
    if violations & operation and "wrong_operation" not in error_types:
        return False
    if violations & return_column and "wrong_return_column" not in error_types:
        return False

    return bool(violations)


def _format_grounded_evidence_for_critic(grounded_schema) -> str:
    if not grounded_schema:
        return "None"
    try:
        rendered = format_grounded_schema_evidence(grounded_schema)
    except Exception:
        rendered = ""
    if rendered:
        return rendered
    return json.dumps(
        make_json_safe(grounded_schema),
        ensure_ascii=False,
        indent=2,
    )


def _record_verified_calls(calls, verified_columns: set[str]) -> None:
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("args") or {}
        if name == "profile_column" and isinstance(args.get("column"), str):
            verified_columns.add(args["column"])

def semantic_state_is_usable(
    semantic_state: dict | None,
    semantic_state_data: dict | None,
) -> tuple[bool, str | None]:
    """
    Decide whether semantic state is safe enough to guide code generation.

    Invalid or ambiguous states fall back to the ordinary schema-hint mode.
    """

    if not semantic_state:
        return False, "semantic state is missing"

    validation_errors = (
        semantic_state_data.get("validation_errors", [])
        if semantic_state_data
        else []
    )

    if validation_errors:
        return False, "semantic state has validation errors"

    if semantic_state.get("certainty") == "ambiguous":
        return False, "semantic state is ambiguous"

    if semantic_state.get("operation_family") == "unknown":
        return False, "semantic operation is unknown"

    return True, None

def run_simple_post_code_react_loop(
    row,
    df,
    llm,
    code,
    pred,
    tool_observations,
    attempts,
    args,
):
    post_react_steps = []

    success = not is_execution_error(pred)
    error = None if success else str(pred)

    if args.post_code_react_mode != "simple-inspect" or not success:
        return code, pred, success, error, post_react_steps, ""

    # 1. Ask for one inspection action.
    action_prompt = make_post_code_react_action_prompt(
        row=row,
        df=df,
        generated_code=code,
        prediction=pred,
        tool_observations=tool_observations,
    )

    action_raw = llm.generate(action_prompt)
    action_result = parse_json_object(action_raw)
    action = action_result.get("action", {"name": "profile_used_columns", "args": {}})

    if not isinstance(action, dict):
        action = {"name": "profile_used_columns", "args": {}}

    action_name = action.get("name", "profile_used_columns")

    if action_name == "accept":
        post_react_steps.append({
            "stage": "post_code_react_action",
            "raw_response": action_raw,
            "action_result": action_result,
            "action": action,
            "observation": "",
            "decision_result": {"decision": "keep", "reason": "Action chose accept."},
        })
        return code, pred, success, error, post_react_steps, ""

    # 2. Execute the action.
    observation = execute_post_code_react_action(action=action, df=df, code=code)

    # 3. Decide keep vs rewrite from the observation.
    decision_prompt = make_post_code_react_decision_prompt(
        row=row,
        df=df,
        generated_code=code,
        prediction=pred,
        observation=observation,
    )

    decision_raw = llm.generate(decision_prompt)
    decision_result = parse_json_object(decision_raw)
    decision = decision_result.get("decision", "keep")

    step_log = {
        "stage": "post_code_react",
        "action_raw_response": action_raw,
        "action_result": action_result,
        "action": action,
        "observation": observation,
        "decision_raw_response": decision_raw,
        "decision_result": decision_result,
        "decision": decision,
    }

    # 4. Keep old answer unless rewrite is explicitly requested.
    if decision != "rewrite":
        post_react_steps.append(step_log)
        return code, pred, success, error, post_react_steps, observation

    # 5. Rewrite once.
    rewrite_instruction = decision_result.get("rewrite_instruction", "Rewrite the code using the observation.")

    repair_prompt = make_post_code_react_repair_prompt(
        row=row,
        df=df,
        previous_code=code,
        previous_prediction=pred,
        observation=observation,
        rewrite_instruction=rewrite_instruction,
    )

    repair_raw = llm.generate(repair_prompt)
    repair_code = extract_answer_body(repair_raw)
    repair_pred = execute_answer_body(repair_code, df)

    step_log.update({
        "repair_raw_response": repair_raw,
        "repair_code": repair_code,
        "repair_prediction": repair_pred,
        "repair_success": not is_execution_error(repair_pred),
    })

    post_react_steps.append(step_log)

    attempts.append({
        "stage": "post_code_react_rewrite",
        "raw_response": repair_raw,
        "code": repair_code,
        "prediction": repair_pred,
        "observation": observation,
        "rewrite_instruction": rewrite_instruction,
    })

    if not is_execution_error(repair_pred):
        return repair_code, repair_pred, True, None, post_react_steps, observation

    # Rewrite crashed: keep old executed prediction.
    step_log["kept_previous_prediction"] = True
    return code, pred, success, error, post_react_steps, observation

def tool_call_key(call):
    return json.dumps(call, sort_keys=True, ensure_ascii=False)


def run_semantic_critic_loop(
    row,
    df,
    llm,
    code,
    pred,
    tool_observations,
    attempts,
    args,
    react_max_tool_calls,
    pre_code_tool_calls=None,
    grounded_schema=None,
):
    """Conservative post-code critic.

    The original successful prediction is always the default. A repaired
    candidate is adopted only when an objective contract violation is present,
    the critic gives a high-confidence supported diagnosis, and deterministic
    validation proves that the candidate resolves the violation without
    introducing another one.
    """

    semantic_react_steps = []
    post_code_observations_list = []

    success = not is_execution_error(pred)
    error = None if success else str(pred)

    if (
        args.post_code_react_mode != "semantic-critic"
        or not success
        or is_execution_error(pred)
    ):
        return code, pred, success, error, semantic_react_steps, ""

    question = row["question"]
    answer_type = row.get("type", "unknown")
    columns = list(df.columns)

    original_code = code
    original_pred = pred
    original_violations = semantic_critic_trigger_reasons(
        question=question,
        answer_type=answer_type,
        pred=original_pred,
        code=original_code,
        columns=columns,
    )

    # Do not expose already contract-consistent answers to speculative repair.
    if not original_violations:
        return code, pred, success, error, semantic_react_steps, ""

    grounded_evidence = _format_grounded_evidence_for_critic(grounded_schema)

    verified_columns: set[str] = set()
    _record_verified_calls(pre_code_tool_calls or [], verified_columns)
    if (grounded_schema or {}).get("active"):
        for column_info in (grounded_schema or {}).get("columns", []):
            if (
                isinstance(column_info, dict)
                and isinstance(column_info.get("exact_name"), str)
            ):
                verified_columns.add(column_info["exact_name"])

    executed_call_keys = {
        tool_call_key(call)
        for call in (pre_code_tool_calls or [])
        if isinstance(call, dict)
    }

    semantic_repair_count = 0
    semantic_critic_max_steps = 2

    for semantic_step_idx in range(1, semantic_critic_max_steps + 1):
        try:
            critic_prompt = make_semantic_react_critic_prompt(
                question=question,
                answer_type=answer_type,
                columns=columns,
                dtypes={c: str(df[c].dtype) for c in df.columns},
                preview=df.head(5).to_string(index=False),
                tool_observations=tool_observations,
                post_code_observations="\n\n".join(post_code_observations_list),
                grounded_schema_evidence=grounded_evidence,
                generated_code=original_code,
                prediction=original_pred,
                execution_error=None,
                deterministic_violations=sorted(original_violations),
            )

            critic_raw = llm.generate(critic_prompt)
            critic_result = parse_semantic_critic(
                critic_raw,
                max_tool_calls=react_max_tool_calls,
            )

            # Preserve conservative fields even if the existing parser ignores
            # unknown JSON keys.
            raw_object = parse_json_object(critic_raw)
            if isinstance(raw_object, dict):
                for field in ("confidence", "evidence"):
                    if field in raw_object:
                        critic_result[field] = raw_object[field]

            decision = str(critic_result.get("decision", "accept")).lower()
            confidence = str(critic_result.get("confidence", "low")).lower()
            error_types = _normalise_error_types(
                critic_result.get("error_type", "uncertain")
            )

            step_log = {
                "stage": f"semantic_critic_{semantic_step_idx}",
                "critic_raw": critic_raw,
                "critic_result": critic_result,
                "decision": decision,
                "accepted": decision == "accept",
                "trigger_reasons": sorted(original_violations),
                "tool_calls": [],
                "observation": "",
                "verified_columns": sorted(verified_columns),
            }

            if decision == "accept":
                semantic_react_steps.append(step_log)
                break

            verification_calls = critic_result.get("verification_tool_calls", [])
            new_verification_calls = []
            for call in verification_calls:
                if not isinstance(call, dict):
                    continue
                key = tool_call_key(call)
                if key in executed_call_keys:
                    continue
                executed_call_keys.add(key)
                new_verification_calls.append(call)

            # Both need_evidence and repair must collect proposed new evidence
            # before the critic can change a successful answer.
            if new_verification_calls:
                observation = execute_tool_calls(new_verification_calls, df)
                _record_verified_calls(new_verification_calls, verified_columns)

                labelled_observation = (
                    f"Post-code semantic critic step {semantic_step_idx} "
                    f"observations:\n{observation}"
                )
                post_code_observations_list.append(labelled_observation)
                step_log["tool_calls"] = new_verification_calls
                step_log["observation"] = observation
                step_log["verified_columns"] = sorted(verified_columns)
                step_log["kept_previous_prediction"] = True
                semantic_react_steps.append(step_log)

                if semantic_step_idx < semantic_critic_max_steps:
                    continue
                break

            if decision == "need_evidence":
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = (
                    "The critic remained uncertain without new evidence."
                )
                semantic_react_steps.append(step_log)
                break

            if decision != "repair":
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = "Unsupported critic decision."
                semantic_react_steps.append(step_log)
                break

            if semantic_repair_count >= args.post_code_react_max_retries:
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = (
                    "Maximum semantic repair retries reached."
                )
                semantic_react_steps.append(step_log)
                break

            if confidence != "high":
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = (
                    f"Critic confidence was {confidence!r}, not high."
                )
                semantic_react_steps.append(step_log)
                break

            if not _evidence_is_nonempty(critic_result.get("evidence")):
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = (
                    "No concrete supporting evidence was supplied."
                )
                semantic_react_steps.append(step_log)
                break

            if not _error_types_support_violations(
                error_types,
                original_violations,
            ):
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = (
                    "The critic error type is not safe for the objective "
                    "contract violation."
                )
                semantic_react_steps.append(step_log)
                break

            if not critic_result.get("repair_instruction"):
                step_log["kept_previous_prediction"] = True
                step_log["repair_skipped"] = "No concrete repair instruction."
                semantic_react_steps.append(step_log)
                break

            semantic_repair_count += 1

            combined_sections = [grounded_evidence]
            if tool_observations:
                combined_sections.append(tool_observations)
            if post_code_observations_list:
                combined_sections.extend(post_code_observations_list)
            combined_observations = "\n\n".join(
                section for section in combined_sections if section and section != "None"
            )

            semantic_fix_prompt = make_semantic_react_repair_prompt(
                row=row,
                df=df,
                previous_code=original_code,
                previous_prediction=original_pred,
                critic_result=critic_result,
                tool_observations=combined_observations,
                deterministic_violations=sorted(original_violations),
            )

            semantic_raw = llm.generate(semantic_fix_prompt)
            semantic_code = extract_answer_body(semantic_raw)
            semantic_pred = execute_answer_body(semantic_code, df)

            candidate_violations = candidate_contract_violations(
                question=question,
                answer_type=answer_type,
                pred=semantic_pred,
                code=semantic_code,
                columns=columns,
            )
            original_used_columns = extract_used_columns(original_code, columns)
            candidate_used_columns = extract_used_columns(semantic_code, columns)
            introduced_columns = candidate_used_columns - original_used_columns
            removed_columns = original_used_columns - candidate_used_columns
            unverified_columns = introduced_columns - verified_columns
            operation_drift = _repair_introduces_operation_drift(
                original_code,
                semantic_code,
            )

            rejection_reasons = []
            if is_execution_error(semantic_pred):
                rejection_reasons.append("candidate execution failed")
            if candidate_violations:
                rejection_reasons.append(
                    "candidate still violates: "
                    + ", ".join(sorted(candidate_violations))
                )
            if unverified_columns:
                rejection_reasons.append(
                    "candidate introduced unverified columns: "
                    + ", ".join(sorted(unverified_columns))
                )
            if removed_columns:
                rejection_reasons.append(
                    "candidate removed original dataframe columns: "
                    + ", ".join(sorted(removed_columns))
                )
            if operation_drift:
                rejection_reasons.append(
                    "candidate introduced broad operation drift: "
                    + ", ".join(operation_drift)
                )
            if make_json_safe(semantic_pred) == make_json_safe(original_pred):
                rejection_reasons.append("candidate prediction did not change")

            adopted = not rejection_reasons
            step_log.update(
                {
                    "repair_raw_response": semantic_raw,
                    "repair_code": semantic_code,
                    "repair_prediction": semantic_pred,
                    "repair_success": not is_execution_error(semantic_pred),
                    "candidate_violations": sorted(candidate_violations),
                    "introduced_columns": sorted(introduced_columns),
                    "removed_columns": sorted(removed_columns),
                    "unverified_columns": sorted(unverified_columns),
                    "operation_drift": operation_drift,
                    "repair_adopted": adopted,
                    "repair_rejection_reasons": rejection_reasons,
                }
            )
            semantic_react_steps.append(step_log)

            attempts.append(
                {
                    "stage": f"semantic_react_fix_{semantic_repair_count}",
                    "raw_response": semantic_raw,
                    "code": semantic_code,
                    "prediction": semantic_pred,
                    "critic_result": critic_result,
                    "candidate_violations": sorted(candidate_violations),
                    "repair_adopted": adopted,
                    "repair_rejection_reasons": rejection_reasons,
                    "post_code_observations": "\n\n".join(
                        post_code_observations_list
                    ),
                }
            )

            if adopted:
                code = semantic_code
                pred = semantic_pred
                success = True
                error = None
            else:
                step_log["kept_previous_prediction"] = True
            break

        except Exception as exc:
            semantic_react_steps.append(
                {
                    "stage": f"semantic_critic_{semantic_step_idx}",
                    "error": str(exc),
                    "kept_previous_prediction": True,
                }
            )
            break

    post_code_observations = "\n\n".join(post_code_observations_list)
    return code, pred, success, error, semantic_react_steps, post_code_observations


def main():
    parser = argparse.ArgumentParser()

    # Default settings
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--output-dir", default="results/smoke_test")

    # max execution retries
    parser.add_argument("--max-retries", type=int, default=0)

    # optional schema modes
    parser.add_argument(
        "--schema-mode",
        choices=[
            "none",
            "hint",
            "grounded",
            "semantic-state",
        ],
        default="none",
        help=(
            "Schema guidance mode. "
            "'hint' uses lexical candidate-column retrieval. "
            "'grounded' adds deterministic dataframe profiles and real "
            "value links without another LLM call. "
            "'semantic-state' uses one LLM call to predict validated "
            "column roles, operation family, filters, and answer kind."
        ),
    )

    parser.add_argument(
        "--grounded-max-columns",
        type=int,
        default=6,
        help=(
            "Maximum number of grounded columns shown "
            "to the code model."
        ),
    )

    parser.add_argument(
        "--grounded-max-index-values",
        type=int,
        default=12000,
        help=(
            "Per-column cap for deterministic value linking."
        ),
    )

    # optional tool usage mode (modes: auto-schema, inspect)
    parser.add_argument(
        "--tool-mode",
        choices=["none", "auto-schema", "inspect", "react-inspect"],
        default="none",
    )

    parser.add_argument(
        "--indices-file",
        default=None,
        help="Optional text file containing 0-based example indices to evaluate, one per line.",
    )

    parser.add_argument(
        "--react-observation-mode",
        choices=["raw", "plan", "plan_plus_raw"],
        default="plan",
        help="Controls what react-inspect passes to the code-generation prompt.",
    )

    parser.add_argument(
        "--post-code-react-max-retries",
        type=int,
        default=1,
        help="Maximum number of semantic critic repair attempts after successful execution.",
    )

    parser.add_argument(
        "--post-code-react-mode",
        choices=["none", "semantic-critic", "simple-inspect"],
        default="none",
        help="Optional post-execution ReAct. simple-inspect runs one inspection action then optionally rewrites once.",
    )

    args = parser.parse_args()

    REACT_MAX_STEPS = 2
    REACT_MAX_TOOL_CALLS = 3
    INSPECT_MAX_TOOL_CALLS = 3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa = load_qa(name="semeval", split="test", limit=args.limit)

    # use indices file to run selected examples - if indices_file argument is not empty
    if args.indices_file is not None:
        with open(args.indices_file, "r", encoding="utf-8") as f:
            selected_indices = [
                int(line.strip())
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        full_qa = load_qa(name="semeval", split="test", limit=None)

        if hasattr(full_qa, "select"):
            qa = full_qa.select(selected_indices)
        else:
            qa = [full_qa[i] for i in selected_indices]
    else:
        selected_indices = None

    llm = OllamaClient(model=args.model)

    predictions = []
    logs = []

    running_success = 0
    running_retried = 0

    # Keep an immutable source table and grounded index per dataset.
    # Each question gets its own dataframe copy because model-generated
    # code may mutate df.
    source_table_cache = {}
    grounded_index_cache = {}

    progress = tqdm(
        qa,
        total=len(qa),
        desc="Running examples",
        dynamic_ncols=True,
    )

    # for i, row in enumerate(progress, start=1):
    for local_i, row in enumerate(progress, start=1):
        original_index = (
            selected_indices[local_i - 1]
            if selected_indices is not None
            else local_i - 1
        )

        table_key = (
            row["dataset"],
            bool(args.lite),
        )

        source_df = source_table_cache.get(table_key)

        if source_df is None:
            source_df = load_table(
                row["dataset"],
                lite=args.lite,
            )
            source_table_cache[table_key] = source_df

        # Never let generated code mutate the cached source table.
        df = copy.deepcopy(source_df)

        tool_raw = None
        tool_calls = []
        tool_observations = ""
        react_steps = []
        latest_plan = {}

        semantic_react_steps = []

        # optional tool stage
        if args.tool_mode == "auto-schema":
            # search for relevant columns using the question - automatically
            try:
                tool_observations = find_columns(
                    df=df,
                    query=row["question"],
                    top_k=8,
                )
                tool_calls = [
                    {
                        "name": "find_columns",
                        "args": {
                            "query": row["question"],
                            "top_k": 8,
                        },
                    }
                ]
            except Exception as e:
                tool_observations = f"Tool error: {e}"

        elif args.tool_mode == "inspect":
            # ask the model if it wants to use any of the tools
            try:
                tool_prompt = make_tool_planning_prompt(
                    row=row,
                    df=df,
                    max_tool_calls=INSPECT_MAX_TOOL_CALLS,
                )

                tool_raw = llm.generate(tool_prompt)
                tool_calls = parse_tool_calls(tool_raw, max_tool_calls=INSPECT_MAX_TOOL_CALLS)
                tool_observations = execute_tool_calls(tool_calls, df)

            except Exception as e:
                tool_observations = f"Tool planning failed: {e}"


        elif args.tool_mode == "react-inspect":

            # iterative tool planning: plan -> act -> observe -> update compact plan

            react_steps = []

            all_tool_calls = []

            all_observations = []

            seen_tool_calls = set()

            latest_plan = {}

            semantic_react_steps = []

            try:

                for step in range(1, REACT_MAX_STEPS + 1):

                    previous_observations = (

                        "\n\n".join(all_observations)

                        if all_observations

                        else ""

                    )

                    tool_prompt = make_react_tool_planning_prompt(

                        row=row,

                        df=df,

                        previous_observations=previous_observations,

                        step=step,

                        max_steps=REACT_MAX_STEPS,

                        max_tool_calls=REACT_MAX_TOOL_CALLS,

                    )

                    raw_plan = llm.generate(tool_prompt)

                    plan = parse_react_plan(

                        raw_plan,

                        max_tool_calls=REACT_MAX_TOOL_CALLS,

                    )

                    thought = plan.get("thought", "")

                    current_plan = plan.get("current_plan", {})

                    if current_plan:
                        latest_plan = current_plan

                    step_calls = plan.get("tool_calls", [])

                    # Remove repeated calls within the same example.

                    new_calls = []

                    for call in step_calls:

                        key = tool_call_key(call)

                        if key not in seen_tool_calls:
                            seen_tool_calls.add(key)

                            new_calls.append(call)

                    step_calls = new_calls

                    if not step_calls:
                        react_steps.append(

                            {

                                "step": step,

                                "thought": thought,

                                "raw_response": raw_plan,

                                "current_plan": current_plan,

                                "tool_calls": [],

                                "observation": "",

                                "stop": True,

                            }

                        )

                        break

                    observation = execute_tool_calls(step_calls, df)

                    all_tool_calls.extend(step_calls)

                    all_observations.append(

                        f"Step {step} observations:\n{observation}"

                    )

                    react_steps.append(

                        {

                            "step": step,

                            "thought": thought,

                            "raw_response": raw_plan,

                            "current_plan": current_plan,

                            "tool_calls": step_calls,

                            "observation": observation,

                            "stop": bool(plan.get("stop", False)),

                        }

                    )

                    # If the model explicitly says stop after these calls,

                    # allow the next iteration only if there are remaining steps?

                    # For now, respect stop after saving the observation.

                    if plan.get("stop", False):
                        break

                compact_plan = format_react_plan(latest_plan)

                raw_observations = "\n\n".join(all_observations)

                if args.react_observation_mode == "raw":

                    tool_observations = raw_observations


                elif args.react_observation_mode == "plan":

                    tool_observations = compact_plan


                elif args.react_observation_mode == "plan_plus_raw":

                    if compact_plan and raw_observations:

                        tool_observations = (

                                compact_plan

                                + "\n\nSupporting observations:\n"

                                + raw_observations

                        )

                    else:

                        tool_observations = compact_plan or raw_observations

                tool_raw = json.dumps(react_steps, ensure_ascii=False)

                tool_calls = all_tool_calls


            except Exception as e:

                tool_observations = f"ReAct tool planning failed: {e}"

                tool_raw = None

                tool_calls = []

                react_steps = []

        grounded_schema_data = None

        if args.schema_mode == "grounded":
            grounded_index = grounded_index_cache.get(
                table_key
            )

            if grounded_index is None:
                grounded_index = GroundedSchemaIndex(
                    source_df,
                    max_index_values=(
                        args.grounded_max_index_values
                    ),
                )

                grounded_index_cache[
                    table_key
                ] = grounded_index

            grounded_schema_data = (
                grounded_index.build_evidence(
                    question=row["question"],
                    answer_type=row.get(
                        "type",
                        "unknown",
                    ),
                    max_columns=(
                        args.grounded_max_columns
                    ),
                )
            )

        semantic_state_data = None
        semantic_state = None
        semantic_state_text = ""

        if args.schema_mode == "semantic-state":
            try:
                semantic_state_data = build_semantic_state(
                    question=row["question"],
                    df=df,
                    llm=llm,
                    parse_json_object=parse_json_object,
                    top_k=8,
                )

                semantic_state = semantic_state_data["state"]
                semantic_state_text = format_semantic_state(semantic_state)

            except Exception as e:
                semantic_state_data = {
                    "candidate_columns": [],
                    "prompt": None,
                    "raw_response": None,
                    "parsed_state": None,
                    "state": None,
                    "validation_errors": [
                        f"Semantic-state generation failed: {e}"
                    ],
                }

                semantic_state = None
                semantic_state_text = ""

        effective_schema_mode = args.schema_mode
        effective_semantic_state = semantic_state
        semantic_state_fallback_reason = None

        if args.schema_mode == "grounded" and not (
                grounded_schema_data
                and grounded_schema_data.get("active")
        ):
            # Weak schema evidence is safer omitted.
            effective_schema_mode = "none"

        if args.schema_mode == "semantic-state":
            semantic_state_usable, semantic_state_fallback_reason = (
                semantic_state_is_usable(
                    semantic_state=semantic_state,
                    semantic_state_data=semantic_state_data,
                )
            )

            if not semantic_state_usable:
                effective_schema_mode = "hint"
                effective_semantic_state = None

        prompt = make_baseline_prompt(
            row=row,
            df=df,
            schema_mode=effective_schema_mode,
            tool_observations=tool_observations,
            semantic_state=effective_semantic_state,
            grounded_schema=grounded_schema_data,
        )

        raw = llm.generate(prompt)
        attempts = []

        try:
            # extract code from prediction and execute it
            code = extract_answer_body(raw)
            pred = execute_answer_body(code, df)

            attempts.append(
                {
                    "stage": "initial",
                    "raw_response": raw,
                    "code": code,
                    "prediction": pred,
                }
            )

            # error fixing loop
            retry_count = 0

            while retry_count < args.max_retries and is_execution_error(pred):
                retry_count += 1

                fix_prompt = make_error_fix_prompt(
                    row=row,
                    df=df,
                    previous_code=code,
                    error=str(pred),
                    #tool_observations=tool_observations,  # with 8B model: adds noise - keep for use with larger models
                )

                fixed_raw = llm.generate(fix_prompt)
                fixed_code = extract_answer_body(fixed_raw)
                fixed_pred = execute_answer_body(fixed_code, df)

                attempts.append(
                    {
                        "stage": f"fix_{retry_count}",
                        "raw_response": fixed_raw,
                        "code": fixed_code,
                        "prediction": fixed_pred,
                    }
                )

                code = fixed_code
                pred = fixed_pred

            if is_execution_error(pred):
                success = False
                error = str(pred)
                post_code_observations = ""
            else:
                success = True
                error = None

                if args.post_code_react_mode == "simple-inspect":
                    (
                        code,
                        pred,
                        success,
                        error,
                        post_code_react_steps,
                        post_code_observations,
                    ) = run_simple_post_code_react_loop(
                        row=row,
                        df=df,
                        llm=llm,
                        code=code,
                        pred=pred,
                        tool_observations=tool_observations,
                        attempts=attempts,
                        args=args,
                    )
                    semantic_react_steps = post_code_react_steps
                else:
                    (
                        code,
                        pred,
                        success,
                        error,
                        semantic_react_steps,
                        post_code_observations,
                    ) = run_semantic_critic_loop(
                        row=row,
                        df=df,
                        llm=llm,
                        code=code,
                        pred=pred,
                        tool_observations=tool_observations,
                        attempts=attempts,
                        args=args,
                        react_max_tool_calls=REACT_MAX_TOOL_CALLS,
                        pre_code_tool_calls=tool_calls,
                        grounded_schema=grounded_schema_data,
                    )


        except Exception as e:

            code = None

            pred = f"__CODE_ERROR__: {e}"

            error = str(e)

            success = False

            post_code_observations = ""

            attempts.append(

                {

                    "stage": "exception",

                    "raw_response": raw,

                    "code": None,

                    "prediction": pred,

                    "error": error,

                }

            )

        # stores prediction and detailed log for every example
        pred = make_json_safe(pred)
        predictions.append(pred)

        log_item = {
            "dataset": row["dataset"],
            "question": row["question"],
            "type": row.get("type"),

            "schema_mode": args.schema_mode,
            "tool_mode": args.tool_mode,
            "tool_raw": tool_raw,
            "tool_calls": tool_calls,
            "tool_observations": tool_observations,
            "react_steps": react_steps if args.tool_mode == "react-inspect" else None,
            "num_react_steps": len(react_steps) if args.tool_mode == "react-inspect" else 0,
            "num_tool_calls": len(tool_calls),

            "react_observation_mode": (
                args.react_observation_mode if args.tool_mode == "react-inspect" else None
            ),
            "react_final_plan": latest_plan if args.tool_mode == "react-inspect" else None,

            "post_code_react_mode": args.post_code_react_mode,
            "post_code_react_max_retries": args.post_code_react_max_retries,
            "semantic_react_steps": semantic_react_steps,
            "num_semantic_react_steps": len(semantic_react_steps),
            "post_code_observations": post_code_observations,

            "raw_response": raw,
            "extracted_code": code,
            "prediction": pred,
            "success": success,
            "error": error,
            "num_attempts": len(attempts),
            "attempts": attempts,

            "semantic_state": semantic_state,
            "semantic_state_candidates": (
                semantic_state_data.get("candidate_columns", [])
                if semantic_state_data
                else []
            ),
            "semantic_state_raw_response": (
                semantic_state_data.get("raw_response")
                if semantic_state_data
                else None
            ),
            "semantic_state_parsed": (
                semantic_state_data.get("parsed_state")
                if semantic_state_data
                else None
            ),
            "semantic_state_validation_errors": (
                semantic_state_data.get("validation_errors", [])
                if semantic_state_data
                else []
            ),
            "semantic_state_valid": (
                bool(semantic_state)
                and not semantic_state_data.get("validation_errors", [])
                if semantic_state_data
                else False
            ),

            "effective_schema_mode": effective_schema_mode,
            "semantic_state_fallback_reason": semantic_state_fallback_reason,

            "example_index": original_index,

            "grounded_schema": grounded_schema_data,

            "grounded_schema_active": bool(
                grounded_schema_data
                and grounded_schema_data.get("active")
            ),

            "grounded_selected_columns": (
                grounded_schema_data.get(
                    "selected_columns",
                    [],
                )
                if grounded_schema_data
                else []
            ),

            "grounded_value_links": (
                grounded_schema_data.get(
                    "value_links",
                    [],
                )
                if grounded_schema_data
                else []
            ),
        }

        logs.append(log_item)

        if success:
            running_success += 1

        if len(attempts) > 1:
            running_retried += 1

        progress.set_postfix(
            {
                "success": f"{running_success}/{local_i}",
                "retried": running_retried,
                "dataset": row["dataset"],
            }
        )

    # save logs and prediction list
    with open(output_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(make_json_safe(item), ensure_ascii=False) + "\n")

    with open(output_dir / "predictions.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred).replace("\n", " ") + "\n")

    # compute execution stats
    num_success = sum(1 for item in logs if item["success"])
    num_failed = len(logs) - num_success
    num_retried = sum(1 for item in logs if item.get("num_attempts", 1) > 1)

    attempt_counts = {}
    for item in logs:
        n = item.get("num_attempts", 1)
        attempt_counts[str(n)] = attempt_counts.get(str(n), 0) + 1

    # approximates total model calls
    total_model_calls = sum(item.get("num_attempts", 1) for item in logs)

    if args.tool_mode == "inspect":
        total_model_calls += len(logs)

    elif args.tool_mode == "react-inspect":
        total_model_calls += sum(
            item.get("num_react_steps", 0)
            for item in logs
        )

    if args.post_code_react_mode == "semantic-critic":
        for item in logs:
            for step in item.get("semantic_react_steps", []):
                # one LLM call for the critic/verifier
                total_model_calls += 1

                # one extra LLM call if repair code was generated
                if step.get("repair_raw_response") is not None:
                    total_model_calls += 1

    if args.schema_mode == "semantic-state":
        total_model_calls += len(logs)

    # evaluate predictions using the official task evaluator
    try:
        evaluator = Evaluator(qa=qa)
        acc = evaluator.eval(predictions, lite=args.lite)
        eval_error = None
    except Exception as e:
        acc = None
        eval_error = str(e)

    metrics = {
        "accuracy": acc,
        "evaluation_error": eval_error,

        "model": args.model,
        "limit": args.limit,
        "lite": args.lite,

        "schema_mode": args.schema_mode,
        "tool_mode": args.tool_mode,
        "react_observation_mode": (
            args.react_observation_mode if args.tool_mode == "react-inspect" else None
        ),

        "semantic_state_enabled": args.schema_mode == "semantic-state",

        "semantic_state_valid_examples": sum(
            1
            for item in logs
            if item.get("semantic_state") is not None
        ),

        "semantic_state_clean_examples": sum(
            1
            for item in logs
            if item.get("semantic_state") is not None
            and not item.get("semantic_state_validation_errors")
        ),

        "semantic_state_validation_error_examples": sum(
            1
            for item in logs
            if item.get("semantic_state_validation_errors")
        ),

        "semantic_state_ambiguous_examples": sum(
            1
            for item in logs
            if (
                    item.get("semantic_state")
                    and item["semantic_state"].get("certainty") == "ambiguous"
            )
        ),

        "semantic_state_operation_counts": {
            operation: sum(
                1
                for item in logs
                if (
                        item.get("semantic_state")
                        and item["semantic_state"].get("operation_family") == operation
                )
            )
            for operation in sorted(ALLOWED_OPERATIONS)
        },

        "semantic_state_aggregation_counts": {
            aggregation: sum(
                1
                for item in logs
                if (
                        item.get("semantic_state")
                        and item["semantic_state"].get("aggregation") == aggregation
                )
            )
            for aggregation in sorted(ALLOWED_AGGREGATIONS)
        },

        "semantic_state_operation_aggregation_counts": {
            f"{operation}:{aggregation}": sum(
                1
                for item in logs
                if (
                        item.get("semantic_state")
                        and item["semantic_state"].get("operation_family") == operation
                        and item["semantic_state"].get("aggregation") == aggregation
                )
            )
            for operation in sorted(ALLOWED_OPERATIONS)
            for aggregation in sorted(ALLOWED_AGGREGATIONS)
        },

        "grounded_schema_enabled": (
                args.schema_mode == "grounded"
        ),

        "grounded_schema_active_examples": sum(
            1
            for item in logs
            if item.get("grounded_schema_active")
        ),

        "grounded_schema_inactive_examples": sum(
            1
            for item in logs
            if (
                    args.schema_mode == "grounded"
                    and not item.get(
                "grounded_schema_active"
            )
            )
        ),

        "grounded_examples_with_value_links": sum(
            1
            for item in logs
            if item.get("grounded_value_links")
        ),

        "grounded_max_columns": (
            args.grounded_max_columns
        ),

        "grounded_max_index_values": (
            args.grounded_max_index_values
        ),

        "max_retries": args.max_retries,
        "error_fixing_enabled": args.max_retries > 0,

        "total_examples": len(logs),
        "execution_success": num_success,
        "execution_failed": num_failed,
        "execution_success_rate": num_success / len(logs) if logs else 0,

        "react_max_steps": REACT_MAX_STEPS if args.tool_mode == "react-inspect" else None,
        "react_max_tool_calls": REACT_MAX_TOOL_CALLS if args.tool_mode == "react-inspect" else None,
        "avg_react_steps": (
            sum(item.get("num_react_steps", 0) for item in logs) / len(logs)
            if args.tool_mode == "react-inspect" and logs
            else None
        ),
        "avg_tool_calls": (
            sum(item.get("num_tool_calls", 0) for item in logs) / len(logs)
            if logs
            else 0
        ),

        "post_code_react_mode": args.post_code_react_mode,
        "post_code_react_max_retries": args.post_code_react_max_retries,
        "avg_semantic_react_steps": (
            sum(item.get("num_semantic_react_steps", 0) for item in logs) / len(logs)
            if logs
            else 0
        ),
        "semantic_react_repaired_examples": sum(
    1
    for item in logs
    for step in item.get("semantic_react_steps", [])
    if step.get("repair_raw_response") is not None
    ),

    "semantic_react_accept_examples": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "accept"
    ),

    "semantic_react_evidence_examples": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "need_evidence"
    ),

    "semantic_react_repair_decisions": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "repair"
    ),

    "semantic_react_non_accept_decisions": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") in {"need_evidence", "repair"}
    ),

        "indices_file": args.indices_file,
        "selected_indices": selected_indices,

        "retried_examples": num_retried,
        "attempt_counts": attempt_counts,
        "total_model_calls_estimate": total_model_calls,
        "avg_model_calls_per_example": total_model_calls / len(logs) if logs else 0,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, indent=2)

    print(f"Accuracy: {acc}")
    print(f"Execution success: {num_success}/{len(logs)}")
    print(f"Retried examples: {num_retried}")
    print(f"Schema mode: {args.schema_mode}")
    print(f"Tool mode: {args.tool_mode}")
    print(f"Max retries: {args.max_retries}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()