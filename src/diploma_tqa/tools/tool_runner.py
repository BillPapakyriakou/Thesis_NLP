import json
import re

from diploma_tqa.tools.dataframe_tools import find_columns, profile_column, find_values


def extract_json_object(text: str) -> dict:
    # Extract json object from model response - model should return json,
    # but may output it in markdown code fences
    text = text.strip()

    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # if response contains extra data, try to find the json part
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

    raise ValueError(f"Could not parse tool JSON: {text[:300]}")


def parse_tool_calls(raw: str, max_tool_calls: int = 3) -> list[dict]:
    # this reads the models json response and extracts only valid tool calls

    data = extract_json_object(raw)
    calls = data.get("tool_calls", [])

    if not isinstance(calls, list):
        return []

    valid = []
    for call in calls[:max_tool_calls]:
        if not isinstance(call, dict):
            continue

        name = call.get("name")
        args = call.get("args", {})

        # only allow known tools with valid dictionary arguments
        # Example: {{"name": "profile_column", "args": {{"column": "author_name"}}}}
        if name in {"find_columns", "profile_column", "find_values"} and isinstance(args, dict):
            valid.append({"name": name, "args": args})

    return valid


def format_react_plan(plan: dict) -> str:
    if not plan:
        return ""

    lines = ["ReAct inspection plan for code generation:"]

    relevant_columns = plan.get("relevant_columns", [])
    value_mappings = plan.get("value_mappings", [])
    operation = plan.get("operation", "")
    avoid = plan.get("avoid", [])

    if relevant_columns:
        lines.append(f"- Relevant columns: {relevant_columns}")

    if value_mappings:
        lines.append("- Value mappings:")
        for item in value_mappings:
            lines.append(f"  - {item}")

    if operation:
        lines.append(f"- Operation: {operation}")

    if avoid:
        lines.append("- Avoid:")
        for item in avoid:
            lines.append(f"  - {item}")

    return "\n".join(lines)

def parse_react_plan(raw: str, max_tool_calls: int = 3) -> dict:
    obj = extract_json_object(raw)

    if not isinstance(obj, dict):
        return {
            "thought": "",
            "tool_calls": [],
            "stop": True,
            "parse_error": "Invalid JSON object",
        }

    thought = obj.get("thought", "")
    if not isinstance(thought, str):
        thought = ""

    stop = bool(obj.get("stop", False))
    calls = obj.get("tool_calls", [])

    valid_calls = []
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue

            name = call.get("name")
            args = call.get("args", {})

            if name in {"find_columns", "profile_column", "find_values"} and isinstance(args, dict):
                valid_calls.append({"name": name, "args": args})

            if len(valid_calls) >= max_tool_calls:
                break

    current_plan = obj.get("current_plan", {})
    if not isinstance(current_plan, dict):
        current_plan = {}

    def clean_string_list(value):
        if not isinstance(value, list):
            return []
        return [str(x) for x in value if isinstance(x, (str, int, float))]

    current_plan = {
        "relevant_columns": clean_string_list(current_plan.get("relevant_columns", [])),
        "value_mappings": clean_string_list(current_plan.get("value_mappings", [])),
        "operation": str(current_plan.get("operation", "") or ""),
        "avoid": clean_string_list(current_plan.get("avoid", [])),
    }

    return {
        "thought": thought,
        "tool_calls": valid_calls,
        "stop": stop or len(valid_calls) == 0,
        "current_plan": current_plan,
        "parse_error": None,
    }


def parse_semantic_critic(raw: str, max_tool_calls: int = 3) -> dict:
    obj = extract_json_object(raw)

    if not isinstance(obj, dict):
        return {
            "decision": "accept",
            "accept": True,
            "reason": "Invalid critic JSON; accepting original prediction.",
            "error_type": "none",
            "answer_contract": {},
            "verification_tool_calls": [],
            "repair_instruction": "",
            "must_use_columns": [],
            "avoid_columns": [],
            "must_return": "",
            "parse_error": "Invalid JSON object",
        }

    def clean_string_list(value):
        if not isinstance(value, list):
            return []
        return [str(x) for x in value if isinstance(x, (str, int, float))]

    decision = str(obj.get("decision", "") or "").strip().lower()

    if decision not in {"accept", "need_evidence", "repair"}:
        accept = obj.get("accept", True)
        if isinstance(accept, bool):
            decision = "accept" if accept else "repair"
        else:
            decision = "accept"

    calls = obj.get("verification_tool_calls", [])
    valid_calls = []

    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue

            name = call.get("name")
            args = call.get("args", {})

            if name in {"find_columns", "profile_column", "find_values"} and isinstance(args, dict):
                valid_calls.append({"name": name, "args": args})

            if len(valid_calls) >= max_tool_calls:
                break

    answer_contract = obj.get("answer_contract", {})
    if not isinstance(answer_contract, dict):
        answer_contract = {}

    return {
        "decision": decision,
        "accept": decision == "accept",
        "reason": str(obj.get("reason", "") or ""),
        "error_type": str(obj.get("error_type", "none") or "none"),
        "answer_contract": answer_contract,
        "verification_tool_calls": valid_calls,
        "repair_instruction": str(obj.get("repair_instruction", "") or ""),
        "must_use_columns": clean_string_list(obj.get("must_use_columns", [])),
        "avoid_columns": clean_string_list(obj.get("avoid_columns", [])),
        "must_return": str(obj.get("must_return", "") or ""),
        "parse_error": None,
    }


def execute_tool_calls(tool_calls: list[dict], df) -> str:
    # runs the requested tool calls and returns the output as text

    observations = []

    for idx, call in enumerate(tool_calls, start=1):
        name = call["name"]
        args = call["args"]

        try:
            if name == "find_columns":
                result = find_columns(
                    df=df,
                    query=str(args.get("query", "")),
                    top_k=int(args.get("top_k", 5)),
                )
            elif name == "profile_column":
                result = profile_column(
                    df=df,
                    column=str(args.get("column", "")),
                )
            elif name == "find_values":
                result = find_values(
                    df=df,
                    column=str(args.get("column", "")),
                    query=str(args.get("query", "")),
                    top_k=int(args.get("top_k", 5)),
                    min_score=float(args.get("min_score", 0.65)),
                )
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool error for {name}: {e}"

        # Keeps tool observation
        observations.append(f"Tool observation {idx}:\n{result}")

    if not observations:
        return ""

    return "\n\n".join(observations)