import json
import re

from diploma_tqa.tools.dataframe_tools import find_columns, profile_column


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

    raise ValueError(f"Could not parse tool JSON: {text[:300]}")


def parse_tool_calls(raw: str, max_tool_calls: int = 2) -> list[dict]:
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

        if name in {"find_columns", "profile_column"} and isinstance(args, dict):
            valid.append({"name": name, "args": args})

    return valid


def execute_tool_calls(tool_calls: list[dict], df) -> str:
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
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool error for {name}: {e}"

        observations.append(f"Tool observation {idx}:\n{result}")

    if not observations:
        return ""

    return "\n\n".join(observations)