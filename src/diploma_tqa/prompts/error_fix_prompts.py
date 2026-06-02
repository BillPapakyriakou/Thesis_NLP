def make_error_fix_prompt(
    row: dict,
    df,
    previous_code: str,
    error: str,
    tool_observations: str = "",
) -> str:

    # Builds the error fixing prompt for a table-question example.

    question = row["question"]
    answer_type = row.get("type", "unknown")

    tool_section = ""
    if tool_observations:
        tool_section = f"""

Tool observations:
{tool_observations}
"""

    return f"""
You are fixing Python pandas code for a table question answering task.

The previous code failed during execution.

Question:
{question}

Expected answer type:
{answer_type}

Execution error:
{error}

Available dataframe columns:
{list(df.columns)}
{tool_section}

Column dtypes:
{df.dtypes.astype(str).to_dict()}

First 5 rows:
{df.head(5).to_string(index=False, max_colwidth=80)}

Previous code body:
{previous_code}

Fix the code.

Rules:
- Use only the dataframe df.
- Use only the exact column names listed above.
- Use tool observations when they identify exact column names or useful column values.
- Do not invent or simplify column names.
- If a column name contains suffixes like <gx:number> or <gx:category>, use the full exact column name.
- If a column contains dictionary-like strings such as "{{'key': value}}", parse them with ast.literal_eval before accessing keys.
- Return exactly one value of the expected answer type.
- Do not write imports.
- Do not use markdown.
- Do not repeat def answer(df).
- Output only the function body, without leading indentation. The system will indent it automatically.
- Stop immediately after the return statement.

Output only the corrected function body:
""".strip()