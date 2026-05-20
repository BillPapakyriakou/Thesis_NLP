def make_tool_planning_prompt(row: dict, df, max_tool_calls: int = 2) -> str:
    question = row["question"]
    answer_type = row.get("type", "unknown")

    return f"""
You may inspect the dataframe before writing Pandas code.

Your goal is to request useful tool calls that help answer the question.
Request tools only if they help identify exact column names or understand column values.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{list(df.columns)}

Available tools:
1. find_columns(query)
   - Use this to find exact dataframe column names.
   - Example: {{"name": "find_columns", "args": {{"query": "author name"}}}}

2. profile_column(column)
   - Use this to inspect dtype, sample values, top values, and numeric summary.
   - The column argument must be an exact column name from Available columns.
   - Example: {{"name": "profile_column", "args": {{"column": "author_name"}}}}

Rules:
- Return valid JSON only.
- Do not write markdown.
- Request at most {max_tool_calls} tool calls.
- If no tool is needed, return {{"tool_calls": []}}.
- Prefer find_columns when unsure about an exact column name.
- Prefer profile_column when unsure what values a column contains.

Return JSON in this format:
{{"tool_calls": [{{"name": "find_columns", "args": {{"query": "..."}}}}]}}
""".strip()