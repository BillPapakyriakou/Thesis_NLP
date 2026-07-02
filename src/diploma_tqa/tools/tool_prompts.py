def make_tool_planning_prompt(row: dict, df, max_tool_calls: int = 2) -> str:

    # Build prompt that asks the model whether it wants to use available tools for better dataframe inspection

    question = row["question"]
    answer_type = row.get("type", "unknown")

    return f"""
You may inspect the dataframe before writing Pandas code.

Your goal is to reduce uncertainty before code generation.
- Request the number of useful tool calls needed, up to {max_tool_calls}.
- Use multiple tool calls when they answer different uncertainties, such as finding a column and then inspecting or matching values in that column.
- Return no tool calls only for simple questions where the relevant columns and values are already obvious.

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

3. find_values(column, query)
   - Use only to match a concrete value/entity from the question to cell values in a known column.
   - Good for quoted titles, names, countries, cities, organizations, products, or specific phrases.
   - Do not use for abstract concepts or operations like year, date, rating, count, average, maximum, amount, contract, procurement, or category.
   - The column must be an exact column name from Available columns.
   - Example: {{"name": "find_values", "args": {{"column": "title", "query": "value with a view"}}}}

Rules:
- Return valid JSON only.
- Do not write markdown.
- Tool arguments must use exact column names from Available columns. Never use guessed column names.
- Request at most {max_tool_calls} tool calls.
- Return {{"tool_calls": []}} only when no useful tool call would reduce uncertainty.
- Prefer find_columns when unsure about an exact column name.
- Prefer profile_column when unsure what values a column contains or how values are formatted.
- Prefer find_values only when the question mentions a concrete entity, title, name, country, city, organization, product, or quoted phrase that may appear as a cell value.
- Do not use find_values for abstract concepts, operations, or column meanings.
- Do not call find_values unless you already know the exact column name to search.
- If the question asks about most common, least common, maximum, minimum, top-k, average, or distribution of values in a column, use profile_column on the relevant column when available.
- If the question mentions a quoted phrase, title, name, country, city, organization, or product, use find_values when a likely text/category column is available.

Return JSON in this format:
{{"tool_calls": [{{"name": "find_columns", "args": {{"query": "..."}}}}]}}
""".strip()


def make_react_tool_planning_prompt(row, df, previous_observations="", step=1, max_steps=3, max_tool_calls=3):
    question = row["question"]
    answer_type = row.get("type", "")

    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    return f"""
You may inspect the dataframe before Pandas code generation.

Your goal is to gather only the missing evidence needed to answer the question correctly.
Use tools only to identify exact column names, inspect column values/formats, or match literal question values to actual cell values.

This is an iterative planning step. Use previous observations to decide whether another tool call is useful.
Do not answer the question. Do not write code.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{cols}

Dtypes:
{dtypes}

Previous observations:
{previous_observations if previous_observations else "None"}

Available tools:
1. find_columns(query)
   - Use this to find exact dataframe column names.
   - Example: {{"name": "find_columns", "args": {{"query": "author name"}}}}

2. profile_column(column)
   - Use this to inspect dtype, sample values, top values, and numeric summary.
   - Use this when the answer depends on value format, numeric range, common values, dictionary-like strings, maximum/minimum, top-k, average, or most/least common values.
   - The column argument must be an exact column name from Available columns.
   - Example: {{"name": "profile_column", "args": {{"column": "author_name"}}}}

3. find_values(column, query)
   - Match a literal value/entity from the question to actual cell values in a known column.
   - Use for concrete cell values such as quoted titles, person names, countries, cities, organizations, products, codes, categories, labels, exact phrases, or literal years/dates used as filters.
   - Do not use for operations or abstract column meanings such as count, average, maximum, minimum, top-k, distribution, amount, or "first three letters".
   - The column must be an exact column name from Available columns.
   - Example: {{"name": "find_values", "args": {{"column": "title", "query": "value with a view"}}}}

Rules:
- Return valid JSON only.
- Do not write markdown.
- Tool arguments must use exact column names from Available columns. Never use guessed column names.
- Request at most {max_tool_calls} tool calls.
- Request only tool calls that reduce uncertainty before code generation.
- Use multiple tool calls only when they answer different uncertainties.
- Prefer find_columns when unsure about an exact column name.
- Prefer profile_column when column values, formats, ranges, common values, or nested/dictionary-like values matter.
- Prefer find_values when the question contains a literal cell value that may need exact matching.
- Never use placeholders such as "<result from previous tool call>" as column names.

Planning rules:
- Do not convert uncertain observations into facts.
- The plan must separate computation columns from return columns.
- If the question asks for a label/name/category, identify the return column that contains that label.
- If the question asks for numeric values or IDs, do not choose a name/label column unless observations prove that it stores the requested numeric values.
- For "largest/smallest values", sort the values themselves unless the question says most common, most frequent, or highest count.
- For "most common/frequent", use value counts/mode.
- For "highest/maximum X by group/entity", rank by X but return the requested entity/label column, not the dataframe index.
- For boolean questions about sale/discount/availability/status, prefer explicit status/flag columns over numeric proxy columns.
- If a computation column and a return/display column differ, include a value_mappings entry.
- Use avoid for misleading columns or interpretations, with a reason.

Stopping rule:
Return stop=true only when:
- needed filter/group/aggregate/sort/return columns are known,
- literal filter values are matched or not needed,
- the operation type is clear,
- the expected answer representation is clear,
- no useful tool call remains.

Otherwise return stop=false with useful tool calls and the best evidence plan so far.

Return JSON in this format:
{{
  "remaining_uncertainty": "Need to inspect whether the candidate column stores numeric ids or labels.",
  "tool_calls": [
    {{"name": "profile_column", "args": {{"column": "some_exact_column"}}}}
  ],
  "stop": false,
  "current_plan": {{
    "operation_type": "unknown | unique_values | sort_values_take_k | value_counts_top_k | argmax_return_label | groupby_aggregate_return_label | boolean_any_condition | filter_then_aggregate",
    "columns": {{
      "filter": [],
      "group_by": [],
      "aggregate": [],
      "sort_by": [],
      "return": []
    }},
    "value_mappings": [],
    "matched_values": [],
    "avoid": [],
    "warnings": []
  }}
}}
""".strip()