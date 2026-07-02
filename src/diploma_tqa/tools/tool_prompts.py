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

Your goal is to gather only the missing information needed to answer the question.
Use tools only if they help identify exact column names, understand column values,
or match concrete entities to cell values.

This is an iterative planning step. Use previous observations to decide whether another tool call is useful.
If enough information is already available, stop.

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
   - The column argument must be an exact column name from Available columns.
   - Example: {{"name": "profile_column", "args": {{"column": "author_name"}}}}

3. find_values(column, query)
   - Match a literal value/entity from the question to cell values in a known column.
   - Use for quoted titles, person names, countries, cities, organizations, products, codes, or exact phrases.
   - Do not use for operations or abstract concepts such as year, date, rating, count, average, maximum, minimum, top, first, amount, contract, procurement, category, or distribution.
   - The column must be an exact column name from Available columns.
   - Example: {{"name": "find_values", "args": {{"column": "title", "query": "value with a view"}}}}

Rules:
- Return valid JSON only.
- Do not write markdown.
- Tool arguments must use exact column names from Available columns. Never use guessed column names.
- Request at most {max_tool_calls} tool calls.
- Request only tool calls that reduce uncertainty before code generation.
- Return no tool calls only when the relevant columns and values are already obvious.
- Use multiple tool calls only when they answer different uncertainties.
- Prefer find_columns when unsure about an exact column name.
- Prefer profile_column when the answer depends on value format, distribution, numeric range, common values, maximum, minimum, top-k, average, or most/least common values.
- Prefer find_values only when the question contains a literal cell value such as a quoted title, person name, country, city, organization, product, code, or exact phrase.
- Do not use find_values for operations, comparisons, aggregations, column meanings, or generic words from the question.
- Never use placeholders such as "<result from previous tool call>" as column names.

Additional ReAct rules:
- Do not answer the question.
- Do not write code.
- The "thought" field must be one short sentence describing what uncertainty remains.
- Maintain a "current_plan" object that summarizes what the coder should do.
- The current_plan must be based only on the question, available columns, dtypes, and observations.
- If several columns could match the question, inspect or compare them before choosing.
- Do not commit to one column only because find_columns returned it.
- Use the "avoid" field for columns or interpretations that look misleading.
- If enough information is available, return stop=true with no tool calls and a complete current_plan.
- Otherwise return stop=false with useful tool calls and the best current_plan so far.

Return JSON in this format:
{{
  "thought": "I need to compare the possible weekday columns.",
  "tool_calls": [
    {{"name": "profile_column", "args": {{"column": "Weekday"}}}},
    {{"name": "profile_column", "args": {{"column": "Weekday_1"}}}}
  ],
  "stop": false,
  "current_plan": {{
    "relevant_columns": ["Weekday", "Weekday_1"],
    "value_mappings": [],
    "operation": "Choose the column whose values are weekday names.",
    "avoid": ["Do not return numeric weekday ids if the question asks for names."]
  }}
}}

Return JSON:
""".strip()




def make_semantic_react_critic_prompt(
    question,
    answer_type,
    columns,
    dtypes,
    preview,
    tool_observations,
    generated_code,
    prediction,
    execution_error=None,
    post_code_observations="",
):
    return f"""
You are a post-code semantic ReAct critic for a Pandas table-question-answering system.

Your job is to decide whether the generated code and prediction actually answer the question.

You can do one of three things:
1. accept: the code and prediction answer the question.
2. need_evidence: you need more dataframe evidence before deciding.
3. repair: the semantic error is clear enough to give a concrete code-repair instruction.

Do not answer the original question directly.
Do not write code.
Return valid JSON only.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{columns}

Dtypes:
{dtypes}

Data preview:
{preview}

Pre-code inspection observations:
{tool_observations if tool_observations else "None"}

Post-code verification observations:
{post_code_observations if post_code_observations else "None"}

Generated code:
{generated_code}

Execution error:
{execution_error if execution_error else "None"}

Prediction:
{prediction}

Available verification tools:
1. find_columns(query)
   - Use when the generated code may have used the wrong column or missed a better candidate.
   - Example: {{"name": "find_columns", "args": {{"query": "sale discount offer status"}}}}

2. profile_column(column)
   - Use to inspect values, formats, numeric ranges, common values, or dictionary-like strings.
   - The column must be an exact column name from Available columns.
   - Example: {{"name": "profile_column", "args": {{"column": "Month"}}}}

3. find_values(column, query)
   - Use to match a literal question value to actual cell values in a known column.
   - Use for names, titles, countries, cities, products, organizations, codes, labels, categories, years, dates, seasons, or exact phrases.
   - The column must be an exact column name from Available columns.
   - Example: {{"name": "find_values", "args": {{"column": "year", "query": "2012"}}}}

Before deciding, build an answer contract:
- What operation does the question require?
- Which columns are used for filtering?
- Which columns are used for grouping?
- Which columns are used for aggregation or sorting?
- Which column/value should be returned?
- What representation is expected: number, category, boolean, list[number], or list[category]?
- If the computation column and return column differ, make that explicit.

Common semantic errors to check:
- Returning a dataframe index instead of the requested label/name/value.
- Ranking by the right metric but returning the wrong column.
- Using a numeric ID column when the question asks for a name/label/month/weekday.
- Using a name/label column when the question asks for numerical values or IDs.
- Using value_counts/mode when the question asks for largest/smallest values.
- Sorting values when the question asks for most common/most frequent.
- Using a proxy column when an explicit semantic/status column may exist.
- Ignoring a literal entity, year, date, or season from the question.
- Returning the wrong expected answer type.
- Mishandling dictionary-like string columns.

Decision rules:
- Use decision="accept" only when the code, prediction, and answer contract are consistent.
- Use decision="need_evidence" when column meaning, value format, or entity matching is unclear.
- Use decision="repair" when the code contradicts the answer contract, even if the exact corrected code is not obvious.
- If the prediction type or returned value does not match the question, repair rather than accept.
- If the code returns an index but the question asks for a name/category/value, repair.
- If the code uses a column that is not mentioned in the answer contract, repair or request evidence.
- Do not request evidence that is already present.
- Never accept only because the code executed successfully. Executable code can still be semantically wrong.
- Request at most 3 verification tool calls.

Return JSON in this format:
{{
  "decision": "accept | need_evidence | repair",
  "accept": true or false,
  "reason": "one short explanation",
  "error_type": "none | wrong_column | wrong_operation | wrong_return_column | wrong_value_mapping | wrong_answer_type | entity_mismatch | code_error | uncertain",
  "answer_contract": {{
    "operation": "short description",
    "filter_columns": [],
    "group_columns": [],
    "aggregate_columns": [],
    "sort_columns": [],
    "return_columns": [],
    "expected_representation": "number | category | boolean | list[number] | list[category] | unknown",
    "notes": []
  }},
  "verification_tool_calls": [
    {{"name": "find_columns", "args": {{"query": "..."}}}}
  ],
  "repair_instruction": "If decision is repair, give a concrete instruction for regenerating the code. Otherwise empty string.",
  "must_use_columns": [],
  "avoid_columns": [],
  "must_return": "Describe what the corrected code should return. If not repairing, empty string."
}}
""".strip()


def make_semantic_react_repair_prompt(
    row,
    df,
    previous_code,
    previous_prediction,
    critic_result,
    tool_observations="",
):
    question = row["question"]
    answer_type = row.get("type", "unknown")

    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    preview = df.head(5).to_string(index=False)

    answer_contract = critic_result.get("answer_contract", {})
    reason = critic_result.get("reason", "")
    repair_instruction = critic_result.get("repair_instruction", "")
    must_use_columns = critic_result.get("must_use_columns", [])
    avoid_columns = critic_result.get("avoid_columns", [])
    must_return = critic_result.get("must_return", "")

    return f"""
You are repairing Pandas code for a table-question-answering task.

The previous code executed, but a post-code semantic critic found that it may not answer the question correctly.
Rewrite the body of answer(df).

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{columns}

Dtypes:
{dtypes}

Data preview:
{preview}

Inspection and verification observations:
{tool_observations if tool_observations else "None"}

Previous code:
{previous_code}

Previous prediction:
{previous_prediction}

Critic reason:
{reason}

Answer contract:
{answer_contract}

Repair instruction:
{repair_instruction}

Must use columns if relevant:
{must_use_columns}

Avoid columns if relevant:
{avoid_columns}

Corrected code must return:
{must_return}

Rules:
- Return only the body of answer(df), not a full function definition.
- Do not write markdown.
- The code must be executable as an indented function body.
- Always include an explicit return statement.
- Every variable that is returned must be defined in the code.
- Use only columns that exist in the dataframe.
- Do not return dataframe indices unless the question explicitly asks for indices.
- If you use idxmax or idxmin, store the index, then use it to select the requested return column.
- If ranking by one column but the question asks for a name/category/value, return the requested column value, not the index.
- If the question asks for numerical values or IDs, do not return name/label columns.
- If the question asks for names/labels/months/weekdays, do not return numeric ID columns unless that is the only available representation.
- If the question asks for largest/smallest values, sort the values themselves unless it asks for most common or most frequent.
- If the question asks for most common/frequent, use value_counts or mode.
- Use exact matched values from observations when available.
- Return a value compatible with the expected answer type.

Corrected answer(df) body:
""".strip()