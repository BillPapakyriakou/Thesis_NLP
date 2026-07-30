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
    grounded_schema_evidence="",
    deterministic_violations=None,
):
    deterministic_violations = deterministic_violations or []

    return f"""
You are a conservative post-code semantic critic for a Pandas table-question-answering system.

The current code already executed successfully and is the default answer.
Your burden of proof is to preserve it unless a specific, demonstrable contract violation exists.

You can do one of three things:
1. accept: preserve the current code and prediction.
2. need_evidence: request dataframe evidence before deciding.
3. repair: request one minimal repair only when the error is clear, high-confidence, and supported by evidence.

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

Deterministic contract violations detected by the runner:
{deterministic_violations if deterministic_violations else "None"}

Grounded schema evidence:
{grounded_schema_evidence if grounded_schema_evidence else "None"}

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
   - Use only when an exact dataframe column may be missing from consideration.

2. profile_column(column)
   - Use to verify the semantic meaning, value format, range, or examples of an exact column.
   - When proposing a column replacement, profile the proposed column before repair.

3. find_values(column, query)
   - Use only to verify a literal entity or exact cell value from the question.
   - A failed value match is not evidence that the answer should be 0, False, None, or an empty list.

Preservation rules:
- Do not repair merely because another implementation is possible.
- Do not repair a code path that is plausibly consistent with the question and evidence.
- A different interpretation is not enough to repair.
- If uncertainty remains, use need_evidence. If no new useful evidence can be requested, accept.
- Never change a column without verified evidence that the replacement represents the question better.
- Additional helper columns or intermediate computations are not inherently errors.
- Absence from the five-row preview is not evidence of absence from the dataframe.
- Do not infer a unit conversion unless the source unit is explicitly established.
- If your reason says the current code or prediction is correct, decision must be accept.

Semantic rules:
- "exactly" means equality, not at least or approximately.
- "most common" and "most frequent" require frequency counting.
- "largest" and "smallest" values refer to magnitude, not frequency, unless the question explicitly says common/frequent.
- "non-unique" or "they can be repeated" means duplicates are allowed; it does not mean most frequent.
- "unique", "different", and "distinct" require deduplication.
- For a rank column, rank 1 is normally better than a larger rank unless table evidence states otherwise.
- Do not turn a correct requested list length into a shorter list through unnecessary grouping.

Repair eligibility:
- decision="repair" requires confidence="high".
- decision="repair" requires at least one concrete evidence item.
- The repair must address one of the deterministic violations supplied by the runner.
- Do not use repair for speculative wrong_column, entity_mismatch, or uncertain diagnoses.
- If more dataframe evidence is needed, use need_evidence rather than repair.
- Give a concrete, minimal repair instruction. Do not redesign unrelated parts of the code.
- Request at most 3 verification tool calls.

Return one compact JSON object in exactly this shape. Use one-line string values, escape embedded quotes, and do not add comments or markdown:
{{
  "decision": "accept",
  "confidence": "high",
  "reason": "short explanation",
  "evidence": [],
  "error_type": "none",
  "verification_tool_calls": [],
  "repair_instruction": "",
  "must_use_columns": [],
  "avoid_columns": [],
  "must_return": ""
}}

Replace the example values with your decision. Allowed values are:
- decision: accept, need_evidence, repair
- confidence: low, medium, high
- error_type: none, wrong_column, wrong_operation, wrong_return_column, wrong_value_mapping, wrong_answer_type, entity_mismatch, code_error, uncertain

For evidence, use a JSON array of short strings. Example:
["The expected type is list[category] but the prediction contains 6 values when exactly 5 were requested."]

For verification_tool_calls, use objects such as:
[{{"name": "profile_column", "args": {{"column": "Lifter Name"}}}}]

The output must be parseable by json.loads without repair.
""".strip()


def make_semantic_react_repair_prompt(
    row,
    df,
    previous_code,
    previous_prediction,
    critic_result,
    tool_observations="",
    deterministic_violations=None,
):
    question = row["question"]
    answer_type = row.get("type", "unknown")
    deterministic_violations = deterministic_violations or []

    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    preview = df.head(5).to_string(index=False)

    answer_contract = critic_result.get("answer_contract", {})
    reason = critic_result.get("reason", "")
    evidence = critic_result.get("evidence", [])
    repair_instruction = critic_result.get("repair_instruction", "")
    must_use_columns = critic_result.get("must_use_columns", [])
    avoid_columns = critic_result.get("avoid_columns", [])
    must_return = critic_result.get("must_return", "")

    return f"""
You are making one minimal repair to Pandas code for a table-question-answering task.

The previous code executed successfully. Preserve every part of it that the verified critic evidence has not specifically disproved.

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

Deterministic violations that the repair must resolve:
{deterministic_violations}

Inspection, grounded-schema, and verification observations:
{tool_observations if tool_observations else "None"}

Previous code:
{previous_code}

Previous prediction:
{previous_prediction}

Critic reason:
{reason}

Critic evidence:
{evidence}

Answer contract:
{answer_contract}

Minimal repair instruction:
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
- Always include an explicit return statement.
- Use only existing dataframe columns.
- Make the smallest possible change needed to resolve the listed deterministic violation.
- Preserve existing filters, grouping, duplicate handling, list length, comparison boundaries, unit interpretation, and return column unless the verified evidence explicitly identifies that exact component as wrong.
- Do not introduce a new dataframe column unless it is present in the supplied verified observations or grounded schema evidence.
- Do not change equality to an inequality when the question says exactly.
- Do not change largest/smallest magnitude into most-frequent logic.
- Do not add grouping merely because the question mentions a season, year, person, or category.
- Do not perform unit conversion unless the input unit is established by the evidence.
- If the question requests N items, return exactly N items unless it explicitly permits fewer.
- If the question requests unique/distinct/different values, deduplicate the output.
- For top-ranked/best-ranked rows using a rank column, smaller rank values are better unless evidence says otherwise.
- Return a value compatible with the expected answer type.

Corrected answer(df) body:
""".strip()


def make_post_code_react_action_prompt(row, df, generated_code, prediction, tool_observations=""):
    question = row["question"]
    answer_type = row.get("type", "unknown")
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    return f"""
You are debugging a Pandas table-QA answer.

The code executed successfully, but executable code can still use the wrong column, wrong value, or wrong operation.

Your job is to choose at most one useful inspection action before deciding whether to keep or rewrite.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{columns}

Dtypes:
{dtypes}

Pre-code inspection observations:
{tool_observations if tool_observations else "None"}

Generated code:
{generated_code}

Prediction:
{prediction}

Available actions:
1. profile_used_columns
   - Inspect the dataframe columns used by the generated code.
   - Prefer this action when unsure whether the code used sensible columns.

2. profile_column(column)
   - Inspect one exact column.
   - Use this when a likely relevant column was not used by the generated code.

3. find_values(column, query)
   - Search for a literal entity/value from the question in one exact column.
   - Use only for concrete names, titles, places, dates, labels, codes, or quoted values.

4. accept
   - Use only if the code is obviously aligned with the question and no inspection would help.

Rules:
- Return valid JSON only.
- Do not write code.
- Do not answer the original question.
- Choose only one action.
- Prefer profile_used_columns unless another action is clearly more useful.
- Do not use find_values for operations such as count, max, average, most common, or total.

Return JSON:
{{
  "thought": "one short sentence explaining what you want to check",
  "action": {{"name": "profile_used_columns | profile_column | find_values | accept", "args": {{}}}}
}}
""".strip()


def make_post_code_react_decision_prompt(row, df, generated_code, prediction, observation):
    question = row["question"]
    answer_type = row.get("type", "unknown")
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    return f"""
You inspected the dataframe after code execution.

Decide whether to keep the current answer or rewrite the code once.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{columns}

Dtypes:
{dtypes}

Generated code:
{generated_code}

Prediction:
{prediction}

Observation:
{observation}

Decision rules:
- Use keep if the observation supports that the code used the right columns, values, operation, and return type.
- Use rewrite if the code likely used the wrong column, wrong value, wrong operation, wrong return type, or produced NaN/null.
- If rewriting, give one concrete instruction to the coder.
- Do not answer the original question directly.
- Return valid JSON only.

Return JSON:
{{
  "decision": "keep | rewrite",
  "reason": "one short reason",
  "rewrite_instruction": "empty if keep; concrete instruction if rewrite"
}}
""".strip()


def make_post_code_react_repair_prompt(row, df, previous_code, previous_prediction, observation, rewrite_instruction):
    question = row["question"]
    answer_type = row.get("type", "unknown")
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    preview = df.head(5).to_string(index=False)

    return f"""
Rewrite the body of answer(df).

The previous code executed, but post-code inspection found a likely semantic mistake.

Question:
{question}

Expected answer type:
{answer_type}

Available columns:
{columns}

Dtypes:
{dtypes}

First 5 rows:
{preview}

Previous code:
{previous_code}

Previous prediction:
{previous_prediction}

Post-code observation:
{observation}

Rewrite instruction:
{rewrite_instruction}

Rules:
- Return only the body of answer(df), not a full function definition.
- Do not write markdown.
- Use only exact dataframe column names.
- Return exactly the requested answer value.
- Do not return a dataframe index unless the question explicitly asks for an index.
- If ranking or comparing numeric values, convert with pd.to_numeric(..., errors="coerce").
- Always include an explicit return statement.
- Use this exact structure whenever possible:
  result = ...
  return result
- Do not write a bare expression followed by return result.
- Every returned variable must be assigned in the code.

Corrected answer(df) body:
""".strip()
