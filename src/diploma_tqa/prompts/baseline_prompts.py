import json

from diploma_tqa.schema.schema_linker import make_schema_hint


def make_baseline_prompt(
    row: dict,
    df,
    schema_mode: str = "none",
    tool_observations: str = "",
    semantic_state=None,
) -> str:
    """
    Build the main code-generation prompt for a table-question example.

    Supported schema modes:
    - none: no extra schema guidance
    - hint: lexical schema-linking hint
    - semantic-state: validated LLM-generated semantic interpretation
    """

    question = row["question"]
    answer_type = row.get("type", "unknown")

    schema_section = ""

    if schema_mode == "hint":
        schema_section = f"""
Schema hint:
{make_schema_hint(question, list(df.columns))}
""".strip()

    elif schema_mode == "semantic-state":
        if semantic_state:
            semantic_state_json = json.dumps(
                semantic_state,
                ensure_ascii=False,
                indent=2,
            )

            schema_section = f"""
Proposed semantic interpretation:
{semantic_state_json}

Semantic-interpretation instructions:
- Treat this interpretation as a planning suggestion, not as verified truth.
- The original question, exact DataFrame columns, dtypes, sample rows, and tool
  observations are more authoritative.
- You may use exact DataFrame columns not listed in the interpretation when the
  question clearly requires them.
- Ignore any field that conflicts with the question or table evidence.
- Never invent filters, literal values, proxy relationships, or column meanings.
- Generic phrases such as "given season", "given year", "in the dataset", and
  "for a person" are not literal filter values.
- Do not interpret one column as another concept without explicit evidence.

Meaning of the semantic fields:
- intent describes the user-level task, not a required Pandas implementation.
- answer_column supplies the value to return. It may be null for a computed
  scalar number or boolean.
- value_column supplies values to aggregate, count, compare, or rank.
- group_column defines groups. It is null for row-level ranking.
- aggregation says how rows are combined.
- direction is highest or lowest for rank questions.
- top_k is used only when multiple ranked results are explicitly requested.

Implementation guidance for rank questions:
- If intent="rank", group_column is not null, and aggregation is not "none",
  aggregate value_column by group_column, then select the highest/lowest group.
- If intent="rank" and group_column is null, rank individual rows by
  value_column. Return answer_column from the winning row when answer_column is
  present; otherwise return the winning value itself.
- Do not group merely because answer_column exists.
- Do not infer aggregation="sum" from words such as "largest", "highest",
  "most", or "longest" unless the question explicitly asks to combine rows.

Aggregation guidance:
- aggregation="mode" means the most frequent value.
- aggregation="nunique" means the number of distinct non-null values.
- intent="unique_values" means return actual unique values using
  .dropna().unique().tolist(), not the distinct count.
- aggregation="count" means count rows or non-null values according to the
  original question.

Numeric and dtype guidance:
- Before sum, mean, min, max, ranking, or numeric comparison, inspect the dtype.
- If a numeric value is stored as formatted text, clean it before using
  pd.to_numeric(..., errors="coerce").
- Do not call nlargest(), nsmallest(), max(), or min() directly on categorical
  or string columns when numeric comparison is intended.

Example grouped rank:
Question:
Which department has the highest average monthly income?

Relevant interpretation:
intent="rank", answer_column="Department", value_column="MonthlyIncome",
group_column="Department", aggregation="mean", direction="highest"

Computation:
df.groupby("Department")["MonthlyIncome"].mean().idxmax()

Example row-level rank:
Question:
What is the department of the employee with the highest monthly income?

Relevant interpretation:
intent="rank", answer_column="Department", value_column="MonthlyIncome",
group_column=null, aggregation="none", direction="highest"

Computation:
df.loc[df["MonthlyIncome"].idxmax(), "Department"]
""".strip()

        else:
            schema_section = """
Semantic-state generation was unavailable or invalid.
Answer using the original question, DataFrame schema, and tool observations.
""".strip()

    tool_section = ""

    if tool_observations:
        tool_section = f"""
Tool observations:
{tool_observations}
""".strip()

    optional_sections = "\n\n".join(
        section
        for section in [
            schema_section,
            tool_section,
        ]
        if section
    )

    if optional_sections:
        optional_sections = "\n\n" + optional_sections

    return f"""
You are a Python pandas assistant. Your task is to answer a question about a dataframe.

Write syntactically correct Python code that completes the function below.

Question:
{question}

Expected answer type:
{answer_type}

Allowed output types:
- boolean: True or False
- category: one string/category value
- number: one int or float
- list[category]: a list of string/category values
- list[number]: a list of int/float values

Important rules:
- Use only pandas/numpy operations on df.
- Do not guess entity names, values, or answers. Compute them from df.
- Return only the value requested by the question, with no extra associated values.
- Never invent column names. Use only exact column names from DataFrame columns or tool observations.
- If a natural-language field name is not present, choose the closest exact column from the provided columns.
- If the question asks "any" or "is there", return one boolean using .any() or a grouped condition.
- If the question asks "all", "every", or "for any of their posts", use .all() or groupby when needed.
- If the question asks "mainly", interpret it as proportion > 0.5.
- If the question asks for the "most", "highest", "largest", "longest", or "top N",
  compute the relevant entity using sorting, groupby, idxmax(), or nlargest().
- Before numeric ranking, comparison, or aggregation, inspect the dtype.
- If the column is already numeric, use it directly.
- If numeric values contain currency symbols, spaces, non-breaking spaces,
  or decimal commas, clean them before pd.to_numeric(..., errors="coerce").
- Preserve decimal commas correctly: "6,20" should become "6.20", not "620".
- Do not call nlargest(), nsmallest(), max(), or min() directly on categorical
  or string columns when a numeric comparison is intended.
- If a top entity has an empty or missing name/value, skip it and use the next valid one.
- Do not return a Series, DataFrame, tuple, or dictionary unless the expected
  answer type explicitly requires a list.
- Stop immediately after the return statement.
- Use tool observations when they identify exact column names or useful column values.
- If a column contains dictionary-like strings such as "{{'key': value}}",
  parse them with ast.literal_eval(x).get("key") instead of direct indexing
  ["key"], because some rows may not contain every key.
- Do not call literal_eval directly.
- When extracting multiple fields from a dictionary-like column, parse the
  column once into a separate variable and extract all fields from that parsed object.
- Do not overwrite the original column before extracting all required fields.
- Follow requested output transformations from the original question, such as
  converting month numbers to month names or returning the first three letters.

DataFrame columns:
{list(df.columns)}{optional_sections}

Column dtypes:
{df.dtypes.astype(str).to_dict()}

First 5 rows:
{df.head(5).to_string(index=False, max_colwidth=80)}

Complete this function:

def answer(df):
    result = None
    # write code here
    return result
""".strip()