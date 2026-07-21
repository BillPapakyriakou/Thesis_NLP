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
Proposed semantic state:
{semantic_state_json}

Semantic-state instructions:
- Treat the semantic state as a planning suggestion, not as verified truth.
- The original question, full DataFrame schema, dtypes, sample rows, and tool
  observations are more authoritative than the semantic state.
- You may use exact DataFrame columns that are not present in the semantic
  state when the question clearly requires them.
- Ignore any role, filter, operation, or aggregation that conflicts with the
  original question or table evidence.
- Never invent a filter, literal value, proxy relationship, or column meaning.
- Generic phrases such as "given season", "given year", "in the dataset", and
  "for a person" are not literal filter values.
- Do not interpret one column as another concept without explicit evidence.
  For example, do not use Heredity as a proxy for gender.

Operation guidance:
- Use grouped_argmax or grouped_argmin only when the question asks to compare
  groups after aggregating rows.
- Explicit grouped aggregation signals include:
  "total", "sum", "combined", "average", "mean", "number of", and "count".
- If the question asks for the row associated with the largest or smallest
  individual value, use row-level idxmax() or idxmin().
- Do not infer aggregation="sum" merely from words such as "largest",
  "highest", "most", or "longest".

Aggregation guidance:
- aggregation="mode" means the most frequent value.
- aggregation="nunique" means the number of distinct non-null values.
- For operation_family="unique_values", return the actual unique values using
  .dropna().unique().tolist(), not the distinct count.
- aggregation="count" means counting rows or non-null values according to the
  original question.

Implementation guidance:
- Before sum, mean, min, max, ranking, or numeric comparison, inspect the dtype.
- If a numeric value is stored as formatted text, clean it before using
  pd.to_numeric(..., errors="coerce").
- Follow requested output transformations from the original question, such as
  converting month numbers to month names or returning the first three letters.

Example grouped aggregation:

Question:
Which department has the highest average monthly income?

Computation:
df.groupby("Department")["MonthlyIncome"].mean().idxmax()

Example row-level selection:

Question:
What is the department of the employee with the highest monthly income?

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