from diploma_tqa.schema.schema_linker import make_schema_hint

def make_baseline_prompt(row: dict, df, schema_mode: str = "none", tool_observations: str = "",) -> str:

    # Builds the main code-generation prompt for a table-question example.

    question = row["question"]
    answer_type = row.get("type", "unknown")

    schema_hint = ""

    if schema_mode == "hint":
        schema_hint = f"""

    Schema hint:
    {make_schema_hint(question, list(df.columns))}
    """

    tool_section = ""

    if tool_observations:
        tool_section = f"""

    Tool observations:
    {tool_observations}
    """

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
- Never invent column names. Use only exact column names from DataFrame columns or tool observations. If a natural-language field name is not present, choose the closest exact column from the provided columns.
- If the question asks "any" or "is there", return a single boolean using .any() or a grouped condition.
- If the question asks "all", "every", or "for any of their posts", use .all() or groupby when needed.
- If the question asks "mainly", interpret it as proportion > 0.5.
- If the question asks for the "most", "highest", "largest", "longest", or "top N", compute the relevant entity using sorting, groupby, idxmax(), or nlargest(). Before numeric ranking or comparison, convert the target column with pd.to_numeric(..., errors="coerce") unless it is already numeric.
- Do not call nlargest(), nsmallest(), max(), or min() directly on categorical/string columns when a numeric comparison is intended.
- If a top entity has an empty or missing name/value, skip it and use the next valid one.
- Do not return a Series, DataFrame, tuple, or dictionary unless the expected answer type explicitly requires a list.
- Stop immediately after the return statement.
- Use tool observations when they identify exact column names or useful column values.
- If a column contains dictionary-like strings such as "{{'key': value}}", parse them with ast.literal_eval(x).get("key") instead of direct indexing ["key"], because some rows may not contain every key. Do not call literal_eval directly.
- When extracting multiple fields from a dictionary-like column, parse the column once into a separate variable and extract all fields from that parsed object. Do not overwrite the original column before extracting all needed fields.

DataFrame columns:
{list(df.columns)}
{schema_hint}
{tool_section}

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