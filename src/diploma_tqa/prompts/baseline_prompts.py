from diploma_tqa.schema.schema_linker import make_schema_hint

def make_baseline_prompt(row: dict, df, schema_mode: str = "none") -> str:
    question = row["question"]
    answer_type = row.get("type", "unknown")

    schema_hint = ""

    if schema_mode == "hint":
        schema_hint = f"""

    Schema hint:
    {make_schema_hint(question, list(df.columns))}
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
- If the question asks "any" or "is there", return a single boolean using .any() or a grouped condition.
- If the question asks "all", "every", or "for any of their posts", use .all() or groupby when needed.
- If the question asks "mainly", interpret it as proportion > 0.5.
- If the question asks for the "most", "highest", "largest", "longest", or "top N", compute the relevant entity using sorting, groupby, idxmax(), or nlargest().
- If a top entity has an empty or missing name/value, skip it and use the next valid one.
- Do not return a Series, DataFrame, tuple, or dictionary unless the expected answer type explicitly requires a list.
- Stop immediately after the return statement.

DataFrame columns:
{list(df.columns)}
{schema_hint}

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