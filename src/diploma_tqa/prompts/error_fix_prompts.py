def make_error_fix_prompt(row: dict, df, previous_code: str, error: str) -> str:
    question = row["question"]
    answer_type = row.get("type", "unknown")

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

Column dtypes:
{df.dtypes.astype(str).to_dict()}

First 5 rows:
{df.head(5).to_string(index=False, max_colwidth=80)}

Previous code body:
{previous_code}

Fix the code.

Rules:
- Use only the dataframe df.
- Use only exact column names from the available columns list.
- If the failed code uses a column that is not in the available columns, replace it with the closest exact available column.
- Do not reuse a failed column name unless it appears exactly in the available columns list.
- Do not invent or simplify column names.
- Pay attention to annotated column names such as <gx:number>, <gx:category>, and <gx:list[category]>.
- If a column name contains suffixes like <gx:number> or <gx:category>, use the full exact column name.
- If selecting the row with the highest/lowest value in one column and returning another column, use df.loc[df[column].idxmax(), target_column] or df.loc[df[column].idxmin(), target_column].
- Do not call nlargest/nsmallest on a text/category Series with another column name. Use DataFrame.nlargest(n, column) or idxmax/idxmin instead.
- Return exactly one value of the expected answer type.
- Do not write imports.
- Do not use markdown.
- Do not repeat def answer(df).
- Output code without leading indentation. The system will indent it automatically.
- Stop immediately after the return statement.

def answer(df):
""".strip()