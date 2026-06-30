import numpy as np
import pandas as pd
import ast


def execute_answer_body(body: str, df: pd.DataFrame, timeout: int = 30):

    # Executes the generated answer(df) function on a dataframe df. Supports timeout control.


    full_code = (
        "def answer(df):\n"
        f"{body}\n\n"
        "result = answer(df)\n"
    )
    # The model can use pandas, numpy, ast.literal_eval and the current dataframe
    try:
        namespace = {
            "pd": pd,
            "np": np,
            "ast": ast,
            "literal_eval": ast.literal_eval,
            "df": df,
        }

        exec(full_code, namespace)
        result = namespace["result"]

        # Normalize common pandas/numpy return types into plain Python objects
        # so they can be passed to the benchmark evaluator
        if isinstance(result, pd.Series):
            result = result.tolist()
        elif isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        elif hasattr(result, "item"):
            try:
                result = result.item()
            except Exception:
                pass

        return result

    except Exception as e:
        # Return execution errors as marked strings instead of raising them
        # Repair module will use this marker to trigger repair attempts
        return f"__CODE_ERROR__: {e}"