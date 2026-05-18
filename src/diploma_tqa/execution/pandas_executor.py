import numpy as np
import pandas as pd


def execute_answer_body(body: str, df: pd.DataFrame, timeout: int = 30):
    full_code = (
        "def answer(df):\n"
        f"{body}\n\n"
        "result = answer(df)\n"
    )

    try:
        namespace = {
            "pd": pd,
            "np": np,
            "df": df,
        }

        exec(full_code, namespace)
        result = namespace["result"]

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
        return f"__CODE_ERROR__: {e}"