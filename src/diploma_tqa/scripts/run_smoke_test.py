import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from databench_eval import Evaluator

from diploma_tqa.data.databench_loader import load_qa, load_table
from diploma_tqa.llms.ollama_client import OllamaClient
from diploma_tqa.prompts.baseline_prompts import make_baseline_prompt
from diploma_tqa.prompts.error_fix_prompts import make_error_fix_prompt
from diploma_tqa.execution.code_extract import extract_answer_body
from diploma_tqa.execution.pandas_executor import execute_answer_body

from diploma_tqa.tools.dataframe_tools import find_columns
from diploma_tqa.tools.tool_prompts import make_tool_planning_prompt
from diploma_tqa.tools.tool_runner import parse_tool_calls, execute_tool_calls



def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.tolist())

    if isinstance(obj, pd.DataFrame):
        return make_json_safe(obj.to_dict(orient="records"))

    return obj

def is_execution_error(pred) -> bool:
    return isinstance(pred, str) and (
        pred.startswith("__CODE_ERROR__")
        or pred.startswith("__TIMEOUT__")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--output-dir", default="results/smoke_test")

    parser.add_argument("--max-retries", type=int, default=0)

    parser.add_argument(
        "--schema-mode",
        choices=["none", "hint"],
        default="none",
    )

    parser.add_argument(
        "--tool-mode",
        choices=["none", "auto-schema", "inspect"],
        default="none",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa = load_qa(name="semeval", split="test", limit=args.limit)
    llm = OllamaClient(model=args.model)

    predictions = []
    logs = []

    running_success = 0
    running_retried = 0

    progress = tqdm(
        qa,
        total=len(qa),
        desc="Running examples",
        dynamic_ncols=True,
    )

    for i, row in enumerate(progress, start=1):
        df = load_table(row["dataset"], lite=args.lite)

        tool_raw = None
        tool_calls = []
        tool_observations = ""

        # ------------------------------------------------------------
        # Optional tool stage
        # ------------------------------------------------------------
        if args.tool_mode == "auto-schema":
            try:
                tool_observations = find_columns(
                    df=df,
                    query=row["question"],
                    top_k=8,
                )
                tool_calls = [
                    {
                        "name": "find_columns",
                        "args": {
                            "query": row["question"],
                            "top_k": 8,
                        },
                    }
                ]
            except Exception as e:
                tool_observations = f"Tool error: {e}"

        elif args.tool_mode == "inspect":
            try:
                tool_prompt = make_tool_planning_prompt(
                    row=row,
                    df=df,
                    max_tool_calls=2,
                )

                tool_raw = llm.generate(tool_prompt)
                tool_calls = parse_tool_calls(tool_raw, max_tool_calls=2)
                tool_observations = execute_tool_calls(tool_calls, df)

            except Exception as e:
                tool_observations = f"Tool planning failed: {e}"

        # ------------------------------------------------------------
        # Initial code generation
        # ------------------------------------------------------------
        prompt = make_baseline_prompt(
            row=row,
            df=df,
            schema_mode=args.schema_mode,
            tool_observations=tool_observations,
        )

        raw = llm.generate(prompt)
        attempts = []

        try:
            code = extract_answer_body(raw)
            pred = execute_answer_body(code, df)

            attempts.append(
                {
                    "stage": "initial",
                    "raw_response": raw,
                    "code": code,
                    "prediction": pred,
                }
            )

            # --------------------------------------------------------
            # Optional error-fixing loop
            # --------------------------------------------------------
            retry_count = 0

            while retry_count < args.max_retries and is_execution_error(pred):
                retry_count += 1

                fix_prompt = make_error_fix_prompt(
                    row=row,
                    df=df,
                    previous_code=code,
                    error=str(pred),
                    #tool_observations=tool_observations,
                )

                fixed_raw = llm.generate(fix_prompt)
                fixed_code = extract_answer_body(fixed_raw)
                fixed_pred = execute_answer_body(fixed_code, df)

                attempts.append(
                    {
                        "stage": f"fix_{retry_count}",
                        "raw_response": fixed_raw,
                        "code": fixed_code,
                        "prediction": fixed_pred,
                    }
                )

                code = fixed_code
                pred = fixed_pred

            if is_execution_error(pred):
                success = False
                error = str(pred)
            else:
                success = True
                error = None

        except Exception as e:
            code = None
            pred = f"__CODE_ERROR__: {e}"
            error = str(e)
            success = False

            attempts.append(
                {
                    "stage": "exception",
                    "raw_response": raw,
                    "code": None,
                    "prediction": pred,
                    "error": error,
                }
            )

        pred = make_json_safe(pred)
        predictions.append(pred)

        log_item = {
            "dataset": row["dataset"],
            "question": row["question"],
            "type": row.get("type"),

            "schema_mode": args.schema_mode,
            "tool_mode": args.tool_mode,
            "tool_raw": tool_raw,
            "tool_calls": tool_calls,
            "tool_observations": tool_observations,

            "raw_response": raw,
            "extracted_code": code,
            "prediction": pred,
            "success": success,
            "error": error,
            "num_attempts": len(attempts),
            "attempts": attempts,
        }

        logs.append(log_item)

        if success:
            running_success += 1

        if len(attempts) > 1:
            running_retried += 1

        progress.set_postfix(
            {
                "success": f"{running_success}/{i}",
                "retried": running_retried,
                "dataset": row["dataset"],
            }
        )

    # ------------------------------------------------------------
    # Save logs and predictions
    # ------------------------------------------------------------
    with open(output_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(make_json_safe(item), ensure_ascii=False) + "\n")

    with open(output_dir / "predictions.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred).replace("\n", " ") + "\n")

    # ------------------------------------------------------------
    # Execution stats
    # ------------------------------------------------------------
    num_success = sum(1 for item in logs if item["success"])
    num_failed = len(logs) - num_success
    num_retried = sum(1 for item in logs if item.get("num_attempts", 1) > 1)

    attempt_counts = {}
    for item in logs:
        n = item.get("num_attempts", 1)
        attempt_counts[str(n)] = attempt_counts.get(str(n), 0) + 1

    total_model_calls = sum(item.get("num_attempts", 1) for item in logs)

    if args.tool_mode == "inspect":
        # One extra LLM call per example for tool planning.
        total_model_calls += len(logs)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    try:
        evaluator = Evaluator(qa=qa)
        acc = evaluator.eval(predictions, lite=args.lite)
        eval_error = None
    except Exception as e:
        acc = None
        eval_error = str(e)

    metrics = {
        "accuracy": acc,
        "evaluation_error": eval_error,

        "model": args.model,
        "limit": args.limit,
        "lite": args.lite,

        "schema_mode": args.schema_mode,
        "tool_mode": args.tool_mode,

        "max_retries": args.max_retries,
        "error_fixing_enabled": args.max_retries > 0,

        "total_examples": len(logs),
        "execution_success": num_success,
        "execution_failed": num_failed,
        "execution_success_rate": num_success / len(logs) if logs else 0,

        "retried_examples": num_retried,
        "attempt_counts": attempt_counts,
        "total_model_calls_estimate": total_model_calls,
        "avg_model_calls_per_example": total_model_calls / len(logs) if logs else 0,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, indent=2)

    print(f"Accuracy: {acc}")
    print(f"Execution success: {num_success}/{len(logs)}")
    print(f"Retried examples: {num_retried}")
    print(f"Schema mode: {args.schema_mode}")
    print(f"Tool mode: {args.tool_mode}")
    print(f"Max retries: {args.max_retries}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()