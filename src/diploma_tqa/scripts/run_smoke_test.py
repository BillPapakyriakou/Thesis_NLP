import argparse
import json
from pathlib import Path

from databench_eval import Evaluator
from diploma_tqa.data.databench_loader import load_qa, load_table
from diploma_tqa.llms.ollama_client import OllamaClient
from diploma_tqa.prompts.baseline_prompts import make_baseline_prompt
from diploma_tqa.execution.code_extract import extract_answer_body
from diploma_tqa.execution.pandas_executor import execute_answer_body
from diploma_tqa.prompts.error_fix_prompts import make_error_fix_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--output-dir", default="results/smoke_test")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--schema-mode",choices=["none", "hint"],default="none",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa = load_qa(name="semeval", split="dev", limit=args.limit)
    llm = OllamaClient(model=args.model)

    predictions = []
    logs = []

    for row in qa:
        df = load_table(row["dataset"], lite=args.lite)
        prompt = make_baseline_prompt(
            row=row,
            df=df,
            schema_mode=args.schema_mode,
        )

        raw = llm.generate(prompt)
        attempts = []

        try:
            code = extract_answer_body(raw)
            pred = execute_answer_body(code, df)

            attempts.append({
                "stage": "initial",
                "raw_response": raw,
                "code": code,
                "prediction": pred,
            })

            retry_count = 0

            while (
                    retry_count < args.max_retries
                    and (
                            str(pred).startswith("__CODE_ERROR__")
                            or str(pred).startswith("__TIMEOUT__")
                    )
            ):
                retry_count += 1

                fix_prompt = make_error_fix_prompt(
                    row=row,
                    df=df,
                    previous_code=code,
                    error=str(pred),
                )

                fixed_raw = llm.generate(fix_prompt)
                fixed_code = extract_answer_body(fixed_raw)
                fixed_pred = execute_answer_body(fixed_code, df)

                attempts.append({
                    "stage": f"fix_{retry_count}",
                    "raw_response": fixed_raw,
                    "code": fixed_code,
                    "prediction": fixed_pred,
                })

                code = fixed_code
                pred = fixed_pred

            if str(pred).startswith("__CODE_ERROR__") or str(pred).startswith("__TIMEOUT__"):
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

            attempts.append({
                "stage": "exception",
                "raw_response": raw,
                "code": None,
                "prediction": pred,
                "error": error,
            })

        predictions.append(pred)
        logs.append({
            "dataset": row["dataset"],
            "question": row["question"],
            "type": row.get("type"),
            "raw_response": raw,
            "extracted_code": code,
            "prediction": pred,
            "success": success,
            "error": error,
            "num_attempts": len(attempts),
            "attempts": attempts,
        })

    with open(output_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_dir / "predictions.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred).replace("\n", " ") + "\n")

    num_success = sum(1 for item in logs if item["success"])
    num_failed = len(logs) - num_success
    num_retried = sum(1 for item in logs if item.get("num_attempts", 1) > 1)

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
        "limit": args.limit,
        "lite": args.lite,
        "schema_mode": args.schema_mode,
        "total_examples": len(logs),
        "execution_success": num_success,
        "execution_failed": num_failed,
        "execution_success_rate": num_success / len(logs) if logs else 0,
        "retried_examples": num_retried,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {acc}")
    print(f"Execution success: {num_success}/{len(logs)}")
    print(f"Retried examples: {num_retried}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()