import argparse
import json
from pathlib import Path

from databench_eval import Evaluator
from diploma_tqa.data.databench_loader import load_qa, load_table
from diploma_tqa.llms.ollama_client import OllamaClient
from diploma_tqa.prompts.baseline_prompts import make_baseline_prompt
from diploma_tqa.execution.code_extract import extract_answer_body
from diploma_tqa.execution.pandas_executor import execute_answer_body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--output-dir", default="results/smoke_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa = load_qa(name="semeval", split="dev", limit=args.limit)
    llm = OllamaClient(model=args.model)

    predictions = []
    logs = []

    for row in qa:
        df = load_table(row["dataset"], lite=args.lite)
        prompt = make_baseline_prompt(row, df)

        raw = llm.generate(prompt)

        try:
            code = extract_answer_body(raw)
            pred = execute_answer_body(code, df)
            error = None
            success = not (
                    str(pred).startswith("__CODE_ERROR__")
                    or str(pred).startswith("__TIMEOUT__")
            )
        except Exception as e:
            code = None
            pred = f"__CODE_ERROR__: {e}"
            error = str(e)
            success = False

        predictions.append(pred)
        logs.append({
            "dataset": row["dataset"],
            "question": row["question"],
            "type": row.get("type"),
            "raw_response": raw,
            "extracted_expression": code,
            "prediction": pred,
            "success": success,
            "error": error,
        })

    with open(output_dir / "logs.jsonl", "w") as f:
        for item in logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_dir / "predictions.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred).replace("\n", " ") + "\n")

    evaluator = Evaluator(qa=qa)
    acc = evaluator.eval(predictions, lite=args.lite)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": acc, "limit": args.limit, "lite": args.lite}, f, indent=2)

    print(f"Accuracy: {acc:.3f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()