import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_error_prediction(pred) -> bool:
    return isinstance(pred, str) and (
        pred.startswith("__CODE_ERROR__")
        or pred.startswith("__TIMEOUT__")
    )


def short_error(error: str | None) -> str:
    if not error:
        return "None"

    error = str(error)

    # Normalize common KeyError-style messages.
    if error.startswith("__CODE_ERROR__: "):
        error = error.replace("__CODE_ERROR__: ", "", 1)

    return error[:160]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        help="Path to an experiment output directory containing logs.jsonl and metrics.json",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many remaining failures to print",
    )
    parser.add_argument(
        "--show-fixed",
        type=int,
        default=10,
        help="How many fixed-by-retry examples to print",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_path = output_dir / "logs.jsonl"
    metrics_path = output_dir / "metrics.json"

    logs = load_jsonl(logs_path)
    metrics = load_metrics(metrics_path)

    total = len(logs)
    success_count = sum(1 for item in logs if item.get("success"))
    failed_count = total - success_count

    attempt_counts = Counter(item.get("num_attempts", 1) for item in logs)
    dataset_counts = Counter(item.get("dataset", "UNKNOWN") for item in logs)
    dataset_failures = Counter(
        item.get("dataset", "UNKNOWN") for item in logs if not item.get("success")
    )
    type_failures = Counter(
        item.get("type", "UNKNOWN") for item in logs if not item.get("success")
    )

    retried = [item for item in logs if item.get("num_attempts", 1) > 1]

    fixed_by_retry = []
    still_failed_after_retry = []

    for item in retried:
        attempts = item.get("attempts", [])
        if not attempts:
            continue

        first_pred = attempts[0].get("prediction")
        final_pred = item.get("prediction")

        initially_failed = is_error_prediction(first_pred)
        finally_succeeded = not is_error_prediction(final_pred) and item.get("success")

        if initially_failed and finally_succeeded:
            fixed_by_retry.append(item)
        elif initially_failed and not item.get("success"):
            still_failed_after_retry.append(item)

    error_counts = Counter(
        short_error(item.get("error")) for item in logs if not item.get("success")
    )

    print("=" * 80)
    print("Experiment:", output_dir)
    print("=" * 80)

    print("\nMETRICS")
    print("-" * 80)
    print("Accuracy:", metrics.get("accuracy"))
    print("Evaluation error:", metrics.get("evaluation_error"))
    print("Total examples:", total)
    print(f"Execution success: {success_count}/{total}")
    print(f"Execution success rate: {success_count / total:.3f}" if total else "N/A")
    print("Retried examples:", len(retried))
    print("Fixed by retry:", len(fixed_by_retry))
    print("Still failed after retry:", len(still_failed_after_retry))

    print("\nATTEMPT DISTRIBUTION")
    print("-" * 80)
    for attempts, count in sorted(attempt_counts.items()):
        print(f"{attempts} attempt(s): {count}")

    print("\nDATASETS")
    print("-" * 80)
    for dataset, count in sorted(dataset_counts.items()):
        failures = dataset_failures.get(dataset, 0)
        print(f"{dataset}: {count} examples, {failures} failures")

    print("\nFAILURES BY ANSWER TYPE")
    print("-" * 80)
    for answer_type, count in type_failures.most_common():
        print(f"{answer_type}: {count}")

    print("\nERROR MESSAGES")
    print("-" * 80)
    for error, count in error_counts.most_common():
        print(f"{count}x | {error}")

    print("\nFIXED BY RETRY EXAMPLES")
    print("-" * 80)
    for item in fixed_by_retry[: args.show_fixed]:
        attempts = item.get("attempts", [])
        first_error = attempts[0].get("prediction") if attempts else None
        print(f"Dataset: {item.get('dataset')}")
        print(f"Type: {item.get('type')}")
        print(f"Question: {item.get('question')}")
        print(f"Initial error: {first_error}")
        print(f"Final prediction: {item.get('prediction')}")
        print(f"Attempts: {item.get('num_attempts')}")
        print("-" * 80)

    print("\nREMAINING FAILURES")
    print("-" * 80)
    for item in [x for x in logs if not x.get("success")][: args.show_failures]:
        print(f"Dataset: {item.get('dataset')}")
        print(f"Type: {item.get('type')}")
        print(f"Question: {item.get('question')}")
        print(f"Error: {item.get('error')}")
        print(f"Attempts: {item.get('num_attempts')}")
        attempts = item.get("attempts", [])
        if attempts:
            print("Last code:")
            print(attempts[-1].get("code"))
        print("-" * 80)


if __name__ == "__main__":
    main()