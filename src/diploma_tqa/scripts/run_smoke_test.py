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

from diploma_tqa.tools.tool_prompts import (
    make_tool_planning_prompt,
    make_react_tool_planning_prompt,
    make_post_code_react_action_prompt,
    make_post_code_react_decision_prompt,
    make_post_code_react_repair_prompt,
)

from diploma_tqa.tools.tool_runner import (
    parse_tool_calls,
    parse_react_plan,
    execute_tool_calls,
    format_react_plan,
    parse_json_object,
    execute_post_code_react_action,
)

def make_json_safe(obj):

    # convert numpy/pandas objects into Python objects

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
    # check whether model prediction is an error
    return isinstance(pred, str) and (
        pred.startswith("__CODE_ERROR__")
        or pred.startswith("__TIMEOUT__")
    )


def run_simple_post_code_react_loop(
    row,
    df,
    llm,
    code,
    pred,
    tool_observations,
    attempts,
    args,
):
    post_react_steps = []

    success = not is_execution_error(pred)
    error = None if success else str(pred)

    if args.post_code_react_mode != "simple-inspect" or not success:
        return code, pred, success, error, post_react_steps, ""

    # 1. Ask for one inspection action.
    action_prompt = make_post_code_react_action_prompt(
        row=row,
        df=df,
        generated_code=code,
        prediction=pred,
        tool_observations=tool_observations,
    )

    action_raw = llm.generate(action_prompt)
    action_result = parse_json_object(action_raw)
    action = action_result.get("action", {"name": "profile_used_columns", "args": {}})

    if not isinstance(action, dict):
        action = {"name": "profile_used_columns", "args": {}}

    action_name = action.get("name", "profile_used_columns")

    if action_name == "accept":
        post_react_steps.append({
            "stage": "post_code_react_action",
            "raw_response": action_raw,
            "action_result": action_result,
            "action": action,
            "observation": "",
            "decision_result": {"decision": "keep", "reason": "Action chose accept."},
        })
        return code, pred, success, error, post_react_steps, ""

    # 2. Execute the action.
    observation = execute_post_code_react_action(action=action, df=df, code=code)

    # 3. Decide keep vs rewrite from the observation.
    decision_prompt = make_post_code_react_decision_prompt(
        row=row,
        df=df,
        generated_code=code,
        prediction=pred,
        observation=observation,
    )

    decision_raw = llm.generate(decision_prompt)
    decision_result = parse_json_object(decision_raw)
    decision = decision_result.get("decision", "keep")

    step_log = {
        "stage": "post_code_react",
        "action_raw_response": action_raw,
        "action_result": action_result,
        "action": action,
        "observation": observation,
        "decision_raw_response": decision_raw,
        "decision_result": decision_result,
        "decision": decision,
    }

    # 4. Keep old answer unless rewrite is explicitly requested.
    if decision != "rewrite":
        post_react_steps.append(step_log)
        return code, pred, success, error, post_react_steps, observation

    # 5. Rewrite once.
    rewrite_instruction = decision_result.get("rewrite_instruction", "Rewrite the code using the observation.")

    repair_prompt = make_post_code_react_repair_prompt(
        row=row,
        df=df,
        previous_code=code,
        previous_prediction=pred,
        observation=observation,
        rewrite_instruction=rewrite_instruction,
    )

    repair_raw = llm.generate(repair_prompt)
    repair_code = extract_answer_body(repair_raw)
    repair_pred = execute_answer_body(repair_code, df)

    step_log.update({
        "repair_raw_response": repair_raw,
        "repair_code": repair_code,
        "repair_prediction": repair_pred,
        "repair_success": not is_execution_error(repair_pred),
    })

    post_react_steps.append(step_log)

    attempts.append({
        "stage": "post_code_react_rewrite",
        "raw_response": repair_raw,
        "code": repair_code,
        "prediction": repair_pred,
        "observation": observation,
        "rewrite_instruction": rewrite_instruction,
    })

    if not is_execution_error(repair_pred):
        return repair_code, repair_pred, True, None, post_react_steps, observation

    # Rewrite crashed: keep old executed prediction.
    step_log["kept_previous_prediction"] = True
    return code, pred, success, error, post_react_steps, observation

def tool_call_key(call):
    return json.dumps(call, sort_keys=True, ensure_ascii=False)


def run_semantic_critic_loop(
    row,
    df,
    llm,
    code,
    pred,
    tool_observations,
    attempts,
    args,
    react_max_tool_calls,
):
    semantic_react_steps = []
    post_code_observations_list = []

    success = not is_execution_error(pred)
    error = None if success else str(pred)

    if (
        args.post_code_react_mode != "semantic-critic"
        or not success
        or is_execution_error(pred)
    ):
        return code, pred, success, error, semantic_react_steps, ""

    semantic_repair_count = 0
    semantic_critic_max_steps = 2

    for semantic_step_idx in range(1, semantic_critic_max_steps + 1):
        try:
            critic_prompt = make_semantic_react_critic_prompt(
                question=row["question"],
                answer_type=row.get("type", "unknown"),
                columns=list(df.columns),
                dtypes={c: str(df[c].dtype) for c in df.columns},
                preview=df.head(5).to_string(index=False),
                tool_observations=tool_observations,
                post_code_observations="\n\n".join(post_code_observations_list),
                generated_code=code,
                prediction=pred,
                execution_error=None,
            )

            critic_raw = llm.generate(critic_prompt)

            critic_result = parse_semantic_critic(
                critic_raw,
                max_tool_calls=react_max_tool_calls,
            )

            decision = critic_result.get("decision", "accept")

            step_log = {
                "stage": f"semantic_critic_{semantic_step_idx}",
                "critic_raw": critic_raw,
                "critic_result": critic_result,
                "decision": decision,
                "accepted": decision == "accept",
                "tool_calls": [],
                "observation": "",
            }

            # Case 1: critic accepts current prediction.
            if decision == "accept":
                semantic_react_steps.append(step_log)
                break

            # Case 2: critic wants more evidence before deciding.
            if decision == "need_evidence":
                verification_calls = critic_result.get("verification_tool_calls", [])

                if verification_calls:
                    observation = execute_tool_calls(verification_calls, df)

                    post_code_observations_list.append(
                        f"Post-code semantic critic step {semantic_step_idx} observations:\n{observation}"
                    )

                    step_log["tool_calls"] = verification_calls
                    step_log["observation"] = observation
                    semantic_react_steps.append(step_log)

                    if semantic_step_idx < semantic_critic_max_steps:
                        continue

                # If final step still wants evidence, force a repair attempt.
                critic_result["decision"] = "repair"
                critic_result["accept"] = False
                decision = "repair"

                if not critic_result.get("repair_instruction"):
                    critic_result["repair_instruction"] = (
                        "The critic could not confidently accept the previous code. "
                        "Rewrite the code using the answer contract and all available observations. "
                        "Make sure the computation columns and return columns match the question."
                    )

                if not critic_result.get("must_return"):
                    critic_result["must_return"] = (
                        "Return exactly the requested answer value with the expected answer type."
                    )

                step_log["converted_need_evidence_to_repair"] = True
                # Do not append/break here; let execution fall through into the repair block.

            # Case 3: critic has enough evidence and requests repair.
            if decision == "repair":
                if semantic_repair_count >= args.post_code_react_max_retries:
                    step_log["repair_skipped"] = "Maximum semantic repair retries reached."
                    semantic_react_steps.append(step_log)
                    break

                if not critic_result.get("repair_instruction"):
                    step_log["repair_skipped"] = "No repair instruction provided."
                    semantic_react_steps.append(step_log)
                    break

                semantic_repair_count += 1

                combined_observations = tool_observations

                if post_code_observations_list:
                    combined_observations = (
                        f"{tool_observations}\n\n"
                        "Post-code verification observations:\n"
                        f"{chr(10).join(post_code_observations_list)}"
                    )

                semantic_fix_prompt = make_semantic_react_repair_prompt(
                    row=row,
                    df=df,
                    previous_code=code,
                    previous_prediction=pred,
                    critic_result=critic_result,
                    tool_observations=combined_observations,
                )

                semantic_raw = llm.generate(semantic_fix_prompt)
                semantic_code = extract_answer_body(semantic_raw)
                semantic_pred = execute_answer_body(semantic_code, df)

                step_log.update(
                    {
                        "repair_raw_response": semantic_raw,
                        "repair_code": semantic_code,
                        "repair_prediction": semantic_pred,
                        "repair_success": not is_execution_error(semantic_pred),
                    }
                )

                semantic_react_steps.append(step_log)

                attempts.append(
                    {
                        "stage": f"semantic_react_fix_{semantic_repair_count}",
                        "raw_response": semantic_raw,
                        "code": semantic_code,
                        "prediction": semantic_pred,
                        "critic_result": critic_result,
                        "post_code_observations": "\n\n".join(post_code_observations_list),
                    }
                )

                # Keep old successful prediction if repair crashes.
                if not is_execution_error(semantic_pred):
                    code = semantic_code
                    pred = semantic_pred
                    success = True
                    error = None
                    break

                step_log["kept_previous_prediction"] = True
                break

        except Exception as e:
            semantic_react_steps.append(
                {
                    "stage": f"semantic_critic_{semantic_step_idx}",
                    "error": str(e),
                }
            )
            break

    post_code_observations = "\n\n".join(post_code_observations_list)

    return code, pred, success, error, semantic_react_steps, post_code_observations


def main():
    parser = argparse.ArgumentParser()

    # Default settings
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--output-dir", default="results/smoke_test")

    # max execution retries
    parser.add_argument("--max-retries", type=int, default=0)

    # optional schema-hint mode
    parser.add_argument(
        "--schema-mode",
        choices=["none", "hint"],
        default="none",
    )

    # optional tool usage mode (modes: auto-schema, inspect)
    parser.add_argument(
        "--tool-mode",
        choices=["none", "auto-schema", "inspect", "react-inspect"],
        default="none",
    )

    parser.add_argument(
        "--indices-file",
        default=None,
        help="Optional text file containing 0-based example indices to evaluate, one per line.",
    )

    parser.add_argument(
        "--react-observation-mode",
        choices=["raw", "plan", "plan_plus_raw"],
        default="plan",
        help="Controls what react-inspect passes to the code-generation prompt.",
    )

    parser.add_argument(
        "--post-code-react-max-retries",
        type=int,
        default=1,
        help="Maximum number of semantic critic repair attempts after successful execution.",
    )

    parser.add_argument(
        "--post-code-react-mode",
        choices=["none", "semantic-critic", "simple-inspect"],
        default="none",
        help="Optional post-execution ReAct. simple-inspect runs one inspection action then optionally rewrites once.",
    )

    args = parser.parse_args()

    REACT_MAX_STEPS = 2
    REACT_MAX_TOOL_CALLS = 3
    INSPECT_MAX_TOOL_CALLS = 3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa = load_qa(name="semeval", split="test", limit=args.limit)

    # use indices file to run selected examples - if indices_file argument is not empty
    if args.indices_file is not None:
        with open(args.indices_file, "r", encoding="utf-8") as f:
            selected_indices = [
                int(line.strip())
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        full_qa = load_qa(name="semeval", split="test", limit=None)

        if hasattr(full_qa, "select"):
            qa = full_qa.select(selected_indices)
        else:
            qa = [full_qa[i] for i in selected_indices]
    else:
        selected_indices = None

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

    # for i, row in enumerate(progress, start=1):
    for local_i, row in enumerate(progress, start=1):
        original_index = (
            selected_indices[local_i - 1]
            if selected_indices is not None
            else local_i - 1
        )
        df = load_table(row["dataset"], lite=args.lite)

        tool_raw = None
        tool_calls = []
        tool_observations = ""
        react_steps = []
        latest_plan = {}

        semantic_react_steps = []

        # optional tool stage
        if args.tool_mode == "auto-schema":
            # search for relevant columns using the question - automatically
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
            # ask the model if it wants to use any of the tools
            try:
                tool_prompt = make_tool_planning_prompt(
                    row=row,
                    df=df,
                    max_tool_calls=INSPECT_MAX_TOOL_CALLS,
                )

                tool_raw = llm.generate(tool_prompt)
                tool_calls = parse_tool_calls(tool_raw, max_tool_calls=INSPECT_MAX_TOOL_CALLS)
                tool_observations = execute_tool_calls(tool_calls, df)

            except Exception as e:
                tool_observations = f"Tool planning failed: {e}"


        elif args.tool_mode == "react-inspect":

            # iterative tool planning: plan -> act -> observe -> update compact plan

            react_steps = []

            all_tool_calls = []

            all_observations = []

            seen_tool_calls = set()

            latest_plan = {}

            semantic_react_steps = []

            try:

                for step in range(1, REACT_MAX_STEPS + 1):

                    previous_observations = (

                        "\n\n".join(all_observations)

                        if all_observations

                        else ""

                    )

                    tool_prompt = make_react_tool_planning_prompt(

                        row=row,

                        df=df,

                        previous_observations=previous_observations,

                        step=step,

                        max_steps=REACT_MAX_STEPS,

                        max_tool_calls=REACT_MAX_TOOL_CALLS,

                    )

                    raw_plan = llm.generate(tool_prompt)

                    plan = parse_react_plan(

                        raw_plan,

                        max_tool_calls=REACT_MAX_TOOL_CALLS,

                    )

                    thought = plan.get("thought", "")

                    current_plan = plan.get("current_plan", {})

                    if current_plan:
                        latest_plan = current_plan

                    step_calls = plan.get("tool_calls", [])

                    # Remove repeated calls within the same example.

                    new_calls = []

                    for call in step_calls:

                        key = tool_call_key(call)

                        if key not in seen_tool_calls:
                            seen_tool_calls.add(key)

                            new_calls.append(call)

                    step_calls = new_calls

                    if not step_calls:
                        react_steps.append(

                            {

                                "step": step,

                                "thought": thought,

                                "raw_response": raw_plan,

                                "current_plan": current_plan,

                                "tool_calls": [],

                                "observation": "",

                                "stop": True,

                            }

                        )

                        break

                    observation = execute_tool_calls(step_calls, df)

                    all_tool_calls.extend(step_calls)

                    all_observations.append(

                        f"Step {step} observations:\n{observation}"

                    )

                    react_steps.append(

                        {

                            "step": step,

                            "thought": thought,

                            "raw_response": raw_plan,

                            "current_plan": current_plan,

                            "tool_calls": step_calls,

                            "observation": observation,

                            "stop": bool(plan.get("stop", False)),

                        }

                    )

                    # If the model explicitly says stop after these calls,

                    # allow the next iteration only if there are remaining steps?

                    # For now, respect stop after saving the observation.

                    if plan.get("stop", False):
                        break

                compact_plan = format_react_plan(latest_plan)

                raw_observations = "\n\n".join(all_observations)

                if args.react_observation_mode == "raw":

                    tool_observations = raw_observations


                elif args.react_observation_mode == "plan":

                    tool_observations = compact_plan


                elif args.react_observation_mode == "plan_plus_raw":

                    if compact_plan and raw_observations:

                        tool_observations = (

                                compact_plan

                                + "\n\nSupporting observations:\n"

                                + raw_observations

                        )

                    else:

                        tool_observations = compact_plan or raw_observations

                tool_raw = json.dumps(react_steps, ensure_ascii=False)

                tool_calls = all_tool_calls


            except Exception as e:

                tool_observations = f"ReAct tool planning failed: {e}"

                tool_raw = None

                tool_calls = []

                react_steps = []

        # initial code generation and main prompt creation
        prompt = make_baseline_prompt(
            row=row,
            df=df,
            schema_mode=args.schema_mode,
            tool_observations=tool_observations,
        )

        raw = llm.generate(prompt)
        attempts = []

        try:
            # extract code from prediction and execute it
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

            # error fixing loop
            retry_count = 0

            while retry_count < args.max_retries and is_execution_error(pred):
                retry_count += 1

                fix_prompt = make_error_fix_prompt(
                    row=row,
                    df=df,
                    previous_code=code,
                    error=str(pred),
                    #tool_observations=tool_observations,  # with 8B model: adds noise - keep for use with larger models
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
                post_code_observations = ""
            else:
                success = True
                error = None

                if args.post_code_react_mode == "simple-inspect":
                    (
                        code,
                        pred,
                        success,
                        error,
                        post_code_react_steps,
                        post_code_observations,
                    ) = run_simple_post_code_react_loop(
                        row=row,
                        df=df,
                        llm=llm,
                        code=code,
                        pred=pred,
                        tool_observations=tool_observations,
                        attempts=attempts,
                        args=args,
                    )
                    semantic_react_steps = post_code_react_steps
                else:
                    (
                        code,
                        pred,
                        success,
                        error,
                        semantic_react_steps,
                        post_code_observations,
                    ) = run_semantic_critic_loop(
                        row=row,
                        df=df,
                        llm=llm,
                        code=code,
                        pred=pred,
                        tool_observations=tool_observations,
                        attempts=attempts,
                        args=args,
                        react_max_tool_calls=REACT_MAX_TOOL_CALLS,
                    )


        except Exception as e:

            code = None

            pred = f"__CODE_ERROR__: {e}"

            error = str(e)

            success = False

            post_code_observations = ""

            attempts.append(

                {

                    "stage": "exception",

                    "raw_response": raw,

                    "code": None,

                    "prediction": pred,

                    "error": error,

                }

            )

        # stores prediction and detailed log for every example
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
            "react_steps": react_steps if args.tool_mode == "react-inspect" else None,
            "num_react_steps": len(react_steps) if args.tool_mode == "react-inspect" else 0,
            "num_tool_calls": len(tool_calls),

            "react_observation_mode": (
                args.react_observation_mode if args.tool_mode == "react-inspect" else None
            ),
            "react_final_plan": latest_plan if args.tool_mode == "react-inspect" else None,

            "post_code_react_mode": args.post_code_react_mode,
            "post_code_react_max_retries": args.post_code_react_max_retries,
            "semantic_react_steps": semantic_react_steps,
            "num_semantic_react_steps": len(semantic_react_steps),
            "post_code_observations": post_code_observations,

            "raw_response": raw,
            "extracted_code": code,
            "prediction": pred,
            "success": success,
            "error": error,
            "num_attempts": len(attempts),
            "attempts": attempts,

            "example_index": original_index,
        }

        logs.append(log_item)

        if success:
            running_success += 1

        if len(attempts) > 1:
            running_retried += 1

        progress.set_postfix(
            {
                "success": f"{running_success}/{local_i}",
                "retried": running_retried,
                "dataset": row["dataset"],
            }
        )

    # save logs and prediction list
    with open(output_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(make_json_safe(item), ensure_ascii=False) + "\n")

    with open(output_dir / "predictions.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred).replace("\n", " ") + "\n")

    # compute execution stats
    num_success = sum(1 for item in logs if item["success"])
    num_failed = len(logs) - num_success
    num_retried = sum(1 for item in logs if item.get("num_attempts", 1) > 1)

    attempt_counts = {}
    for item in logs:
        n = item.get("num_attempts", 1)
        attempt_counts[str(n)] = attempt_counts.get(str(n), 0) + 1

    # approximates total model calls
    total_model_calls = sum(item.get("num_attempts", 1) for item in logs)

    if args.tool_mode == "inspect":
        total_model_calls += len(logs)

    elif args.tool_mode == "react-inspect":
        total_model_calls += sum(
            item.get("num_react_steps", 0)
            for item in logs
        )

    if args.post_code_react_mode == "semantic-critic":
        for item in logs:
            for step in item.get("semantic_react_steps", []):
                # one LLM call for the critic/verifier
                total_model_calls += 1

                # one extra LLM call if repair code was generated
                if step.get("repair_raw_response") is not None:
                    total_model_calls += 1


    # evaluate predictions using the official task evaluator
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
        "react_observation_mode": (
            args.react_observation_mode if args.tool_mode == "react-inspect" else None
        ),

        "max_retries": args.max_retries,
        "error_fixing_enabled": args.max_retries > 0,

        "total_examples": len(logs),
        "execution_success": num_success,
        "execution_failed": num_failed,
        "execution_success_rate": num_success / len(logs) if logs else 0,

        "react_max_steps": REACT_MAX_STEPS if args.tool_mode == "react-inspect" else None,
        "react_max_tool_calls": REACT_MAX_TOOL_CALLS if args.tool_mode == "react-inspect" else None,
        "avg_react_steps": (
            sum(item.get("num_react_steps", 0) for item in logs) / len(logs)
            if args.tool_mode == "react-inspect" and logs
            else None
        ),
        "avg_tool_calls": (
            sum(item.get("num_tool_calls", 0) for item in logs) / len(logs)
            if logs
            else 0
        ),

        "post_code_react_mode": args.post_code_react_mode,
        "post_code_react_max_retries": args.post_code_react_max_retries,
        "avg_semantic_react_steps": (
            sum(item.get("num_semantic_react_steps", 0) for item in logs) / len(logs)
            if logs
            else 0
        ),
        "semantic_react_repaired_examples": sum(
    1
    for item in logs
    for step in item.get("semantic_react_steps", [])
    if step.get("repair_raw_response") is not None
    ),

    "semantic_react_accept_examples": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "accept"
    ),

    "semantic_react_evidence_examples": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "need_evidence"
    ),

    "semantic_react_repair_decisions": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") == "repair"
    ),

    "semantic_react_non_accept_decisions": sum(
        1
        for item in logs
        for step in item.get("semantic_react_steps", [])
        if step.get("decision") in {"need_evidence", "repair"}
    ),

        "indices_file": args.indices_file,
        "selected_indices": selected_indices,

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