import json
import re
import time
from pathlib import Path
import jsonschema
import ollama

EVAL_PATH = Path("data/eval.jsonl")
RESULTS_PATH = Path("eval/results.json")
MODELS = ["qwen2.5:7b", "actuator-7b"]


def extract_target_schema(system_content: str) -> dict | None:
    match = re.search(r"Target JSON Schema:\s*(\{.*\})", system_content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def normalize_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return args
    return args


def compare_tool_calls(expected_calls: list[dict], actual_calls: list[dict]) -> tuple[bool, bool]:
    """Returns (valid_structure, exact_match)."""
    if not actual_calls:
        return False, False

    parsed_actual = []
    for tc in actual_calls:
        fn = tc.get("function", {})
        name = fn.get("name")
        args = normalize_args(fn.get("arguments", {}))
        if not name or not isinstance(args, dict):
            return False, False
        parsed_actual.append((name, args))

    parsed_expected = []
    for tc in expected_calls:
        fn = tc.get("function", {})
        parsed_expected.append((fn.get("name"), normalize_args(fn.get("arguments", {}))))

    # Order-insensitive comparison for parallel calls
    exp_sorted = sorted(parsed_expected, key=lambda x: (x[0], json.dumps(x[1], sort_keys=True)))
    act_sorted = sorted(parsed_actual, key=lambda x: (x[0], json.dumps(x[1], sort_keys=True)))
    return True, exp_sorted == act_sorted


def evaluate_model(model_name: str, eval_rows: list[dict]) -> dict:
    print(f"\nEvaluating {model_name} across {len(eval_rows)} held-out tasks...")
    stats = {
        "tool_total": 0,
        "tool_valid_json": 0,
        "tool_exact_match": 0,
        "abstain_total": 0,
        "abstain_correct": 0,
        "schema_total": 0,
        "schema_raw_json": 0,
        "schema_valid_match": 0,
        "latencies_ms": [],
    }

    for idx, row in enumerate(eval_rows, 1):
        cat = row["category"]
        tools = row.get("tools") or None
        messages = row["messages"]

        # Prompt is all turns up to the final expected assistant turn
        prompt_messages = messages[:-1]
        expected_msg = messages[-1]

        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model=model_name,
                messages=prompt_messages,
                tools=tools,
                options={"temperature": 0.0, "num_predict": 1024},
            )
            msg = resp.get("message", {})
        except Exception as e:
            msg = {"content": "", "tool_calls": []}
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats["latencies_ms"].append(elapsed_ms)

        actual_content = (msg.get("content") or "").strip()
        actual_tools = msg.get("tool_calls") or []

        if cat == "strict_json_schema":
            stats["schema_total"] += 1
            schema = extract_target_schema(prompt_messages[0]["content"])
            # Check raw JSON without markdown fences
            if not actual_content.startswith("```"):
                try:
                    parsed_obj = json.loads(actual_content)
                    stats["schema_raw_json"] += 1
                    if schema is not None:
                        jsonschema.validate(instance=parsed_obj, schema=schema)
                        stats["schema_valid_match"] += 1
                except Exception:
                    pass
        else:
            expected_tools = expected_msg.get("tool_calls") or []
            if expected_tools:
                stats["tool_total"] += 1
                valid_struct, exact = compare_tool_calls(expected_tools, actual_tools)
                if valid_struct:
                    stats["tool_valid_json"] += 1
                if exact:
                    stats["tool_exact_match"] += 1
            else:
                # Abstention / natural response turn (no tool call expected)
                stats["abstain_total"] += 1
                if not actual_tools and "<tool_call>" not in actual_content and len(actual_content) > 0:
                    stats["abstain_correct"] += 1

        if idx % 30 == 0 or idx == len(eval_rows):
            print(f"  [{model_name}] Completed {idx}/{len(eval_rows)}...")

    def pct(num, den):
        return round((num / den) * 100, 2) if den else 0.0

    avg_latency = round(sum(stats["latencies_ms"]) / len(stats["latencies_ms"]), 1)
    return {
        "model": model_name,
        "valid_tool_syntax_pct": pct(stats["tool_valid_json"], stats["tool_total"]),
        "exact_tool_arg_match_pct": pct(stats["tool_exact_match"], stats["tool_total"]),
        "abstention_accuracy_pct": pct(stats["abstain_correct"], stats["abstain_total"]),
        "raw_json_no_fence_pct": pct(stats["schema_raw_json"], stats["schema_total"]),
        "strict_jsonschema_pass_pct": pct(stats["schema_valid_match"], stats["schema_total"]),
        "avg_latency_ms": avg_latency,
        "counts": {
            "tool_tasks": stats["tool_total"],
            "abstain_tasks": stats["abstain_total"],
            "schema_tasks": stats["schema_total"],
        },
    }


def main():
    eval_rows = [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8")]
    all_results = {}

    for m in MODELS:
        all_results[m] = evaluate_model(m, eval_rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n================ BENCHMARK SUMMARY ================")
    for m, r in all_results.items():
        print(f"\nModel: {m}")
        print(f"  Valid Tool-Call JSON Syntax : {r['valid_tool_syntax_pct']}%")
        print(f"  Exact Function & Arg Match  : {r['exact_tool_arg_match_pct']}%")
        print(f"  Abstention Accuracy         : {r['abstention_accuracy_pct']}%")
        print(f"  Raw JSON (No Markdown Fence): {r['raw_json_no_fence_pct']}%")
        print(f"  Strict jsonschema Compliance: {r['strict_jsonschema_pass_pct']}%")
        print(f"  Avg Latency / Request       : {r['avg_latency_ms']} ms")


if __name__ == "__main__":
    main()
