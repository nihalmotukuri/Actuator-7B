import json
import random
import re
from pathlib import Path
from datasets import load_dataset
import jsonschema

SEED = 42
random.seed(SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_xlam_tool(tool: dict) -> dict:
    """Convert xLAM tool schema into standard OpenAI/Ollama tool definition."""
    raw_params = tool.get("parameters", {})
    properties = {}
    required = []

    if isinstance(raw_params, dict):
        for param_name, meta in raw_params.items():
            if not isinstance(meta, dict):
                continue
            p_type = str(meta.get("type", "string")).lower()
            type_map = {
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }
            json_type = type_map.get(p_type, "string")
            prop_entry = {
                "type": json_type,
                "description": str(meta.get("description", "")).strip(),
            }
            if json_type == "array":
                prop_entry["items"] = {"type": "string"}
            properties[param_name] = prop_entry
            if meta.get("required", False) is True:
                required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": str(tool.get("description", "")).strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_xlam_samples(target_count: int = 2300) -> list[dict]:
    print("Loading Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    ds = ds.shuffle(seed=SEED)

    samples = []
    for row in ds:
        try:
            tools_raw = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
            answers_raw = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
            query = str(row["query"]).strip()

            if not query or not tools_raw or not answers_raw:
                continue

            valid_tool_names = {t["name"] for t in tools_raw if "name" in t}
            if not valid_tool_names:
                continue

            # Ensure every called function exists in provided tools and has valid JSON args
            tool_calls = []
            valid = True
            for ans in answers_raw:
                fn_name = ans.get("name")
                fn_args = ans.get("arguments", {})
                if fn_name not in valid_tool_names or not isinstance(fn_args, dict):
                    valid = False
                    break
                tool_calls.append({
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                })

            if not valid or not tool_calls:
                continue

            norm_tools = [normalize_xlam_tool(t) for t in tools_raw]
            samples.append({
                "category": "xlam_tool_call",
                "tools": norm_tools,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Actuator-7B, a deterministic tool-calling and structured-data engine. Call the exact required functions with strict schema-compliant arguments, or respond concisely when no tool is needed.",
                    },
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "", "tool_calls": tool_calls},
                ],
            })

            if len(samples) >= target_count:
                break
        except Exception:
            continue

    print(f"Collected {len(samples)} verified xLAM samples.")
    return samples


def extract_hermes_tools(system_prompt: str) -> list[dict]:
    """Extract JSON tool definitions embedded inside Hermes <tools> XML tags."""
    match = re.search(r"<tools>\s*(.*?)\s*</tools>", system_prompt, re.DOTALL)
    if not match:
        return []
    block = match.group(1).strip()
    tools = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "function" in obj:
                tools.append(obj)
            elif "name" in obj:
                tools.append({"type": "function", "function": obj})
        except json.JSONDecodeError:
            continue
    return tools


def parse_hermes_tool_calls(assistant_text: str) -> tuple[str, list[dict]]:
    """Convert <tool_call>{...}</tool_call> tags into structured OpenAI/Ollama tool_calls."""
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant_text, re.DOTALL)
    if not matches:
        return assistant_text.strip(), []

    tool_calls = []
    for m in matches:
        obj = json.loads(m)
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not name or not isinstance(args, dict):
            raise ValueError("Invalid tool call structure")
        tool_calls.append({
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    cleaned_text = re.sub(r"<tool_call>.*?</tool_call>", "", assistant_text, flags=re.DOTALL).strip()
    return cleaned_text, tool_calls


def build_hermes_multiturn_samples(target_count: int = 1450) -> list[dict]:
    print("Loading NousResearch/hermes-function-calling-v1 (func_calling & singleturn)...")
    samples = []
    for config in ["func_calling", "func_calling_singleturn"]:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", config, split="train")
        ds = ds.shuffle(seed=SEED)

        for row in ds:
            try:
                convos = row["conversations"]
                if not convos or len(convos) < 2:
                    continue

                tools = []
                messages = []
                valid = True

                for turn in convos:
                    role = turn["from"]
                    val = turn["value"].strip()

                    if role == "system":
                        tools = extract_hermes_tools(val)
                        messages.append({
                            "role": "system",
                            "content": "You are Actuator-7B, a deterministic tool-calling and structured-data engine. Call the exact required functions with strict schema-compliant arguments, or respond concisely when no tool is needed.",
                        })
                    elif role == "human":
                        messages.append({"role": "user", "content": val})
                    elif role == "gpt":
                        clean_txt, t_calls = parse_hermes_tool_calls(val)
                        if t_calls:
                            messages.append({"role": "assistant", "content": clean_txt, "tool_calls": t_calls})
                        else:
                            if not clean_txt:
                                valid = False
                                break
                            messages.append({"role": "assistant", "content": clean_txt})
                    elif role == "tool":
                        # Strip <tool_response> wrapper if present
                        clean_resp = re.sub(r"</?tool_response>", "", val).strip()
                        messages.append({"role": "tool", "content": clean_resp})

                if not valid or not tools or len(messages) < 3:
                    continue
                if messages[-1]["role"] != "assistant":
                    continue

                samples.append({
                    "category": f"hermes_{config}",
                    "tools": tools,
                    "messages": messages,
                })
                if len(samples) >= target_count:
                    break
            except Exception:
                continue
        if len(samples) >= target_count:
            break

    print(f"Collected {len(samples)} verified Hermes multi-turn/abstention samples.")
    return samples


def build_strict_json_samples(target_count: int = 1200) -> list[dict]:
    print("Loading NousResearch/hermes-function-calling-v1 (json_mode)...")
    samples = []
    for config in ["json_mode_singleturn", "json_mode_agentic"]:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", config, split="train")
        ds = ds.shuffle(seed=SEED)

        for row in ds:
            try:
                convos = row["conversations"]
                if len(convos) < 3:
                    continue

                sys_val = convos[0]["value"]
                # Extract JSON schema from system prompt
                schema_match = re.search(r"(\{.*\})", sys_val, re.DOTALL)
                if not schema_match:
                    continue
                schema_obj = json.loads(schema_match.group(1))

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Actuator-7B in Strict JSON Schema Mode. Output ONLY valid, raw JSON "
                            "matching the requested schema with zero markdown fences, zero commentary, and exact data types.\n\n"
                            f"Target JSON Schema:\n{json.dumps(schema_obj)}"
                        ),
                    }
                ]

                valid = True
                for turn in convos[1:]:
                    role = turn["from"]
                    val = turn["value"].strip()
                    if role == "human":
                        messages.append({"role": "user", "content": val})
                    elif role == "gpt":
                        # Strip accidental markdown fences if present in source dataset
                        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", val, flags=re.IGNORECASE).strip()
                        parsed_json = json.loads(cleaned)
                        # Strictly validate against the schema!
                        jsonschema.validate(instance=parsed_json, schema=schema_obj)
                        messages.append({"role": "assistant", "content": json.dumps(parsed_json, ensure_ascii=False)})

                if not valid or len(messages) < 3 or messages[-1]["role"] != "assistant":
                    continue

                samples.append({
                    "category": "strict_json_schema",
                    "tools": [],
                    "messages": messages,
                })
                if len(samples) >= target_count:
                    break
            except Exception:
                continue
        if len(samples) >= target_count:
            break

    print(f"Collected {len(samples)} jsonschema-validated Strict JSON samples.")
    return samples


def main():
    xlam = build_xlam_samples(target_count=2300)
    hermes = build_hermes_multiturn_samples(target_count=1450)
    strict_json = build_strict_json_samples(target_count=1200)

    all_samples = xlam + hermes + strict_json
    random.shuffle(all_samples)

    # Hold out 150 balanced samples for evaluation (50 per category)
    eval_set = []
    train_set = []
    eval_quotas = {"xlam_tool_call": 50, "hermes": 50, "strict_json_schema": 50}

    for item in all_samples:
        cat = item["category"]
        if cat == "xlam_tool_call" and eval_quotas["xlam_tool_call"] > 0:
            eval_set.append(item)
            eval_quotas["xlam_tool_call"] -= 1
        elif cat.startswith("hermes_") and eval_quotas["hermes"] > 0:
            eval_set.append(item)
            eval_quotas["hermes"] -= 1
        elif cat == "strict_json_schema" and eval_quotas["strict_json_schema"] > 0:
            eval_set.append(item)
            eval_quotas["strict_json_schema"] -= 1
        else:
            train_set.append(item)

    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for row in train_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(train_set)} training samples -> {train_path}")
    print(f"Saved {len(eval_set)} evaluation samples -> {eval_path}")


if __name__ == "__main__":
    main()
