import ast
import json
import random
import re
from pathlib import Path
from datasets import load_dataset

SEED = 42
random.seed(SEED)


def parse_hermes_tool_calls(assistant_text: str) -> tuple[str, list[dict]]:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant_text, re.DOTALL)
    if not matches:
        return assistant_text.strip(), []

    tool_calls = []
    for m in matches:
        obj = None
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(m.strip())
                break
            except Exception:
                continue
        if not isinstance(obj, dict):
            raise ValueError("Unparseable tool call")
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


def main():
    samples = []
    for config in ["func_calling", "func_calling_singleturn"]:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", config, split="train").shuffle(seed=SEED)
        for row in ds:
            try:
                raw_tools = row.get("tools")
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
                if not tools or not isinstance(tools, list):
                    continue

                convos = row["conversations"]
                if not convos or len(convos) < 2:
                    continue

                messages = []
                valid = True

                for turn in convos:
                    role = turn["from"]
                    val = turn["value"].strip()

                    if role == "system":
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
                        clean_resp = re.sub(r"</?tool_response>", "", val).strip()
                        messages.append({"role": "tool", "content": clean_resp})

                if not valid or len(messages) < 3 or messages[-1]["role"] != "assistant":
                    continue

                samples.append({
                    "category": f"hermes_{config}",
                    "tools": tools,
                    "messages": messages,
                })
                if len(samples) >= 1450:
                    break
            except Exception:
                continue
        if len(samples) >= 1450:
            break

    print(f"Recovered {len(samples)} verified Hermes multi-turn/abstention samples!")

    train_path = Path("data/train.jsonl")
    eval_path = Path("data/eval.jsonl")

    existing_train = [json.loads(line) for line in train_path.open(encoding="utf-8")]
    existing_eval = [json.loads(line) for line in eval_path.open(encoding="utf-8")]

    random.shuffle(samples)
    new_eval = samples[:50]
    new_train = samples[50:]

    full_train = existing_train + new_train
    full_eval = existing_eval + new_eval
    random.shuffle(full_train)

    with train_path.open("w", encoding="utf-8") as f:
        for row in full_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for row in full_eval:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated dataset totals -> Train: {len(full_train)} | Eval: {len(full_eval)}")


if __name__ == "__main__":
    main()
