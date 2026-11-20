# Actuator-7B

**Turn messy human requests into mistake-free tool calls and clean JSON—fast, free, and completely private with zero cloud API bills.**

[![Hugging Face GGUF](https://img.shields.io/badge/Hugging%20Face-nihalmotukuri%2Factuator--7b--gguf-yellow?logo=huggingface)](https://huggingface.co/nihalmotukuri/actuator-7b-gguf)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Fine-Tuning](https://img.shields.io/badge/QLoRA-Unsloth-green)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

---

## Why Actuator-7B?

Small local language models (7B–8B) frequently break automated pipelines when used as agent backends:
- **Chatty Syndrome:** Wrapping structured JSON inside conversational prose or markdown fences (` ```json `), crashing downstream `json.loads()` and Pydantic parsers.
- **Hallucinated Arguments:** Inventing parameter keys, passing string types instead of integers/booleans, or dropping `required` fields.
- **Multi-Turn & Error-Recovery Failure:** Apologizing to the user when a tool returns an error instead of silently correcting arguments and retrying.
- **Trigger-Happy Tool Calls:** Invoking irrelevant functions when a standard text response is appropriate.

**Actuator-7B** eliminates conversational filler and acts as a reliable local dispatcher between natural language and strict software APIs—compatible out of the box with **Ollama**, **LM Studio**, **vLLM**, **n8n**, **Cline / Roo Code**, **OpenWebUI**, and **MCP servers** via Qwen native `<tool_call>` chat template.

---

## Benchmark Evaluation (150 Held-Out Tasks)

Evaluated across 150 held-out prompts split evenly across single/parallel tool calls (50), multi-turn & abstention tasks (50), and `jsonschema`-validated structured extraction (50):

| Metric | Stock `Qwen2.5-7B-Instruct` | **`Actuator-7B` (`Q4_K_M`)** | Delta |
| :--- | :---: | :---: | :---: |
| **Valid Tool-Call JSON Syntax** | 100.0% | **100.0%** | — |
| **Exact Function & Argument Match** | 53.73% | **76.12%** | **+22.39%** |
| **Abstention / Irrelevance Accuracy** | 84.85% | **100.0%** | **+15.15%** |
| **Raw JSON (Zero Markdown Fences)** | 96.00% | **100.0%** | **+4.00%** |
| **Strict `jsonschema` Compliance Rate** | 96.00% | **100.0%** | **+4.00%** |

---

## Quickstart

### 1. Run with Ollama

Pull and run directly from Hugging Face:

```bash
ollama run hf.co/nihalmotukuri/actuator-7b-gguf:Q4_K_M
```

### 2. Native Tool-Calling Example (Python + Ollama)

```python
import ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    }
]

response = ollama.chat(
    model="hf.co/nihalmotukuri/actuator-7b-gguf:Q4_K_M",
    messages=[{"role": "user", "content": "Find flights from Hyderabad to Delhi on 2026-11-25."}],
    tools=tools,
)

print(response["message"]["tool_calls"])
```

### 3. Strict JSON Schema Extraction Example

```python
import json
import ollama

schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "vendor": {"type": "string"},
        "total_amount": {"type": "number"},
        "currency": {"type": "string"},
    },
    "required": ["invoice_number", "vendor", "total_amount", "currency"],
}

response = ollama.chat(
    model="hf.co/nihalmotukuri/actuator-7b-gguf:Q4_K_M",
    messages=[
        {
            "role": "system",
            "content": f"Output ONLY valid raw JSON matching this schema:\n{json.dumps(schema)}",
        },
        {
            "role": "user",
            "content": "Invoice #INV-904 from CloudScale Systems. Total payable: 18500 INR.",
        },
    ],
)

data = json.loads(response["message"]["content"])
print(data)
```

---

## Training & Dataset Architecture

Actuator-7B was fine-tuned using **4-bit QLoRA** (`r=16`, `lora_alpha=32`, all linear projections) with **response-only loss masking** (`train_on_responses_only`) on **4,800 verified examples**:

| Dataset Source | Samples | Capability Targeted |
| :--- | :---: | :--- |
| `Salesforce/xlam-function-calling-60k` (filtered & normalized) | 2,250 | Single, parallel, and nested argument tool calling |
| `NousResearch/hermes-function-calling-v1` (`func_calling`) | 1,400 | Multi-turn execution loops, error recovery, and abstention |
| `NousResearch/hermes-function-calling-v1` (`json_mode`, `jsonschema`-verified) | 1,150 | Raw, markdown-free JSON Schema adherence |

### Training Hyperparameters
- **Base Model:** `unsloth/Qwen2.5-7B-Instruct`
- **Context Length:** `3,072` tokens
- **Effective Batch Size:** `16`
- **Optimizer & Schedule:** `adamw_8bit`, cosine LR schedule (`2e-4`, 15 warmup steps, `bf16`)
- **Hardware:** NVIDIA GeForce RTX 5070 Ti (12 GB VRAM)

---

## Repository Structure

```text
actuator-7b/
├── scripts/
│   ├── build_dataset.py      # Curates & validates xLAM + Strict JSON Schema splits
│   ├── patch_hermes.py       # Parses multi-turn & abstention conversations
│   ├── train_actuator.py     # Unsloth QLoRA training with response-only masking
│   ├── export_gguf.py        # Merges LoRA & exports Q4_K_M / Q8_0 GGUFs
│   └── upload_hf.py          # Publishes GGUFs & model card to Hugging Face
├── eval/
│   ├── run_benchmark.py      # 150-task automated benchmark suite
│   └── results.json          # Raw evaluation metrics vs stock Qwen2.5-7B-Instruct
├── modelfile/
│   └── Modelfile             # Ollama configuration with native Qwen {{ .Tools }} template
└── README.md
```

---

## License

- **Code & Scripts:** Apache 2.0
- **Model Weights:** Inherits the [Qwen2.5-7B-Instruct License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE) (Apache 2.0).
