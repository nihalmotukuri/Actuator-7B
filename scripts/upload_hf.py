import json
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "nihalmotukuri/actuator-7b-gguf"
api = HfApi()


def main():
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    existing_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))

    q4_file = next(Path("outputs/gguf").rglob("*Q4_K_M.gguf"))
    q8_file = next(Path("outputs/gguf").rglob("*Q8_0.gguf"))

    if "Actuator-7B-Q4_K_M.gguf" not in existing_files:
        print(f"Uploading {q4_file} -> Actuator-7B-Q4_K_M.gguf via LFS...")
        api.upload_file(
            path_or_fileobj=str(q4_file),
            path_in_repo="Actuator-7B-Q4_K_M.gguf",
            repo_id=REPO_ID,
            repo_type="model",
        )

    if "Actuator-7B-Q8_0.gguf" not in existing_files:
        print(f"Uploading {q8_file} -> Actuator-7B-Q8_0.gguf via LFS...")
        api.upload_file(
            path_or_fileobj=str(q8_file),
            path_in_repo="Actuator-7B-Q8_0.gguf",
            repo_id=REPO_ID,
            repo_type="model",
        )

    params_json = json.dumps({
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["<|im_start|>", "<|im_end|>"]
    }, indent=2)
    api.upload_file(
        path_or_fileobj=params_json.encode("utf-8"),
        path_in_repo="params",
        repo_id=REPO_ID,
        repo_type="model",
    )

    system_prompt = "You are Actuator-7B, a deterministic tool-calling and structured-data engine."
    api.upload_file(
        path_or_fileobj=system_prompt.encode("utf-8"),
        path_in_repo="system",
        repo_id=REPO_ID,
        repo_type="model",
    )

    yaml_header = (
        "---\n"
        "license: apache-2.0\n"
        "base_model: Qwen/Qwen2.5-7B-Instruct\n"
        "tags:\n"
        "  - gguf\n"
        "  - ollama\n"
        "  - llama-cpp\n"
        "  - function-calling\n"
        "  - tool-calling\n"
        "  - json-schema\n"
        "  - agents\n"
        "  - mcp\n"
        "  - qlora\n"
        "  - unsloth\n"
        "  - qwen2.5\n"
        "language:\n"
        "  - en\n"
        "---\n\n"
    )
    readme_body = Path("README.md").read_text(encoding="utf-8")
    hf_readme = Path("outputs/HF_README.md")
    hf_readme.write_text(yaml_header + readme_body, encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(hf_readme),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"\nPublished successfully -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
