from pathlib import Path
from unsloth import FastLanguageModel

ADAPTER_DIR = "outputs/actuator-7b-final-lora"
EXPORT_DIR = Path("outputs/gguf")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading fine-tuned LoRA adapter from {ADAPTER_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=3072,
        dtype=None,
        load_in_4bit=True,
    )

    print("Exporting Q4_K_M GGUF (recommended default)...")
    model.save_pretrained_gguf(
        str(EXPORT_DIR / "q4_k_m"),
        tokenizer,
        quantization_method="q4_k_m",
    )

    print("Exporting Q8_0 GGUF (high-precision variant)...")
    model.save_pretrained_gguf(
        str(EXPORT_DIR / "q8_0"),
        tokenizer,
        quantization_method="q8_0",
    )

    print("\nGGUF export complete! Listing generated files:")
    for f in sorted(EXPORT_DIR.rglob("*.gguf")):
        size_gb = f.stat().st_size / (1024**3)
        print(f" - {f} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
