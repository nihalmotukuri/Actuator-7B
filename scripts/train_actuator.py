import json
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "outputs/actuator-7b-lora"
MAX_SEQ_LENGTH = 3072


def load_jsonl_dataset(path: str, tokenizer) -> Dataset:
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tools = item.get("tools") or None
            messages = item["messages"]

            # Render using Qwen2.5's native tool-calling chat template
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Filter out overly long sequences that would truncate assistant responses
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > MAX_SEQ_LENGTH:
                skipped += 1
                continue

            rows.append({"text": text})

    print(f"Loaded {len(rows)} formatted samples from {path} (skipped {skipped} > {MAX_SEQ_LENGTH} tokens).")
    return Dataset.from_list(rows)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = load_jsonl_dataset("data/train.jsonl", tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=15,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=OUTPUT_DIR,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            report_to="none",
        ),
    )

    # Mask system, tools, user, and tool_response turns; train ONLY on assistant completions
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Sanity check: verify masking preserved assistant tokens properly
    sample_labels = trainer.train_dataset[0]["labels"]
    non_masked = sum(1 for x in sample_labels if x != -100)
    print(f"Sanity check -> Sample 0 trainable assistant tokens: {non_masked} / {len(sample_labels)}")
    assert non_masked > 0, "Error: All tokens were masked! Check chat template markers."

    print("\nStarting QLoRA fine-tuning for Actuator-7B...")
    has_ckpt = any(Path(OUTPUT_DIR).glob("checkpoint-*"))
    trainer_stats = trainer.train(resume_from_checkpoint=has_ckpt)

    final_adapter_dir = Path("outputs/actuator-7b-final-lora")
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"\nTraining complete in {trainer_stats.metrics['train_runtime']:.1f}s!")
    print(f"Saved final LoRA adapter -> {final_adapter_dir}")


if __name__ == "__main__":
    main()
