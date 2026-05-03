import os
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl.trainer.sft_trainer import SFTTrainer
import torch
import datasets

from feature_extractor import features_to_text
from discussion_processor import OUTPUT_DIR

MODEL_DIR = "forecast_model"

def get_example(discussion: str):
    """
    Return a dict of model features & target summary for the provided discussion filename.
    To do this, we read in the model file matching the provided date
    """
    datetime = discussion.split("_")[1]
    date = datetime[:8]
    time = datetime[-4:]
    cycle = f"{(((int(time) // 6) * 6 - 6) % 24):02}"       # Get the most recent model cycle before discussion

    model_features = features_to_text(date, cycle, "006")

    with open(f"{OUTPUT_DIR}/{discussion}", "r") as f:
        contents = f.read()

    return {"features_text": model_features, "simplified": contents}

def format_example(example):
    return {
        "text": f"### Weather Features:\n{example['features_text']}\n\n"
                f"### Forecast Summary:\n{example['simplified']}"
    }

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
checkpoint_path = f"{MODEL_DIR}/checkpoint-30"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if not Path(checkpoint_path).is_dir():
    print("Model file not found. Training model...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # rank - keep small to avoid overfitting
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # which layers to adapt
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # should be a small fraction of total

    dataset = datasets.Dataset.from_list([get_example(discussion) for discussion in os.listdir(OUTPUT_DIR)])
    dataset = dataset.map(format_example)

    # Training arguments - conservative settings for small dataset
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,         # keep low to avoid overfitting
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False                  # CPU training
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # dataset_text_field="text",
        # max_seq_length=512
    )

    trainer.train()

if len(sys.argv) < 4:
    print("Usage: model.py <date (YYYYMMDD)> <forecast cycle (00, 06, 12, 18)> <forecast hour (XX)>")

model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

def generate_forecast_summary(feature_text, max_new_tokens=200, temperature=0.7):
    prompt = f"### Weather Features:\n{feature_text}\n\n### Forecast Summary:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and strip the input prompt from the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_output[len(prompt):]
    return summary.strip()

date = sys.argv[1]
cycle = sys.argv[2]
hour = sys.argv[3]

feature_text = features_to_text(date, cycle, hour)
print(generate_forecast_summary(feature_text))
