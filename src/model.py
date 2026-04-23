import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl.trainer.sft_trainer import SFTTrainer
import datasets

from gfs_reader import download_file, read_grids
from feature_extractor import features_to_text
from discussion_processor import OUTPUT_DIR

def get_example(discussion: str):
    """
    Return a dict of model features & target summary for the provided discussion filename.
    To do this, we read in the model file matching the provided date
    """
    datetime = discussion.split("_")[1]
    date = datetime[:8]
    time = datetime[-4:]
    cycle = f"{(((int(time) // 6) * 6 - 6) % 24):02}"

    model_file = download_file(date, cycle, "006")
    ds_mslp, z500_anom, ds_u250, ds_v250 = read_grids(model_file)
    model_features = features_to_text(ds_mslp, z500_anom, ds_u250, ds_v250)

    with open(f"{OUTPUT_DIR}/{discussion}", "r") as f:
        contents = f.read()

    return {"features_text": model_features, "simplified": contents}

# Prepare dataset
# Each example is a (input_features_text, target_summary) pair
def format_example(example):
    return {
        "text": f"### Weather Features:\n{example['features_text']}\n\n"
                f"### Forecast Summary:\n{example['simplified']}"
    }

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
    output_dir='./forecast_model',
    num_train_epochs=3,         # keep low to avoid overfitting
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_strategy='epoch',
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
