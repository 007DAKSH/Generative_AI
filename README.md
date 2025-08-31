# text1_generation.py
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

print("🚀 Loading dataset...")

# Load dataset from your train.txt file
dataset = load_dataset("text", data_files={"train": "train.txt"})

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding issue

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Load GPT2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"   # Disable WandB/other loggers
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()

# Save final model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("✅ Training complete! Model saved in ./trained_model")
