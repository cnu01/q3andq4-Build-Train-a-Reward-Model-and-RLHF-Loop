import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the CSV
csv_path = "answers.csv"
df = pd.read_csv(csv_path)

# Use 'prompt' + 'answer' as input, 'rating' as reward
# Drop rows without a rating
train_df = df.dropna(subset=["rating"]).copy()
train_df["text"] = train_df["prompt"] + " " + train_df["answer"]
train_df["reward"] = train_df["rating"].astype(float)

# Prepare HuggingFace dataset
train_dataset = Dataset.from_pandas(train_df[["text", "reward"]])
train_dataset = train_dataset.rename_column("reward", "labels")

# Model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Tokenize
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="reward_model",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to=[],
)

# Define compute_metrics for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return {"mse": mean_squared_error(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("reward_model/")
tokenizer.save_pretrained("reward_model/")
print("Reward model trained and saved to reward_model/") 