import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Load the trained reward model
model_dir = "reward_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Load a set of answers to score (use answers.csv or a new CSV)
answers_df = pd.read_csv("answers.csv")  # Or use a new file if you want

# Prepare inputs
texts = answers_df["prompt"] + " " + answers_df["answer"]

scores = []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        score = outputs.logits.item()
        scores.append(score)

answers_df["reward_score"] = scores

# Print results
print(answers_df[["prompt", "answer", "rank", "rating", "reward_score"]])

# Plot: reward_score vs. rating
plt.figure(figsize=(8, 5))
plt.scatter(answers_df["rating"], answers_df["reward_score"], c="blue")
plt.xlabel("Human Rating")
plt.ylabel("Reward Model Score")
plt.title("Reward Model Score vs. Human Rating")
plt.grid(True)
plt.show()

tokenizer.save_pretrained("reward_model/")
