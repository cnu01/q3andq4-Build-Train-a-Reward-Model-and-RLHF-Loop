import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Prompts (edit these if you want)
prompts = [
    "Summarize the plot of Hamlet.",
    "Summarize the process of photosynthesis.",
    "Summarize the causes of World War I.",
    "Summarize how a car engine works.",
    "Summarize the benefits of regular exercise."
]


# Model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

results = []

for prompt in prompts:
    outputs = generator(prompt, max_length=60, num_return_sequences=4, do_sample=True, temperature=0.9)
    for i, output in enumerate(outputs):
        answer = output["generated_text"][len(prompt):].strip()
        results.append({"prompt": prompt, "answer": answer, "rank": "", "rating": ""})  # Leave rank and rating blank for now

# Save to CSV
with open("answers.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["prompt", "answer", "rank", "rating"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Generated answers saved to answers.csv. Please open the file and manually rank the answers (1=best, 4=worst) for each prompt.") 