# Reward Model Training & Evaluation Summary

## Overview
This project aimed to capture human preferences in code by training a reward model using HuggingFace and TRL tools. The reward model was trained to predict human ratings for answers to various prompts, forming the basis for later RLHF (Reinforcement Learning from Human Feedback) experiments.

## Steps Taken
1. **Prompt & Answer Generation:**
   - Five diverse prompts were selected.
   - For each prompt, four candidate answers were generated using a base language model (GPT-2).

2. **Human Ranking & Rating:**
   - Each answer was manually assigned a relative rank (1–4) and an absolute rating (1–5) based on quality and preference.
   - The data was saved in `answers.csv`.

3. **Reward Model Training:**
   - A regression model (DistilBERT) was trained using HuggingFace's Trainer to predict the human-provided ratings.
   - The model and tokenizer were saved in the `reward_model/` directory.

4. **Evaluation:**
   - The trained reward model was used to score a new set of answers.
   - Reward scores were plotted against human ratings to verify correlation.

## Key Findings
- The reward model's scores generally correlated with human ratings, indicating it learned to capture subjective preferences.
- The workflow required careful version management of Python, transformers, and TRL libraries for compatibility.
- Saving both the model and tokenizer is essential for later evaluation and deployment.

## Next Steps
- Use the trained reward model as a feedback mechanism in RLHF (PPO) training.
- Further analyze and refine the model using more data or different prompts if needed.

---
**Deliverables produced:**
- `answers.csv` (labeled data)
- `reward_model/` (trained model)
- `evaluate_reward_model.py` (scoring and plotting script)
- `summary.md` (this summary) 