"""Creates the held-out test prompts for the before/after comparison -> data/test_prompts.json
These prompts are NOT in the training data (different wording and new claims).
Run:  python make_test_prompts.py
"""
import json
import sys

from examples_part1 import EXAMPLES as P1
from fmt import check_prompt

sys.stdout.reconfigure(encoding="utf-8")


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


GD = next(ex for ex in P1 if ex[0] == "What is gradient descent?")

TESTS = [
    # --- explain mode ---
    {"id": "explain_cv", "type": "explain",
     "messages": [user("Can you explain cross-validation in simple words?")]},
    {"id": "explain_precision", "type": "explain",
     "messages": [user("What is precision, and how is it different from recall?")]},
    {"id": "explain_ensemble_new", "type": "explain (topic not in training data)",
     "messages": [user("What is ensemble learning?")]},
    # --- misconception checker ---
    {"id": "check_overfitting", "type": "check (wrong claim)",
     "messages": [user(check_prompt("Overfitting", "Overfitting means the model performs badly on both training and test data."))]},
    {"id": "check_knn", "type": "check (wrong claim)",
     "messages": [user(check_prompt("kNN", "In kNN, the smaller k is, the more stable the model becomes."))]},
    {"id": "check_regularization", "type": "check (correct claim)",
     "messages": [user(check_prompt("Regularization", "Regularization limits the model's parameters and helps prevent overfitting."))]},
    {"id": "check_random_forest", "type": "check (wrong claim)",
     "messages": [user(check_prompt("Random forest", "A random forest is just one very deep tree."))]},
    {"id": "check_split", "type": "check (correct claim)",
     "messages": [user(check_prompt("Train/test split", "The test set should only be used to check the final result."))]},
    # --- off-topic ---
    {"id": "off_food", "type": "off-topic",
     "messages": [user("Can you give me a lasagna recipe?")]},
    {"id": "off_hello", "type": "off-topic",
     "messages": [user("Hi, how are you doing?")]},
    # --- follow-up (memory test) ---
    {"id": "follow_example", "type": "follow-up (memory)",
     "messages": [user(GD[0]), assistant(GD[1]), user("Can you give me a different example?")]},
    {"id": "follow_answer", "type": "follow-up (memory)",
     "messages": [user(GD[0]), assistant(GD[1]),
                  user("They might jump over the lowest point and the loss could go up instead of down.")]},
]

with open("data/test_prompts.json", "w", encoding="utf-8") as f:
    json.dump(TESTS, f, ensure_ascii=False, indent=2)
print(f"data/test_prompts.json: {len(TESTS)} test prompts")
