import sys
import pandas as pd
from sklearn.metrics import f1_score

submission = pd.read_csv(sys.argv[1])
truth = pd.read_csv("data/test_labels.csv")

score = f1_score(truth.target, submission.target, average="macro")
print(f"Submission F1 Score: {score:.4f}")

