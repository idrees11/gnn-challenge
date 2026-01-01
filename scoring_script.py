import pandas as pd
from sklearn.metrics import f1_score
import sys

# Usage: python scoring_script.py <test_labels.csv> <submission.csv>

truth = pd.read_csv(sys.argv[1])
submission = pd.read_csv(sys.argv[2])

# Clean column names to remove extra spaces
truth.columns = truth.columns.str.strip()
submission.columns = submission.columns.str.strip()

# Use 'label' column instead of 'target'
score = f1_score(truth['label'], submission['label'], average="macro")
print(f"Submission F1 Score: {score:.4f}")
