import pandas as pd
from sklearn.metrics import f1_score
import sys

# Usage: python scoring_script.py <test_labels.csv> <submission.csv>

if len(sys.argv) != 3:
    print("Usage: python scoring_script.py <test_labels.csv> <submission.csv>")
    sys.exit(1)

truth_file = sys.argv[1]
submission_file = sys.argv[2]

# Load CSVs
truth = pd.read_csv(truth_file)
submission = pd.read_csv(submission_file)

# Clean column names (remove extra spaces)
truth.columns = truth.columns.str.strip()
submission.columns = submission.columns.str.strip()

# Use 'label' column instead of 'target'
score = f1_score(truth['label'], submission['label'], average="macro")
print(f"Submission F1 Score: {score:.4f}")
