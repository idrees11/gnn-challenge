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

# Determine label column name
label_col = 'label' if 'label' in truth.columns else 'target'

# Check required columns
for col in ['graph_index', label_col]:
    if col not in truth.columns:
        raise ValueError(f"Column '{col}' not found in test labels file!")
    if col not in submission.columns:
        raise ValueError(f"Column '{col}' not found in submission file!")

# Compute F1 score
score = f1_score(truth[label_col], submission[label_col], average="macro")
print(f"Submission F1 Score: {score:.4f}")
