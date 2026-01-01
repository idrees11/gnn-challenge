import pandas as pd
from sklearn.metrics import f1_score
import sys

# ----------------------------
# Usage check
# ----------------------------
if len(sys.argv) != 3:
    print("Usage: python scoring_script.py <test_labels.csv> <submission.csv>")
    sys.exit(1)

truth_file = sys.argv[1]
submission_file = sys.argv[2]

# ----------------------------
# Load CSVs
# ----------------------------
truth = pd.read_csv(truth_file)
submission = pd.read_csv(submission_file)

# Strip extra spaces from column names
truth.columns = truth.columns.str.strip()
submission.columns = submission.columns.str.strip()

# ----------------------------
# Detect label column automatically
# ----------------------------
for col in ['label', 'target']:
    if col in truth.columns:
        truth_col = col
        break
else:
    raise ValueError(f"None of the expected label columns ('label' or 'target') found in {truth_file}")

for col in ['label', 'target']:
    if col in submission.columns:
        submission_col = col
        break
else:
    raise ValueError(f"None of the expected label columns ('label' or 'target') found in {submission_file}")

# ----------------------------
# Compute F1 score
# ----------------------------
score = f1_score(truth[truth_col], submission[submission_col], average="macro")
print(f"Submission F1 Score: {score:.4f}")
