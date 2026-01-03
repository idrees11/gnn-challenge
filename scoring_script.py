import os
import pandas as pd
from sklearn.metrics import f1_score
import sys

print("Running scoring_script.py from:", sys.argv[0])

# ----------------------------
# CONFIG
# ----------------------------
PRIVATE_LABELS_ENV = "TEST_LABELS"
SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

# ----------------------------
# Load private labels
# ----------------------------
private_labels = os.getenv(PRIVATE_LABELS_ENV, "")

truth = None

if private_labels:
    try:
        # Try reading as CSV content (for secrets)
        import io
        truth = pd.read_csv(io.StringIO(private_labels))
        print("Loaded private labels from secret.")
    except Exception:
        # If fails, maybe secret is a file path
        if os.path.exists(private_labels):
            truth = pd.read_csv(private_labels)
            print(f"Loaded private labels from file path: {private_labels}")
        else:
            truth = None

if truth is None:
    print(
        f"WARNING: Private labels not found. Tried env '{PRIVATE_LABELS_ENV}'. "
        "Scoring will be skipped (expected for participant forks)."
    )

# ----------------------------
# Detect label column
# ----------------------------
truth_col = None
if truth is not None:
    truth.columns = truth.columns.str.strip()
    for col in ["label", "target"]:
        if col in truth.columns:
            truth_col = col
            break
    if truth_col is None:
        print("No 'label' or 'target' column found in private labels. Skipping scoring.")
        truth = None

# ----------------------------
# Evaluate submissions
# ----------------------------
scores = []

if not os.path.exists(SUBMISSIONS_FOLDER):
    print(f"Submissions folder not found: {SUBMISSIONS_FOLDER}. Nothing to score.")
else:
    for fname in os.listdir(SUBMISSIONS_FOLDER):
        if fname.endswith(".csv"):
            submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
            submission = pd.read_csv(submission_path)
            submission.columns = submission.columns.str.strip()

            # Detect label column
            submission_col = None
            for col in ["label", "target"]:
                if col in submission.columns:
                    submission_col = col
                    break

            if submission_col is None:
                print(f"Skipping {fname}: No 'label' or 'target' column found")
                continue

            if truth is not None:
                # Check length
                if len(submission) != len(truth):
                    print(f"Skipping {fname}: Length mismatch with ground truth")
                    continue

                # Compute F1 score
                score = f1_score(truth[truth_col], submission[submission_col], average="macro")
                print(f"{fname} -> F1 Score: {score:.4f}")
                scores.append({"submission": fname, "f1_score": score})
            else:
                # Participant fork / local run: skip scoring
                print(f"Found submission (skipping scoring): {fname}")
                scores.append({"submission": fname, "f1_score": None})

# ----------------------------
# Save leaderboard
# ----------------------------
if scores:
    leaderboard = pd.DataFrame(scores)
    if truth is not None:
        leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    print(f"\nLeaderboard saved to: {LEADERBOARD_FILE}")
else:
    print("No valid submissions found to score.")
