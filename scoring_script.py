# scoring_script.py
import os
import pandas as pd
from sklearn.metrics import f1_score

# ----------------------------
# CONFIG: PRIVATE DATA PATH
# ----------------------------
# Path to private test labels (local only)
PRIVATE_DATA_FOLDER = r"C:/Users/Idrees Bhat/Desktop/GNNs/Basira/Islem/private_data"  # <-- change to your path
PRIVATE_LABEL_FILE = os.path.join(PRIVATE_DATA_FOLDER, "test_labels.csv")

# Path to participant submissions folder
SUBMISSIONS_FOLDER = "submissions"

# Leaderboard CSV (updated automatically)
LEADERBOARD_FILE = "leaderboard.csv"

# ----------------------------
# Load private labels
# ----------------------------
if not os.path.exists(PRIVATE_LABEL_FILE):
    raise FileNotFoundError(f"Private labels not found at: {PRIVATE_LABEL_FILE}")

truth = pd.read_csv(PRIVATE_LABEL_FILE)
truth.columns = truth.columns.str.strip()

# Detect label column
for col in ['label', 'target']:
    if col in truth.columns:
        truth_col = col
        break
else:
    raise ValueError(f"No 'label' or 'target' column found in {PRIVATE_LABEL_FILE}")

# ----------------------------
# Evaluate each submission
# ----------------------------
scores = []

for fname in os.listdir(SUBMISSIONS_FOLDER):
    if fname.endswith(".csv"):
        submission_path = os.path.join(SUBMISSIONS_FOLDER, fname)
        submission = pd.read_csv(submission_path)
        submission.columns = submission.columns.str.strip()

        # Detect label column
        for col in ['label', 'target']:
            if col in submission.columns:
                submission_col = col
                break
        else:
            print(f"Skipping {fname}: No 'label' or 'target' column found")
            continue

        # Compute F1 score
        score = f1_score(truth[truth_col], submission[submission_col], average="macro")
        print(f"{fname} -> F1 Score: {score:.4f}")
        scores.append({"submission": fname, "f1_score": score})

# ----------------------------
# Save leaderboard
# ----------------------------
if scores:
    leaderboard = pd.DataFrame(scores)
    leaderboard = leaderboard.sort_values(by="f1_score", ascending=False)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    print(f"\nLeaderboard saved to: {LEADERBOARD_FILE}")
else:
    print("No valid submissions found to score.")
