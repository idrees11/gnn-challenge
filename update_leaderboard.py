"""
update_leaderboard.py
Automatically updates leaderboard/leaderboard.md based on participant submissions.
Uses TEST_LABELS secret for scoring.
"""

import os
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_FILE = "leaderboard/leaderboard.md"

# ----------------------------
# Helper functions
# ----------------------------

def run_scoring_script(file_path):
    """Compute F1 score for a submission using TEST_LABELS secret"""
    # Load submission
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    label_col = "label" if "label" in df.columns else "target"
    if label_col not in df.columns:
        raise ValueError(f"No 'label' or 'target' column found in {file_path}")

    # Load private labels from secret
    private_labels_csv = os.getenv("TEST_LABELS")
    if private_labels_csv is None:
        raise RuntimeError("TEST_LABELS secret not set in GitHub Actions")

    truth = pd.read_csv(io.StringIO(private_labels_csv))
    truth.columns = truth.columns.str.strip()

    if "target" not in truth.columns:
        raise ValueError("Private labels missing 'target' column")

    # Compute F1 score
    score = f1_score(truth['target'], df[label_col], average='macro')
    return score

# ----------------------------
# Update leaderboard
# ----------------------------
def update_leaderboard():
    os.makedirs("leaderboard", exist_ok=True)
    leaderboard = []

    for file in os.listdir(SUBMISSIONS_DIR):
        if file.endswith(".csv"):
            submission_path = os.path.join(SUBMISSIONS_DIR, file)
            participant = file.replace(".csv", "")
            try:
                f1 = run_scoring_script(submission_path)
            except Exception as e:
                print(f"Error scoring {file}: {e}")
                f1 = "Error"

            leaderboard.append({
                "participant": participant,
                "f1_score": f1,
                "submission_file": file,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Sort leaderboard by F1 score, errors go last
    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda x: float(x["f1_score"]) if isinstance(x["f1_score"], (int, float)) else -1,
        reverse=True
    )

    # Write markdown leaderboard
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("# Competition Leaderboard\n\n")
        f.write("| Rank | Participant | F1 Score | Submission | Timestamp |\n")
        f.write("|------|------------|----------|------------|-----------|\n")
        for rank, entry in enumerate(leaderboard_sorted, start=1):
            f.write(f"| {rank} | {entry['participant']} | {entry['f1_score']} | {entry['submission_file']} | {entry['timestamp']} |\n")

    print(f"Leaderboard updated: {LEADERBOARD_FILE}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    update_leaderboard()
