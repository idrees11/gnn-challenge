"""
update_leaderboard.py
Automatically updates leaderboard/leaderboard.md based on participant submissions.
"""

import os
import pandas as pd
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
SUBMISSIONS_DIR = "submissions"
LEADERBOARD_FILE = "leaderboard/leaderboard.md"

# Column names expected in submission
REQUIRED_COLUMNS = ["graph_index", "label"]  # or "target" depending on your setup

# ----------------------------
# Helper functions
# ----------------------------

def read_submission(file_path):
    try:
        df = pd.read_csv(file_path)
        # Detect the label column automatically
        label_col = "label" if "label" in df.columns else "target"
        if label_col not in df.columns:
            raise ValueError(f"No 'label' or 'target' column found in {file_path}")
        return df, label_col
    except Exception as e:
        print(f"Error reading submission {file_path}: {e}")
        return None, None

def read_score_file(file_path):
    """Expect a CSV with 'participant' and 'f1_score' columns"""
    try:
        df = pd.read_csv(file_path)
        if "participant" in df.columns and "f1_score" in df.columns:
            return df
        else:
            print(f"Score file {file_path} missing required columns")
            return None
    except Exception as e:
        print(f"Could not read score file {file_path}: {e}")
        return None

def update_leaderboard():
    os.makedirs("leaderboard", exist_ok=True)

    leaderboard = []

    # Loop over all CSV submissions
    for file in os.listdir(SUBMISSIONS_DIR):
        if file.endswith(".csv"):
            submission_path = os.path.join(SUBMISSIONS_DIR, file)
            df, label_col = read_submission(submission_path)
            if df is None:
                continue

            # Look for a corresponding score file
            score_file = submission_path.replace(".csv", "_score.csv")
            if os.path.exists(score_file):
                score_df = pd.read_csv(score_file)
                f1 = score_df["f1_score"].iloc[0]
            else:
                f1 = "Pending"

            participant = file.replace(".csv", "")
            leaderboard.append({
                "participant": participant,
                "f1_score": f1,
                "submission_file": file,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Sort by F1 score (descending), Pending scores go last
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
