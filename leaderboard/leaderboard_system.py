import io
import os
import pandas as pd
from sklearn.metrics import f1_score

# ----------------------------
# Helper function to score a submission
# ----------------------------
def run_scoring_script(file_content):
    """
    Runs scoring on an uploaded submission file content.
    Returns a dictionary with final_score, accuracy, efficiency_score, novelty_score.
    """
    # Make sure submissions folder exists
    os.makedirs("submissions", exist_ok=True)
    
    # Save uploaded submission temporarily
    temp_path = "submissions/temp_submission.csv"
    with open(temp_path, "w") as f:
        f.write(file_content)
    
    # Load private labels from GitHub Actions secret
    private_labels_csv = os.getenv("TEST_LABELS")
    if private_labels_csv is None:
        raise RuntimeError("TEST_LABELS secret not set. Add it in repo Settings → Secrets → Actions.")
    
    truth = pd.read_csv(io.StringIO(private_labels_csv))
    truth.columns = truth.columns.str.strip()
    
    # Load submission CSV
    submission = pd.read_csv(temp_path)
    submission.columns = submission.columns.str.strip()
    
    # Ensure both have the target column
    if 'target' not in truth.columns or 'target' not in submission.columns:
        raise ValueError("Missing 'target' column in either private labels or submission")
    
    # Compute macro F1 score
    score = f1_score(truth['target'], submission['target'], average='macro')
    
    # Return structured results
    return {
        'final_score': score,
        'accuracy': score,        # you can add separate metrics if needed
        'efficiency_score': 0.7, # placeholder
        'novelty_score': 0.6     # placeholder
    }
