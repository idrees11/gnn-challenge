**🧠 GNN Challenge: Graph Classification with Topological Features**
Overview

Welcome to the Graph Neural Networks (GNN) Challenge!
This competition focuses on graph classification using message passing neural networks, with a special emphasis on topological (structural) feature augmentation.
Participants are challenged to design models that can effectively combine node features, graph structure, and topological descriptors to improve classification performance.
The challenge is designed to be small, fast, yet non-trivial, and is fully solvable using methods covered in DGL Lectures 1.1–4.6.
**🎯 Problem Statement**
Given a graph  𝐺=(𝑉,𝐸), G=(V,E), predict its graph-level class label.
Each graph represents a molecular structure from the MUTAG dataset.
While basic node features are available, the key difficulty lies in leveraging graph topology effectively.
**🧩 Problem Type**
Graph Classification
Supervised Learning
Binary Classification
**📚 Relevant GNN Concepts (DGL 1.1–4.6)**
This challenge can be solved using:
Message passing neural networks (MPNNs)
Graph Isomorphism Networks (GIN)
Neighborhood aggregation
Graph-level readout (global mean pooling)
Structural / topological node features:
Degree
Clustering coefficient
**📦 Dataset**
MUTAG (TUDataset)
Source: TUDataset (automatically downloaded)
Graphs: 188 molecular graphs
Classes: 2
Nodes per graph: ~17 (average)
Edges: Undirected
This dataset is: Small enough for quick experimentation and Rich enough to benefit from structural features
Betweenness centrality
PageRank
k-core number
**🗂️ Data Splits**
The dataset is split once using a fixed random seed to ensure fair comparison:
Split	Percentage
Train	70%
Validation	10%
Test	20%
Files provided in data/:
train.csv → graph indices + labels
test.csv → graph indices only (labels hidden)
⚠️ Test labels are hidden and used only by the organizers for scoring.
**📊 Objective Metric**
Macro F1-score
Why Macro F1?
Sensitive to class imbalance
Difficult to optimize directly
Encourages balanced performance across classes
This is the official ranking metric.
**⚙️ Constraints**
To keep the competition fair and focused:
❌ No external datasets
❌ No pretraining
✅ Only methods covered in DGL Lectures 1.1–4.6
⏱ Models must run within 10 minutes on CPU
✅ Any GNN architecture allowed (GIN, GCN, GraphSAGE, etc.)
**🚀 Getting Started**
1️⃣ Install Dependencies
pip install -r starter_code/requirements.txt
2️⃣ Run the Baseline Model
cd starter_code
python baseline.py
This will:
Train a simple GIN model
Generate predictions on the test set
Save a submission file to submissions/sample_submission.csv
**📤 Submission Format**
Your submission must be a CSV file with exactly the following format:
graph_index,target
0,1
1,0
2,1
...
graph_index: Index of the graph in the dataset
target: Predicted class label (0 or 1)
🧪 Scoring
Submissions are evaluated using:
f1_score(y_true, y_pred, average="macro")
Scores are computed using a hidden test label file.
**🏆 Leaderboard**
Submissions are ranked by Macro F1-score
Higher is better
Ties are broken by submission time
Leaderboard is maintained in:
leaderboard.md
**💡 Tips for Success**
Structural features matter more than you think
Try different combinations of topological features
Regularization is important on small datasets
Simpler models often generalize better
**📁 Repository Structure**
gnn-challenge/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── starter_code/
│   ├── dataset.py
│   ├── model.py
│   ├── baseline.py
│   └── requirements.txt
│
├── submissions/
│   └── sample_submission.csv
│
├── scoring_script.py
├── leaderboard.md
└── README.md
**📜 License**
This project is released under the MIT License.
## 🏆 Leaderboard
The leaderboard is updated automatically for every valid submission.
How it works:
1. Submit a CSV file via Pull Request
2. GitHub Actions evaluates your submission
3. Macro F1-score is computed
4. Leaderboard is updated automatically
Only the **best-performing submissions** appear at the top.
#----------------------------------------------------------------#
#                       **Step-by-Step Commands**                #
#----------------------------------------------------------------# 
:: 1️⃣ Go to the starter_code folder
cd starter_code
:: 2️⃣ Run the baseline to generate submission
python baseline.py
:: 3️⃣ Go back to the repo root
cd ..
:: 4️⃣ Check that submission exists
dir submissions
:: 5️⃣ Score the submission
python scoring_script.py submissions\sample_submission.csv
#-------------------------------------------------------------------#
                        **What Each Step Does:**
#---------------------------------------------------------------------#
cd starter_code → enters the folder where baseline.py lives.
python baseline.py → trains the model and saves the CSV directly into the main submissions folder.
cd .. → returns to the main repo root.
dir submissions → verifies the CSV is there (sample_submission.csv).
python scoring_script.py submissions\sample_submission.csv → runs scoring and prints the F1 score
#------------------------------------------------------------------------------------#
**📬 Contact**
For questions or clarifications, please open a GitHub Issue.
Good luck — and happy graph learning! 🧠📊
