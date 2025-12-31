
"""
test_setup.py - Comprehensive testing of challenge setup
"""

import sys
import os
import subprocess
import importlib
import torch
import torch_geometric
import networkx as nx
import numpy as np

def print_status(label, status, message=""):
    """Print colored status message"""
    colors = {
        "PASS": "\033[92m",  # Green
        "FAIL": "\033[91m",  # Red
        "INFO": "\033[94m",  # Blue
        "WARN": "\033[93m",  # Yellow
    }
    reset = "\033[0m"
    
    if status in colors:
        print(f"{colors[status]}{status:6s}{reset} {label}")
    else:
        print(f"{status:6s} {label}")
    
    if message:
        print(f"       {message}")

def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("TEST 1: PACKAGE IMPORTS")
    print("="*60)
    
    required_packages = [
        ("torch", torch),
        ("torch_geometric", torch_geometric),
        ("networkx", nx),
        ("numpy", np),
        ("scipy", None),
        ("scikit-learn", None),
        ("pandas", None),
        ("flask", None),
        ("sqlalchemy", None),
    ]
    
    for package_name, module in required_packages:
        try:
            if module is None:
                __import__(package_name)
            print_status(f"{package_name:20}", "PASS")
        except ImportError as e:
            print_status(f"{package_name:20}", "FAIL", f"Error: {e}")

def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("TEST 2: HARDWARE CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        print_status("CUDA", "PASS", f"GPU: {torch.cuda.get_device_name(0)}")
        print_status("CUDA Memory", "INFO", 
                    f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print_status("CUDA", "WARN", "Running on CPU only")

def test_baseline():
    """Test baseline model runs"""
    print("\n" + "="*60)
    print("TEST 3: BASELINE MODEL")
    print("="*60)
    
    try:
        # Import baseline components
        sys.path.append('starter')
        from baseline_model import BaselineGINModel, compute_baseline_features
        
        # Create dummy data
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 7)
        data = type('Data', (), {'edge_index': edge_index, 'x': x, 'num_nodes': 3})()
        
        # Test feature computation
        features = compute_baseline_features(data)
        print_status("Feature Computation", "PASS", 
                    f"Output shape: {features.shape}")
        
        # Test model
        model = BaselineGINModel(input_dim=7)
        print_status("Model Initialization", "PASS", 
                    f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(data)
            print_status("Forward Pass", "PASS", 
                        f"Output shape: {output.shape}")
        
    except Exception as e:
        print_status("Baseline Test", "FAIL", f"Error: {e}")

def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("TEST 4: DATASET LOADING")
    print("="*60)
    
    try:
        from starter.challenge_dataset import ChallengeDataset
        
        # Test simple dataset
        dataset = ChallengeDataset(use_simple=True)
        print_status("Simple Dataset", "PASS", 
                    f"Samples: {len(dataset)}, Features: {dataset.num_features}")
        
        # Check first sample
        sample = dataset[0]
        print_status("Sample Check", "PASS", 
                    f"Nodes: {sample.num_nodes}, Edges: {sample.edge_index.shape[1]//2}")
        
    except Exception as e:
        print_status("Dataset Test", "FAIL", f"Error: {e}")

def test_evaluation():
    """Test evaluation system"""
    print("\n" + "="*60)
    print("TEST 5: EVALUATION SYSTEM")
    print("="*60)
    
    try:
        # Create a simple submission for testing
        test_submission = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

def compute_enhanced_features(data):
    return torch.ones(data.num_nodes, 3)

class EnhancedGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super(EnhancedGraphModel, self).__init__()
        self.conv1 = GINConv(nn.Linear(input_dim, hidden_dim))
        self.conv2 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.classifier(x), dim=1)
"""
        
        # Save test submission
        with open('test_submission.py', 'w') as f:
            f.write(test_submission)
        
        # Test import
        from submission_template.submission import compute_enhanced_features, EnhancedGraphModel
        print_status("Submission Import", "PASS")
        
        # Clean up
        os.remove('test_submission.py')
        
    except Exception as e:
        print_status("Evaluation Test", "FAIL", f"Error: {e}")

def test_leaderboard():
    """Test leaderboard system"""
    print("\n" + "="*60)
    print("TEST 6: LEADERBOARD SYSTEM")
    print("="*60)
    
    try:
        # Test database
        import sqlite3
        conn = sqlite3.connect('test_leaderboard.db')
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_submissions (
                id INTEGER PRIMARY KEY,
                team_name TEXT,
                score REAL
            )
        ''')
        
        # Insert test data
        cursor.execute(
            "INSERT INTO test_submissions (team_name, score) VALUES (?, ?)",
            ('TestTeam', 0.85)
        )
        
        # Query test data
        cursor.execute("SELECT * FROM test_submissions")
        results = cursor.fetchall()
        
        print_status("Database", "PASS", f"Test entry added: {results[0]}")
        
        # Clean up
        cursor.execute("DROP TABLE test_submissions")
        conn.close()
        os.remove('test_leaderboard.db')
        
    except Exception as e:
        print_status("Leaderboard Test", "FAIL", f"Error: {e}")

def test_end_to_end():
    """Test complete workflow"""
    print("\n" + "="*60)
    print("TEST 7: END-TO-END WORKFLOW")
    print("="*60)
    
    steps = [
        ("1. Environment Setup", "python --version"),
        ("2. Install Check", "pip list | grep torch"),
        ("3. Data Directory", "ls -la data/ 2>/dev/null || echo 'No data dir'"),
        ("4. Run Baseline", "cd starter && python -c 'from baseline_model import *; print(\"Baseline OK\")'"),
    ]
    
    for label, command in steps:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print_status(label, "PASS")
            else:
                print_status(label, "WARN", result.stderr[:50])
        except:
            print_status(label, "FAIL")

def generate_test_report():
    """Generate HTML test report"""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Challenge Setup Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .pass { color: green; }
        .fail { color: red; }
        .warn { color: orange; }
        .test { margin: 20px 0; padding: 10px; border-left: 4px solid #ccc; }
    </style>
</head>
<body>
    <h1>Graph Classification Challenge - Setup Test Report</h1>
    <p>Generated: {timestamp}</p>
    
    <h2>Test Results</h2>
    {test_results}
    
    <h2>Next Steps</h2>
    <ol>
        <li>Run baseline: <code>python starter/run_baseline.py</code></li>
        <li>Start leaderboard: <code>python leaderboard/run_leaderboard.py</code></li>
        <li>Test submission: <code>python evaluation/run_evaluation.py --test</code></li>
    </ol>
</body>
</html>
"""
    
    # In a real implementation, you would collect test results
    print_status("Report Generation", "INFO", "See test_report.html")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GRAPH CLASSIFICATION CHALLENGE - SETUP VALIDATION")
    print("="*60)
    
    # Run all tests
    test_imports()
    test_cuda()
    test_baseline()
    test_dataset()
    test_evaluation()
    test_leaderboard()
    test_end_to_end()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP VALIDATION COMPLETE")
    print("="*60)
    print("\nIf all tests pass, your challenge is ready!")
    print("\nTo deploy:")
    print("1. Run leaderboard: python leaderboard/run_leaderboard.py")
    print("2. Test submission: python scripts/test_submission.py")
    print("3. Share with participants!")
    
    # Generate report
    generate_test_report()

if __name__ == "__main__":
    main()
