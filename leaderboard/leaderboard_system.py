"""
leaderboard_system.py
Automated leaderboard system for the challenge
"""

import json
import csv
import sqlite3
import hashlib
import datetime
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os
import threading
import time

# HTML template for leaderboard display
LEADERBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Graph Classification Challenge - Leaderboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }
        .leaderboard { margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .rank-1 { background-color: gold; }
        .rank-2 { background-color: silver; }
        .rank-3 { background-color: #cd7f32; }
        .badge { padding: 4px 8px; border-radius: 12px; font-size: 12px; }
        .badge-new { background: #2ecc71; color: white; }
        .badge-improved { background: #3498db; color: white; }
        .chart-container { margin-top: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 Graph Classification Challenge</h1>
        <p>Topological Feature Engineering Leaderboard</p>
    </div>
    
    <div class="leaderboard">
        <h2>Current Rankings</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Team Name</th>
                    <th>Score</th>
                    <th>Accuracy</th>
                    <th>Efficiency</th>
                    <th>Novelty</th>
                    <th>Submission Time</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for entry in leaderboard %}
                <tr class="rank-{{ entry.rank }}">
                    <td>{{ entry.rank }}</td>
                    <td>{{ entry.team_name }}</td>
                    <td><strong>{{ "%.4f"|format(entry.final_score) }}</strong></td>
                    <td>{{ "%.4f"|format(entry.accuracy) }}</td>
                    <td>{{ "%.4f"|format(entry.efficiency_score) }}</td>
                    <td>{{ "%.4f"|format(entry.novelty_score) }}</td>
                    <td>{{ entry.timestamp }}</td>
                    <td>
                        {% if entry.is_new %}
                        <span class="badge badge-new">NEW</span>
                        {% elif entry.is_improved %}
                        <span class="badge badge-improved">IMPROVED</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="chart-container">
        <h2>Score Distribution</h2>
        <img src="/score_distribution.png" alt="Score Distribution" width="800">
    </div>
    
    <div style="margin-top: 30px;">
        <h3>Submit Your Solution</h3>
        <form action="/submit" method="post" enctype="multipart/form-data">
            <input type="text" name="team_name" placeholder="Team Name" required><br><br>
            <input type="file" name="submission_file" accept=".py" required><br><br>
            <input type="submit" value="Submit Solution">
        </form>
    </div>
    
    <footer style="margin-top: 50px; text-align: center; color: #7f8c8d;">
        <p>Last updated: {{ update_time }}</p>
        <p>Total submissions: {{ total_submissions }}</p>
    </footer>
</body>
</html>
"""

class LeaderboardSystem:
    """Automated leaderboard system"""
    
    def __init__(self, db_file='leaderboard.db'):
        self.db_file = db_file
        self.init_database()
        self.app = Flask(__name__)
        self.setup_routes()
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                submission_hash TEXT UNIQUE,
                final_score REAL,
                accuracy REAL,
                efficiency_score REAL,
                novelty_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_new BOOLEAN DEFAULT 1,
                is_improved BOOLEAN DEFAULT 0
            )
        ''')
        
        # Create history table for tracking improvements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT,
                score REAL,
                submission_time DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_hash(self, file_content):
        """Calculate hash of submission for duplicate detection"""
        return hashlib.sha256(file_content.encode()).hexdigest()
    
    def process_submission(self, team_name, file_content, results):
        """Process and validate a submission"""
        submission_hash = self.calculate_hash(file_content)
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Check for duplicates
        cursor.execute(
            "SELECT final_score FROM submissions WHERE submission_hash = ?",
            (submission_hash,)
        )
        duplicate = cursor.fetchone()
        
        if duplicate:
            conn.close()
            return False, "Duplicate submission"
        
        # Check for existing team
        cursor.execute(
            "SELECT MAX(final_score) FROM submissions WHERE team_name = ?",
            (team_name,)
        )
        existing_score = cursor.fetchone()[0]
        
        is_new = existing_score is None
        is_improved = not is_new and results['final_score'] > existing_score
        
        # Insert submission
        cursor.execute('''
            INSERT INTO submissions 
            (team_name, submission_hash, final_score, accuracy, 
             efficiency_score, novelty_score, is_new, is_improved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            team_name, submission_hash,
            results['final_score'], results['accuracy'],
            results['efficiency_score'], results['novelty_score'],
            is_new, is_improved
        ))
        
        # Record in history
        cursor.execute('''
            INSERT INTO score_history (team_name, score, submission_time)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (team_name, results['final_score']))
        
        conn.commit()
        conn.close()
        
        return True, "Submission accepted"
    
    def get_leaderboard(self, limit=50):
        """Get current leaderboard"""
        conn = sqlite3.connect(self.db_file)
        query = '''
            SELECT 
                ROW_NUMBER() OVER (ORDER BY final_score DESC) as rank,
                team_name,
                final_score,
                accuracy,
                efficiency_score,
                novelty_score,
                timestamp,
                is_new,
                is_improved
            FROM submissions
            ORDER BY final_score DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        # Convert to list of dicts
        leaderboard = df.to_dict('records')
        return leaderboard
    
    def get_statistics(self):
        """Get challenge statistics"""
        conn = sqlite3.connect(self.db_file)
        
        stats = {}
        
        # Total submissions
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM submissions")
        stats['total_submissions'] = cursor.fetchone()[0]
        
        # Unique teams
        cursor.execute("SELECT COUNT(DISTINCT team_name) FROM submissions")
        stats['unique_teams'] = cursor.fetchone()[0]
        
        # Average score
        cursor.execute("SELECT AVG(final_score) FROM submissions")
        stats['average_score'] = cursor.fetchone()[0] or 0
        
        # Top score
        cursor.execute("SELECT MAX(final_score) FROM submissions")
        stats['top_score'] = cursor.fetchone()[0] or 0
        
        conn.close()
        return stats
    
    def generate_score_plot(self):
        """Generate score distribution plot"""
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT final_score FROM submissions", conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.hist(df['final_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = 'static/score_distribution.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def leaderboard():
            leaderboard_data = self.get_leaderboard()
            stats = self.get_statistics()
            
            return render_template_string(
                LEADERBOARD_TEMPLATE,
                leaderboard=leaderboard_data,
                total_submissions=stats['total_submissions'],
                update_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        @self.app.route('/submit', methods=['POST'])
        def submit():
            team_name = request.form.get('team_name')
            submission_file = request.files.get('submission_file')
            
            if not team_name or not submission_file:
                return jsonify({'error': 'Missing team name or submission file'}), 400
            
            try:
                # Read submission file
                file_content = submission_file.read().decode('utf-8')
                
                # Here you would run the scoring script
                # For demo, we'll use mock results
                # In practice: results = run_scoring_script(file_content)
                
                # Mock results for demonstration
                mock_results = {
                    'final_score': 0.75 + (hash(team_name) % 100) / 1000,
                    'accuracy': 0.85,
                    'efficiency_score': 0.65,
                    'novelty_score': 0.7,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Process submission
                success, message = self.process_submission(
                    team_name, file_content, mock_results
                )
                
                if success:
                    # Update plot
                    self.generate_score_plot()
                    return jsonify({
                        'success': True,
                        'message': message,
                        'score': mock_results['final_score'],
                        'rank': 1  # Would calculate actual rank
                    })
                else:
                    return jsonify({'error': message}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/leaderboard')
        def api_leaderboard():
            """JSON API for leaderboard"""
            leaderboard = self.get_leaderboard()
            return jsonify(leaderboard)
        
        @self.app.route('/api/statistics')
        def api_statistics():
            """JSON API for statistics"""
            stats = self.get_statistics()
            return jsonify(stats)
        
        @self.app.route('/api/submit/json', methods=['POST'])
        def api_submit_json():
            """JSON submission endpoint (for programmatic submissions)"""
            data = request.json
            
            if not data or 'team_name' not in data or 'code' not in data:
                return jsonify({'error': 'Missing team_name or code'}), 400
            
            try:
                # Process submission
                mock_results = {
                    'final_score': 0.75 + (hash(data['team_name']) % 100) / 1000,
                    'accuracy': 0.85,
                    'efficiency_score': 0.65,
                    'novelty_score': 0.7
                }
                
                success, message = self.process_submission(
                    data['team_name'], data['code'], mock_results
                )
                
                if success:
                    self.generate_score_plot()
                    return jsonify({
                        'success': True,
                        'message': message,
                        'results': mock_results
                    })
                else:
                    return jsonify({'error': message}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the leaderboard server"""
        # Create static directory for plots
        os.makedirs('static', exist_ok=True)
        
        # Generate initial plot
        self.generate_score_plot()
        
        print(f"🚀 Leaderboard system starting on http://{host}:{port}")
        print(f"📊 Database: {self.db_file}")
        print(f"📈 Static files: static/")
        
        self.app.run(host=host, port=port, debug=debug)

def start_background_updater(leaderboard_system, interval=300):
    """Background thread to update plots and statistics"""
    def updater():
        while True:
            time.sleep(interval)
            print(f"[{datetime.datetime.now()}] Updating leaderboard plots...")
            leaderboard_system.generate_score_plot()
    
    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    return thread

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Challenge Leaderboard System')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--reset', action='store_true', help='Reset database')
    
    args = parser.parse_args()
    
    # Initialize system
    leaderboard = LeaderboardSystem()
    
    # Start background updater
    start_background_updater(leaderboard)
    
    # Run server
    leaderboard.run(host=args.host, port=args.port)
