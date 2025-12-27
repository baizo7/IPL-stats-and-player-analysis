from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import glob
import json

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Data loading functions
def load_data():
    """Load and clean IPL data"""
    csv_file = "all_ipl_matches.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, low_memory=False)
    else:
        df = create_sample_data()
    
    return clean_data(df)

def create_sample_data():
    teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
             'Punjab Kings', 'Sunrisers Hyderabad']
    batters = ['MS Dhoni', 'Rohit Sharma', 'Virat Kohli', 'Andre Russell', 
               'Rishabh Pant', 'Jos Buttler', 'KL Rahul', 'David Warner']
    bowlers = ['Jasprit Bumrah', 'Rashid Khan', 'Yuzvendra Chahal', 'Trent Boult',
               'Kagiso Rabada', 'Mohammed Shami', 'Sunil Narine', 'Bhuvneshwar Kumar']
    
    n_records = 5000
    df = pd.DataFrame({
        'match_id': np.repeat(range(1, 51), 100),
        'batting_team': np.random.choice(teams, n_records),
        'bowling_team': np.random.choice(teams, n_records),
        'ball': np.tile(np.arange(0.1, 20.1, 0.1), n_records // 200 + 1)[:n_records],
        'runs_off_bat': np.random.choice([0, 0, 0, 1, 1, 1, 2, 3, 4, 6], n_records, p=[0.35, 0.15, 0.1, 0.2, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]),
        'extras': np.random.choice([0, 1], n_records, p=[0.9, 0.1]),
        'wicket_type': np.random.choice([None, 'caught', 'bowled', 'lbw', 'run out'], n_records, p=[0.93, 0.03, 0.02, 0.01, 0.01]),
        'batter': np.random.choice(batters, n_records),
        'bowler': np.random.choice(bowlers, n_records)
    })
    return df

def clean_data(df):
    df = df.copy()
    
    # Column mapping - normalize different column names
    column_mappings = {
        'runs_off_bat': ['runs_off_bat', 'batsman_runs', 'runs_scored'],
        'extras': ['extras', 'extra_runs'],
        'wicket_type': ['wicket_type', 'dismissal_kind', 'wicket_kind'],
        'batter': ['batter', 'batsman', 'striker'],
        'bowler': ['bowler', 'bowler_name'],
        'ball': ['ball', 'over_ball'],
        'batting_team': ['batting_team', 'team'],
        'bowling_team': ['bowling_team', 'opponent']
    }
    
    for standard_name, variations in column_mappings.items():
        for var in variations:
            if var in df.columns and standard_name not in df.columns:
                df.rename(columns={var: standard_name}, inplace=True)
                break
    
    if 'ball' in df.columns:
        df['ball'] = pd.to_numeric(df['ball'], errors='coerce').fillna(0)
        df['over'] = df['ball'].astype(int) + 1
    else:
        df['over'] = 1
    
    if 'runs_off_bat' in df.columns and 'extras' in df.columns:
        df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)
        df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0)
        df['total_runs'] = df['runs_off_bat'] + df['extras']
    else:
        df['total_runs'] = 0
    
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 15, 21], 
                         labels=['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
    
    if 'wicket_type' in df.columns:
        df['is_wicket'] = df['wicket_type'].notna().astype(int)
    else:
        df['is_wicket'] = 0
    
    # Bowler types
    def get_bowler_type(bowler_name):
        if pd.isna(bowler_name): return 'Unknown'
        name = str(bowler_name).lower()
        
        if any(p in name for p in ['boult', 'arshdeep', 'starc', 'mustafizur']):
            return 'Left-Arm Pace'
        elif any(p in name for p in ['kuldeep', 'noor', 'shamsi']):
            return 'Left-Arm Wrist Spin'
        elif any(p in name for p in ['jadeja', 'axar', 'krunal']):
            return 'Left-Arm Orthodox'
        elif any(p in name for p in ['rashid', 'chahal', 'bishnoi', 'zampa']):
            return 'Right-Arm Leg Spin'
        elif any(p in name for p in ['ashwin', 'narine', 'chakravarthy']):
            return 'Right-Arm Off Spin'
        else:
            return 'Right-Arm Pace'
    
    # Ensure batter and bowler columns exist before applying function
    if 'batter' not in df.columns:
        df['batter'] = 'Unknown'
    if 'bowler' not in df.columns:
        df['bowler'] = 'Unknown'
        
    df['bowler_type'] = df['bowler'].apply(get_bowler_type)
    
    for col in ['batting_team', 'bowling_team', 'batter', 'bowler']:
        if col in df.columns: 
            df[col] = df[col].fillna('Unknown')
    
    return df

# Load data on startup
df = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/teams')
def get_teams():
    teams = sorted(df['batting_team'].unique().tolist())
    return jsonify(teams)

@app.route('/api/phase-analysis')
def phase_analysis():
    team = request.args.get('team')
    
    if not team:
        return jsonify({'error': 'Team parameter required'}), 400
    
    team_data = df[df['batting_team'] == team]
    phase_stats = team_data.groupby('phase', observed=False).agg({
        'total_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum',
        'runs_off_bat': lambda x: (x == 4).sum() + (x == 6).sum()
    }).reset_index()
    
    phase_stats.columns = ['phase', 'total_runs', 'balls', 'wickets', 'boundaries']
    phase_stats['run_rate'] = (phase_stats['total_runs'] / phase_stats['balls'] * 6).round(2)
    phase_stats['strike_rate'] = (phase_stats['total_runs'] / phase_stats['balls'] * 100).round(2)
    
    return jsonify(phase_stats.to_dict('records'))

@app.route('/api/top-batters')
def top_batters():
    team = request.args.get('team')
    n = int(request.args.get('n', 5))
    
    if not team:
        return jsonify({'error': 'Team parameter required'}), 400
    
    team_data = df[df['batting_team'] == team]
    stats = team_data.groupby('batter').agg({
        'runs_off_bat': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    
    stats = stats[stats['ball'] >= 30]
    stats['strike_rate'] = (stats['runs_off_bat'] / stats['ball'] * 100).round(2)
    stats['average'] = (stats['runs_off_bat'] / stats['is_wicket'].replace(0, 1)).round(2)
    stats = stats.sort_values('runs_off_bat', ascending=False).head(n)
    
    stats.columns = ['batter', 'runs', 'balls', 'dismissals', 'strike_rate', 'average']
    return jsonify(stats.to_dict('records'))

@app.route('/api/matchup-analysis')
def matchup_analysis():
    team = request.args.get('team')
    
    if not team:
        return jsonify({'error': 'Team parameter required'}), 400
    
    team_data = df[df['batting_team'] == team]
    batters = team_data.groupby('batter')['runs_off_bat'].sum().nlargest(10).index.tolist()
    bowler_types = df['bowler_type'].unique().tolist()
    
    matchup_data = []
    for batter in batters:
        batter_stats = {}
        for bowler_type in bowler_types:
            player_data = team_data[(team_data['batter'] == batter) & 
                                   (team_data['bowler_type'] == bowler_type)]
            if len(player_data) > 0:
                runs = player_data['runs_off_bat'].sum()
                balls = len(player_data)
                sr = (runs / balls * 100) if balls > 0 else 0
                batter_stats[bowler_type] = {
                    'runs': int(runs),
                    'balls': int(balls),
                    'strike_rate': round(sr, 2)
                }
            else:
                batter_stats[bowler_type] = {'runs': 0, 'balls': 0, 'strike_rate': 0}
        
        matchup_data.append({
            'batter': batter,
            'matchups': batter_stats
        })
    
    return jsonify({
        'batters': batters,
        'bowler_types': bowler_types,
        'data': matchup_data
    })

@app.route('/api/team-summary')
def team_summary():
    team = request.args.get('team')
    
    if not team:
        return jsonify({'error': 'Team parameter required'}), 400
    
    team_data = df[df['batting_team'] == team]
    
    summary = {
        'total_runs': int(team_data['total_runs'].sum()),
        'total_balls': int(len(team_data)),
        'total_matches': int(team_data['match_id'].nunique()),
        'run_rate': round(team_data['total_runs'].sum() / len(team_data) * 6, 2),
        'fours': int((team_data['runs_off_bat'] == 4).sum()),
        'sixes': int((team_data['runs_off_bat'] == 6).sum()),
        'wickets_lost': int(team_data['is_wicket'].sum()),
        'dot_balls': int((team_data['runs_off_bat'] == 0).sum()),
        'dot_ball_percentage': round((team_data['runs_off_bat'] == 0).sum() / len(team_data) * 100, 2)
    }
    
    return jsonify(summary)

@app.route('/api/over-progression')
def over_progression():
    team = request.args.get('team')
    
    if not team:
        return jsonify({'error': 'Team parameter required'}), 400
    
    team_data = df[df['batting_team'] == team]
    over_stats = team_data.groupby('over').agg({
        'total_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    
    over_stats['cumulative_runs'] = over_stats['total_runs'].cumsum()
    over_stats = over_stats[over_stats['over'] <= 20]
    
    return jsonify(over_stats.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
