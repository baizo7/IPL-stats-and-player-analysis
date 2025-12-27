import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import json
import uuid
import altair as alt
import streamlit.components.v1 as components
import tempfile

# Optional: Try to import manim for animations (not critical for main features)
try:
    from manim import *
    MANIM_AVAILABLE = True
except Exception:
    MANIM_AVAILABLE = False

# IPL Team Colors Dictionary
TEAM_COLORS = {
    'Mumbai Indians': '#004BA0',
    'Chennai Super Kings': '#FDB913',
    'Royal Challengers Bangalore': '#EC1C24',
    'Kolkata Knight Riders': '#3A225D',
    'Delhi Capitals': '#004C93',
    'Sunrisers Hyderabad': '#FF822A',
    'Punjab Kings': '#ED1B24',
    'Rajasthan Royals': '#254AA5',
    'Gujarat Titans': '#1C2841',
    'Lucknow Super Giants': '#D4AF37',
    'Rising Pune Supergiant': '#9F2C6C',
    'Kings XI Punjab': '#ED1B24',
    'Delhi Daredevils': '#004C93',
    'Deccan Chargers': '#6495ED',
    'Kochi Tuskers Kerala': '#9966CC',
    'Pune Warriors': '#2E3192',
    'Gujarat Lions': '#FF8C00'
}

# Set page configuration
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content wrapper */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Main content area */
    .main {
        background: transparent;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    h2 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem !important;
        margin-top: 2rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    h3 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.5rem !important;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
        text-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric cards with glassmorphism */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 1rem;
        color: #1e293b;
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4);
    }
    
    /* Columns with glassmorphism */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 0.5rem 0;
    }
    
    /* Column text color */
    [data-testid="column"] * {
        color: #1e293b !important;
    }
    
    [data-testid="column"] h1,
    [data-testid="column"] h2,
    [data-testid="column"] h3,
    [data-testid="column"] h4,
    [data-testid="column"] h5,
    [data-testid="column"] h6 {
        color: #1e293b !important;
    }
    
    [data-testid="column"] p,
    [data-testid="column"] span,
    [data-testid="column"] div,
    [data-testid="column"] .stMarkdown,
    [data-testid="column"] strong,
    [data-testid="column"] b {
        color: #1e293b !important;
    }
    
    /* Ensure all text elements are visible */
    [data-testid="column"] .element-container * {
        color: #1e293b !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons with gradient */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox label {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.95);
        color: #667eea;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        color: #1e293b !important;
    }
    
    .dataframe th,
    .dataframe td {
        color: #1e293b !important;
    }
    
    /* Dividers */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Stats viewer text - force black in white containers */
    .element-container {
        color: #1e293b;
    }
    
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown div,
    .stMarkdown strong,
    .stMarkdown b {
        color: inherit;
    }
    
    /* Text inside white/glass containers */
    [data-testid="column"] .stMarkdown,
    [data-testid="metric-container"] .stMarkdown,
    .stDataFrame,
    .stTable {
        color: #1e293b !important;
    }
    
    /* All elements inside columns */
    [data-testid="column"] .element-container,
    [data-testid="column"] .stMarkdown p,
    [data-testid="column"] .stMarkdown div,
    [data-testid="column"] .stMarkdown span {
        color: #1e293b !important;
    }
    
    /* Ensure plotly charts have dark text */
    .js-plotly-plot .plotly text {
        fill: #1e293b !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
    }
    
    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #667eea;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #1e293b;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation for elements */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    [data-testid="column"],
    [data-testid="metric-container"] {
        animation: fadeInUp 0.6s ease-out;
    }
        border: none;
        border-top: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Data Loading and Cleaning
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load and clean IPL data"""
    path = "ipl_data"
    csv_file = "all_ipl_matches.csv"
    
    # Check if processed file exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, low_memory=False)
    else:
        # Logic to load from folder or create sample
        if not os.path.exists(path):
            return create_sample_data()
            
        else:
            all_files = glob.glob(os.path.join(path, "*.csv"))
            if not all_files:
                return create_sample_data()
                
            df_list = []
            for filename in all_files:
                try:
                    df = pd.read_csv(filename, index_col=None, header=0, on_bad_lines='skip', encoding='utf-8')
                    if not df.empty:
                        df_list.append(df)
                except:
                    continue
            final_df = pd.concat(df_list, axis=0, ignore_index=True)
            df = final_df

    return clean_data(df)

def create_sample_data():
    # Fallback sample data generator
    teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders']
    n_records = 1000
    df = pd.DataFrame({
        'match_id': np.repeat(range(1, 11), 100),
        'batting_team': np.random.choice(teams, n_records),
        'bowling_team': np.random.choice(teams, n_records),
        'ball': np.tile(np.arange(0.1, 20.1, 0.1), n_records // 200 + 1)[:n_records],
        'runs_off_bat': np.random.choice([0, 1, 4, 6], n_records),
        'extras': 0,
        'wicket_type': np.random.choice([None, 'caught'], n_records, p=[0.95, 0.05]),
        'batter': 'Sample Batter',
        'bowler': 'Sample Bowler'
    })
    return clean_data(df)

def clean_data(df):
    df = df.copy()
    
    # Column mapping
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
                
    # Basic columns
    if 'ball' in df.columns:
        df['ball'] = pd.to_numeric(df['ball'], errors='coerce').fillna(0)
        df['over'] = df['ball'].astype(int) + 1
    elif 'over' in df.columns:
        df['over'] = pd.to_numeric(df['over'], errors='coerce').fillna(1).astype(int)
    else:
        df['over'] = 1
        
    # Runs
    if 'runs_off_bat' in df.columns and 'extras' in df.columns:
        df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)
        df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0)
        df['total_runs'] = df['runs_off_bat'] + df['extras']
    elif 'runs_off_bat' in df.columns:
        df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)
        df['total_runs'] = df['runs_off_bat']
    else:
        df['total_runs'] = 0
        
    # Phase
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 15, 21], labels=['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
    
    # Wickets
    if 'wicket_type' in df.columns:
        df['is_wicket'] = df['wicket_type'].notna().astype(int)
    else:
        df['is_wicket'] = 0
        
    # Bowler types
    if 'bowler' in df.columns:
        def get_bowler_type(bowler_name):
            if pd.isna(bowler_name): return 'Unknown'
            name = str(bowler_name).lower()
            
            left_arm_pacers = ['boult', 'arshdeep', 'natarajan', 'mustafizur', 'curran', 'starc']
            left_arm_wrist = ['kuldeep yadav', 'noor ahmad', 'tabraiz shamsi']
            left_arm_orthodox = ['jadeja', 'axar', 'krunal', 'shahbaz', 'santner']
            right_arm_leg = ['rashid', 'chahal', 'bishnoi', 'hasaranga', 'zampa']
            right_arm_off = ['ashwin', 'narine', 'chakravarthy', 'theekshana', 'livingstone']
            
            if any(p in name for p in left_arm_pacers):
                return 'Left-Arm Pace'
            elif any(p in name for p in left_arm_wrist):
                return 'Left-Arm Wrist Spin'
            elif any(p in name for p in left_arm_orthodox):
                return 'Left-Arm Orthodox'
            elif any(p in name for p in right_arm_leg):
                return 'Right-Arm Leg Spin'
            elif any(p in name for p in right_arm_off):
                return 'Right-Arm Off Spin'
            else:
                return 'Right-Arm Pace'

        df['bowler_type'] = df['bowler'].apply(get_bowler_type)
    else:
        df['bowler_type'] = 'Unknown'
        
    # Fill NaNs
    for col in ['batting_team', 'bowling_team', 'batter', 'bowler']:
        if col in df.columns: df[col] = df[col].fillna('Unknown')
        
    return df

# -----------------------------------------------------------------------------
# 2. Analysis Functions
# -----------------------------------------------------------------------------

def calculate_run_rate_by_phase(df, team):
    team_data = df[df['batting_team'] == team]
    phase_stats = team_data.groupby('phase', observed=False).agg({
        'total_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    phase_stats['run_rate'] = (phase_stats['total_runs'] / phase_stats['ball']) * 6
    phase_stats['wickets'] = phase_stats['is_wicket']
    phase_stats['avg_runs_per_ball'] = phase_stats['total_runs'] / phase_stats['ball']
    return phase_stats

def calculate_player_matchup(df, player, bowler_type):
    if bowler_type == 'All Types':
        player_data = df[df['batter'] == player]
    else:
        player_data = df[(df['batter'] == player) & (df['bowler_type'] == bowler_type)]
    
    if len(player_data) == 0: 
        return None
    
    balls = len(player_data)
    runs = int(player_data['runs_off_bat'].sum())
    dismissals = int(player_data['is_wicket'].sum())
    
    return {
        'balls_faced': int(balls),
        'runs_scored': runs,
        'dismissals': dismissals,
        'strike_rate': float((runs / balls) * 100 if balls > 0 else 0),
        'dismissal_rate': float((dismissals / balls) * 100 if balls > 0 else 0),
        'average': float(runs / dismissals if dismissals > 0 else runs)
    }

def get_top_batters(df, team, n=5):
    team_data = df[df['batting_team'] == team]
    stats = team_data.groupby('batter').agg({
        'runs_off_bat': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    stats = stats[stats['ball'] >= 30].sort_values('runs_off_bat', ascending=False).head(n)
    return stats

def generate_pitch_map_data(df, team=None, bowler_type=None, phase=None):
    """Generate pitch map data with ball positions"""
    import numpy as np
    
    filtered_df = df.copy()
    if team:
        filtered_df = filtered_df[filtered_df['batting_team'] == team]
    if bowler_type and bowler_type != 'All Types':
        filtered_df = filtered_df[filtered_df['bowler_type'] == bowler_type]
    if phase:
        filtered_df = filtered_df[filtered_df['phase'] == phase]
    
    if len(filtered_df) > 500:
        filtered_df = filtered_df.sample(500, random_state=42)
    
    np.random.seed(42)
    pitch_data = []
    
    for idx, row in filtered_df.iterrows():
        runs = int(row.get('runs_off_bat', 0))
        is_wicket = int(row.get('is_wicket', 0))
        
        if is_wicket:
            y = np.random.normal(8, 2)
            x = np.random.normal(0.3, 0.4)
            color = 'red'
            size = 12
        elif runs >= 6:
            y = np.random.choice([np.random.normal(4, 2), np.random.normal(18, 2)])
            x = np.random.normal(0, 0.6)
            color = 'purple'
            size = 14
        elif runs == 4:
            y = np.random.normal(10, 4)
            x = np.random.normal(0, 0.7)
            color = 'green'
            size = 10
        elif runs in [1, 2, 3]:
            y = np.random.normal(9, 3)
            x = np.random.normal(0, 0.5)
            color = 'blue'
            size = 6
        else:
            y = np.random.normal(8, 2.5)
            x = np.random.normal(0.2, 0.4)
            color = 'gray'
            size = 4
        
        x = float(max(-1.2, min(1.2, x)))
        y = float(max(0, min(22, y)))
        
        pitch_data.append({
            'x': x,
            'y': y,
            'runs': runs,
            'wicket': is_wicket,
            'color': color,
            'size': size,
            'batter': str(row.get('batter', 'Unknown')),
            'bowler': str(row.get('bowler', 'Unknown'))
        })
    
    return pitch_data

def generate_pitch_map_data_complete(df, team=None, bowler_type=None, phase=None):
    """Generate complete pitch map data with ball positions"""
    import numpy as np
    
    # Filter data
    filtered_df = df.copy()
    if team:
        filtered_df = filtered_df[filtered_df['batting_team'] == team]
    if bowler_type and bowler_type != 'All Types':
        filtered_df = filtered_df[filtered_df['bowler_type'] == bowler_type]
    if phase:
        filtered_df = filtered_df[filtered_df['phase'] == phase]
    
    # Sample data if too large (for performance)
    if len(filtered_df) > 500:
        filtered_df = filtered_df.sample(500, random_state=42)
    
    # Generate synthetic pitch positions
    np.random.seed(42)
    
    pitch_data = []
    for idx, row in filtered_df.iterrows():
        # Simulate pitch position based on outcome
        # X: -1 to 1 (left to right from bowler's perspective)
        # Y: 0 to 22 (pitch length in yards, 0 = bowler end, 22 = batter end)
        
        runs = row.get('runs_off_bat', 0)
        is_wicket = row.get('is_wicket', 0)
        
        # Good length balls (Y: 6-10 yards)
        # Short balls (Y: 0-6 yards)  
        # Full balls (Y: 10-16 yards)
        # Very full/yorkers (Y: 16-22 yards)
        
        if is_wicket:
            # Wicket balls tend to be good length, on or around off stump
            y = np.random.normal(8, 2)
            x = np.random.normal(0.3, 0.4)  # Around off stump
            color = 'red'
            size = 6
        elif runs >= 6:
            # Sixes - often short or very full
            y = np.random.choice([np.random.normal(4, 2), np.random.normal(18, 2)])
            x = np.random.normal(0, 0.6)
            color = 'purple'
            size = 7
        elif runs == 4:
            # Fours - various lengths
            y = np.random.normal(10, 4)
            x = np.random.normal(0, 0.7)
            color = 'green'
            size = 5
        elif runs in [1, 2, 3]:
            # Singles/doubles - good length
            y = np.random.normal(9, 3)
            x = np.random.normal(0, 0.5)
            color = 'blue'
            size = 3
        else:
            # Dot balls - good line and length
            y = np.random.normal(8, 2.5)
            x = np.random.normal(0.2, 0.4)
            color = 'gray'
            size = 2
        
        # Clamp values to pitch boundaries
        x = max(-1.2, min(1.2, x))
        y = max(0, min(22, y))
        
        pitch_data.append({
            'x': float(x),
            'y': float(y),
            'runs': int(runs),
            'wicket': int(is_wicket),
            'color': color,
            'size': size,
            'batter': str(row.get('batter', 'Unknown')),
            'bowler': str(row.get('bowler', 'Unknown'))
        })
    
    return pitch_data

def generate_wagon_wheel_data(df, team=None, batter=None, phase=None):
    """Generate accurate wagon wheel (shot direction) data based on ball position"""
    import numpy as np
    
    filtered_df = df.copy()
    if team:
        filtered_df = filtered_df[filtered_df['batting_team'] == team]
    if batter:
        filtered_df = filtered_df[filtered_df['batter'] == batter]
    if phase:
        filtered_df = filtered_df[filtered_df['phase'] == phase]
    
    filtered_df = filtered_df[filtered_df['runs_off_bat'] > 0]
    
    if len(filtered_df) > 300:
        filtered_df = filtered_df.sample(300, random_state=42)
    
    np.random.seed(42)
    wagon_data = []
    
    for idx, row in filtered_df.iterrows():
        runs = int(row.get('runs_off_bat', 0))
        
        # Determine shot zone based on runs and add realistic variation
        if runs == 6:
            # Sixes - long distances (65-95m), wider angle distribution
            angle = float(np.random.choice([
                np.random.uniform(-90, -45),   # Square leg/Fine leg
                np.random.uniform(-45, 0),     # Mid-wicket
                np.random.uniform(0, 45),      # Long-on/Straight
                np.random.uniform(45, 90),     # Long-off/Extra cover
                np.random.uniform(90, 135),    # Cover/Point
                np.random.uniform(135, 180),   # Third man/Backward point
            ]))
            distance = float(np.random.uniform(65, 95))
            color = 'red'
            size = 14
            
        elif runs == 4:
            # Fours - medium-long distances (50-70m), all around ground
            angle = float(np.random.choice([
                np.random.uniform(-135, -90),  # Fine leg
                np.random.uniform(-90, -45),   # Square leg
                np.random.uniform(-45, 0),     # Mid-wicket
                np.random.uniform(0, 30),      # Straight/Mid-on
                np.random.uniform(30, 60),     # Long-off
                np.random.uniform(60, 120),    # Extra cover/Cover
                np.random.uniform(120, 180),   # Point/Third man
            ]))
            distance = float(np.random.uniform(50, 70))
            color = 'red'  # Boundaries in red
            size = 11
            
        elif runs == 3:
            # Threes - medium distances (40-55m), good running
            angle = float(np.random.uniform(-120, 150))
            distance = float(np.random.uniform(40, 55))
            color = 'blue'
            size = 8
            
        elif runs == 2:
            # Twos - medium distances (30-50m)
            angle = float(np.random.uniform(-135, 135))
            distance = float(np.random.uniform(30, 50))
            color = 'orange'
            size = 7
            
        else:  # runs == 1
            # Singles - shorter distances (20-40m), all around
            angle = float(np.random.uniform(-180, 180))
            distance = float(np.random.uniform(20, 40))
            color = 'green'
            size = 6
        
        # Convert polar to cartesian coordinates
        rad = np.radians(angle)
        x = float(distance * np.sin(rad))  # Changed to sin for proper mapping
        y = float(distance * np.cos(rad))  # Changed to cos for proper mapping
        
        wagon_data.append({
            'x': x,
            'y': y,
            'angle': angle,
            'distance': distance,
            'runs': runs,
            'color': color,
            'size': size,
            'batter': str(row.get('batter', 'Unknown')),
            'bowler': str(row.get('bowler', 'Unknown'))
        })
    
    return wagon_data

def generate_stumps_view_data(df, team=None, phase=None):
    """Generate stumps view (behind bowler) data"""
    import numpy as np
    
    filtered_df = df.copy()
    if team:
        filtered_df = filtered_df[filtered_df['batting_team'] == team]
    if phase:
        filtered_df = filtered_df[filtered_df['phase'] == phase]
    
    if len(filtered_df) > 400:
        filtered_df = filtered_df.sample(400, random_state=42)
    
    np.random.seed(42)
    stumps_data = []
    
    for idx, row in filtered_df.iterrows():
        runs = int(row.get('runs_off_bat', 0))
        is_wicket = int(row.get('is_wicket', 0))
        
        if is_wicket:
            x = float(np.random.normal(0, 0.5))
            y = float(np.random.normal(1.5, 0.4))
            color = 'red'
            size = 6
        elif runs >= 6:
            x = float(np.random.normal(0, 0.7))
            y = float(np.random.choice([np.random.normal(2.2, 0.3), np.random.normal(0.8, 0.3)]))
            color = 'purple'
            size = 7
        elif runs == 4:
            x = float(np.random.normal(0, 0.9))
            y = float(np.random.normal(1.5, 0.5))
            color = 'green'
            size = 5
        elif runs in [1, 2, 3]:
            x = float(np.random.normal(0, 0.6))
            y = float(np.random.normal(1.5, 0.4))
            color = 'blue'
            size = 3
        else:
            x = float(np.random.normal(0, 0.4))
            y = float(np.random.normal(1.5, 0.3))
            color = 'gray'
            size = 2
        
        x = max(-2.5, min(2.5, x))
        y = max(0.2, min(2.8, y))
        
        stumps_data.append({
            'x': x,
            'y': y,
            'runs': runs,
            'wicket': is_wicket,
            'color': color,
            'size': size,
            'batter': str(row.get('batter', 'Unknown')),
            'bowler': str(row.get('bowler', 'Unknown'))
        })
    
    return stumps_data

def get_player_statistics(df, team, phase=None):
    """Get comprehensive player statistics"""
    team_data = df[df['batting_team'] == team].copy()
    
    if phase:
        team_data = team_data[team_data['phase'] == phase]
    
    batter_stats = team_data.groupby('batter').agg({
        'runs_off_bat': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    
    batter_stats = batter_stats[batter_stats['ball'] >= 30]
    batter_stats['batting_team'] = team
    batter_stats['strike_rate'] = (batter_stats['runs_off_bat'] / batter_stats['ball'] * 100).round(2)
    batter_stats['average'] = (batter_stats['runs_off_bat'] / batter_stats['is_wicket'].replace(0, 1)).round(2)
    
    fours_sixes = team_data[team_data['runs_off_bat'].isin([4, 6])].groupby(['batter', 'runs_off_bat']).size().unstack(fill_value=0)
    if 4 in fours_sixes.columns:
        batter_stats = batter_stats.merge(fours_sixes[[4]].rename(columns={4: 'fours'}), left_on='batter', right_index=True, how='left')
    else:
        batter_stats['fours'] = 0
    if 6 in fours_sixes.columns:
        batter_stats = batter_stats.merge(fours_sixes[[6]].rename(columns={6: 'sixes'}), left_on='batter', right_index=True, how='left')
    else:
        batter_stats['sixes'] = 0
    
    batter_stats['fours'] = batter_stats['fours'].fillna(0).astype(int)
    batter_stats['sixes'] = batter_stats['sixes'].fillna(0).astype(int)
    
    # Calculate highest score per player (per innings)
    innings_scores = team_data.groupby(['batter', 'match_id'])['runs_off_bat'].sum().reset_index()
    highest_scores = innings_scores.groupby('batter')['runs_off_bat'].max().reset_index()
    highest_scores.columns = ['batter', 'highest_score']
    batter_stats = batter_stats.merge(highest_scores, on='batter', how='left')
    batter_stats['highest_score'] = batter_stats['highest_score'].fillna(0).astype(int)
    
    batter_stats = batter_stats.sort_values('runs_off_bat', ascending=False).head(10)
    
    return batter_stats

# -----------------------------------------------------------------------------
# Altair Statistical Visualizations
# -----------------------------------------------------------------------------

def create_runs_distribution_chart(df, team, phase=None):
    """Create enhanced runs distribution analysis with dual visualization"""
    # Filter data for batting team
    team_data = df[df['batting_team'] == team].copy()
    
    # Apply phase filter if specified
    if phase:
        team_data = team_data[team_data['phase'] == phase]
    
    # Check if we have data
    if len(team_data) == 0:
        return None
    
    # Calculate runs distribution statistics
    runs_counts = team_data['runs_off_bat'].value_counts().reset_index()
    runs_counts.columns = ['runs', 'count']
    runs_counts = runs_counts.sort_values('runs')
    
    # Calculate percentages
    total_balls = len(team_data)
    runs_counts['percentage'] = ((runs_counts['count'] / total_balls) * 100).round(1)
    
    # Add cumulative percentage
    runs_counts['cumulative_pct'] = runs_counts['percentage'].cumsum().round(1)
    
    # Calculate total runs contribution per type
    runs_counts['total_runs_val'] = runs_counts['runs'] * runs_counts['count']
    total_runs_sum = runs_counts['total_runs_val'].sum()
    runs_counts['run_contribution_pct'] = ((runs_counts['total_runs_val'] / total_runs_sum) * 100).round(1) if total_runs_sum > 0 else 0
    
    # Create color mapping for different run types with vibrant gradients
    def get_run_color(runs):
        color_map = {
            0: '#7f8c8d',   # Cool gray for dots
            1: '#27ae60',   # Fresh green for singles
            2: '#3498db',   # Bright blue for twos
            3: '#f39c12',   # Vibrant orange for threes
            4: '#8e44ad',   # Deep purple for fours
            5: '#f1c40f',   # Golden yellow for fives
            6: '#c0392b',   # Deep red for sixes
        }
        return color_map.get(runs, '#34495e')
    
    runs_counts['color'] = runs_counts['runs'].apply(get_run_color)
    
    # Create run type labels and categories
    def get_run_label(runs):
        labels = {
            0: 'Dot Balls',
            1: 'Singles',
            2: 'Twos',
            3: 'Threes',
            4: 'Fours',
            5: 'Fives',
            6: 'Sixes'
        }
        return labels.get(runs, f'{int(runs)} Runs')
    
    def get_category(runs):
        if runs == 0:
            return 'Defensive'
        elif runs in [1, 2, 3]:
            return 'Rotating Strike'
        else:
            return 'Boundaries'
    
    runs_counts['label'] = runs_counts['runs'].apply(get_run_label)
    runs_counts['category'] = runs_counts['runs'].apply(get_category)
    
    # Calculate comprehensive statistics
    total_runs = int(team_data['runs_off_bat'].sum())
    avg_runs_per_ball = round(total_runs / total_balls, 2)
    dots = int(len(team_data[team_data['runs_off_bat'] == 0]))
    singles = int(len(team_data[team_data['runs_off_bat'] == 1]))
    twos = int(len(team_data[team_data['runs_off_bat'] == 2]))
    threes = int(len(team_data[team_data['runs_off_bat'] == 3]))
    fours = int(len(team_data[team_data['runs_off_bat'] == 4]))
    sixes = int(len(team_data[team_data['runs_off_bat'] == 6]))
    boundaries = fours + sixes
    
    # Calculate contribution percentages
    runs_from_boundaries = (fours * 4) + (sixes * 6)
    runs_from_singles = singles * 1
    runs_from_twos = twos * 2
    runs_from_threes = threes * 3
    
    boundary_contribution = round((runs_from_boundaries / total_runs) * 100, 1) if total_runs > 0 else 0
    rotation_contribution = round(((runs_from_singles + runs_from_twos + runs_from_threes) / total_runs) * 100, 1) if total_runs > 0 else 0
    
    dot_pct = round((dots / total_balls) * 100, 1)
    boundary_pct = round((boundaries / total_balls) * 100, 1)
    strike_rate = round((total_runs / total_balls) * 100, 2)
    
    # Get team color from TEAM_COLORS dictionary and create variations
    base_color = TEAM_COLORS.get(team, '#667eea')
    
    # Create color variations for different run types using team's base color
    import colorsys
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(r, g, b):
        return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))
    
    def create_color_variations(base_hex, num_variations=7):
        r, g, b = hex_to_rgb(base_hex)
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        variations = []
        for i in range(num_variations):
            # Vary brightness for different runs
            new_v = max(0.3, min(1.0, v + (i - 3) * 0.12))
            new_s = max(0.4, min(1.0, s + (i - 3) * 0.08))
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            variations.append(rgb_to_hex(new_r * 255, new_g * 255, new_b * 255))
        
        return variations
    
    color_variations = create_color_variations(base_color, 7)
    runs_counts['bar_color'] = runs_counts['runs'].apply(lambda x: color_variations[int(x) if x < 7 else 6])
    
    # Create the main bar chart with filled franchise colors
    bars = alt.Chart(runs_counts).mark_bar(
        cornerRadiusTopLeft=10,
        cornerRadiusTopRight=10,
        size=50,
        opacity=1.0,
        filled=True  # Explicitly set filled to True
    ).encode(
        x=alt.X('runs:O', 
                title='Runs Scored per Delivery',
                axis=alt.Axis(
                    labelFontSize=13, 
                    titleFontSize=15,
                    titleFontWeight='bold',
                    labelAngle=0,
                    grid=False
                )),
        y=alt.Y('count:Q', 
                title='Number of Deliveries',
                axis=alt.Axis(
                    labelFontSize=13, 
                    titleFontSize=15,
                    titleFontWeight='bold',
                    gridOpacity=0.3,
                    gridDash=[3, 3]
                )),
        color=alt.Color('bar_color:N', 
                       scale=None, 
                       legend=None),
        tooltip=[
            alt.Tooltip('label:N', title='📊 Ball Type'),
            alt.Tooltip('runs:Q', title='🏏 Runs per Ball'),
            alt.Tooltip('count:Q', title='⚾ Total Deliveries'),
            alt.Tooltip('percentage:Q', title='📈 Ball Frequency (%)', format='.1f'),
            alt.Tooltip('cumulative_pct:Q', title='📊 Cumulative (%)', format='.1f'),
            alt.Tooltip('category:N', title='🎯 Category'),
            alt.Tooltip('total_runs_val:Q', title='💰 Total Runs from Type', format=',d'),
            alt.Tooltip('run_contribution_pct:Q', title='💎 Run Contribution (%)', format='.1f')
        ]
    )
    
    # Add count labels on bars
    count_text = alt.Chart(runs_counts).mark_text(
        align='center',
        baseline='bottom',
        dy=-8,
        fontSize=15,
        fontWeight='bold',
        color='#000000'
    ).encode(
        x=alt.X('runs:O'),
        y=alt.Y('count:Q'),
        text=alt.Text('count:Q', format=',d')
    )
    
    # Add percentage labels
    pct_text = alt.Chart(runs_counts).mark_text(
        align='center',
        baseline='bottom',
        dy=-25,
        fontSize=13,
        fontWeight='bold',
        color='#000000'
    ).encode(
        x=alt.X('runs:O'),
        y=alt.Y('count:Q'),
        text=alt.Text('percentage:Q', format='.1f')
    )
    
    # Combine all layers for main bar chart
    main_chart = (bars + count_text + pct_text).properties(
        width=500,
        height=350,
        title={
            'text': [f'{team} - Runs Distribution'],
            'subtitle': [
                f'📊 {total_balls} balls • {total_runs} runs • SR: {strike_rate}',
                f'🎯 Dots: {dot_pct}% • Boundaries: {boundary_pct}%'
            ],
            'fontSize': 18,
            'subtitleFontSize': 11,
            'subtitleColor': '#555555',
            'anchor': 'start',
            'offset': 20
        }
    )
    
    # Secondary Chart: Runs Contribution Donut
    # Filter out 0 runs (dots) as they don't contribute to total runs
    runs_source_data = runs_counts[runs_counts['runs'] > 0].copy()
    
    donut = alt.Chart(runs_source_data).mark_arc(innerRadius=60, outerRadius=95, stroke='white', strokeWidth=2).encode(
        theta=alt.Theta(field="total_runs_val", type="quantitative", stack=True),
        color=alt.Color(field="bar_color", type="nominal", scale=None, legend=None),
        order=alt.Order("runs", sort="ascending"),
        tooltip=[
            alt.Tooltip('label:N', title='Run Type'),
            alt.Tooltip('total_runs_val:Q', title='Total Runs', format=',d'),
            alt.Tooltip('run_contribution_pct:Q', title='Contribution %', format='.1f')
        ]
    ).properties(
        width=240,
        height=240,
        title={
            'text': '💰 Runs Source',
            'subtitle': 'Contribution %',
            'fontSize': 15,
            'subtitleFontSize': 11,
            'subtitleColor': '#777777'
        }
    )
    
    # Add percentage labels to donut
    donut_text = alt.Chart(runs_source_data).mark_text(radius=125, fontSize=12, fontWeight='bold').encode(
        theta=alt.Theta(field="total_runs_val", type="quantitative", stack=True),
        text=alt.Text("run_contribution_pct:Q", format=".0f"),
        order=alt.Order("runs", sort="ascending"),
        color=alt.value("#000000")
    )
    
    # Add run type labels inside donut
    donut_labels = alt.Chart(runs_source_data).mark_text(radius=77, fontSize=11, fontWeight='bold').encode(
        theta=alt.Theta(field="total_runs_val", type="quantitative", stack=True),
        text=alt.Text("runs:Q"),
        order=alt.Order("runs", sort="ascending"),
        color=alt.value("white")
    )
    
    final_donut = donut + donut_text + donut_labels
    
    # Third Chart: Horizontal Bar Chart for Run Contribution
    horizontal_bars = alt.Chart(runs_source_data).mark_bar(
        cornerRadiusTopRight=8,
        cornerRadiusBottomRight=8,
        height=20
    ).encode(
        x=alt.X('run_contribution_pct:Q',
                title='Contribution to Total Runs (%)',
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    labelFontSize=11,
                    titleFontSize=13,
                    titleFontWeight='bold',
                    grid=True,
                    gridOpacity=0.3
                )),
        y=alt.Y('label:N',
                title='',
                sort='-x',
                axis=alt.Axis(
                    labelFontSize=12,
                    titleFontSize=13
                )),
        color=alt.Color('bar_color:N', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('label:N', title='Run Type'),
            alt.Tooltip('total_runs_val:Q', title='Total Runs', format=',d'),
            alt.Tooltip('run_contribution_pct:Q', title='Contribution %', format='.1f')
        ]
    ).properties(
        width=770,
        height=200,
        title={
            'text': '📊 Run Contribution Breakdown',
            'subtitle': 'Percentage contribution to total runs',
            'fontSize': 15,
            'subtitleFontSize': 11,
            'subtitleColor': '#777777'
        }
    )
    
    # Add percentage text labels on horizontal bars
    h_bar_text = alt.Chart(runs_source_data).mark_text(
        align='left',
        baseline='middle',
        dx=5,
        fontSize=12,
        fontWeight='bold',
        color='#000000'
    ).encode(
        x=alt.X('run_contribution_pct:Q'),
        y=alt.Y('label:N', sort='-x'),
        text=alt.Text('run_contribution_pct:Q', format='.1f')
    )
    
    horizontal_chart = horizontal_bars + h_bar_text
    
    # Combine charts: top row (bar + donut), bottom row (horizontal bar)
    top_row = alt.hconcat(main_chart, final_donut, spacing=30)
    combined_chart = alt.vconcat(top_row, horizontal_chart, spacing=25).resolve_scale(
        color='independent'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        gridOpacity=0.2,
        domainColor='#95a5a6',
        tickColor='#95a5a6'
    ).configure_view(
        strokeWidth=0,
        fill='#fafafa'
    ).interactive()
    
    return combined_chart

def create_strike_rate_comparison(df, phase=None):
    """Create strike rate comparison chart for top batters across teams"""
    if phase:
        data = df[df['phase'] == phase].copy()
    else:
        data = df.copy()
    
    batter_stats = data.groupby(['batter', 'batting_team']).agg({
        'runs_off_bat': 'sum',
        'ball': 'count'
    }).reset_index()
    
    batter_stats = batter_stats[batter_stats['ball'] >= 50]
    batter_stats['strike_rate'] = (batter_stats['runs_off_bat'] / batter_stats['ball'] * 100).round(2)
    batter_stats = batter_stats.sort_values('strike_rate', ascending=False).head(15)
    
    # Add team colors
    batter_stats['team_color'] = batter_stats['batting_team'].map(TEAM_COLORS)
    batter_stats['team_color'] = batter_stats['team_color'].fillna('#667eea')
    
    chart = alt.Chart(batter_stats).mark_bar().encode(
        x=alt.X('strike_rate:Q', title='Strike Rate', scale=alt.Scale(domain=[0, 250])),
        y=alt.Y('batter:N', sort='-x', title='Batter'),
        color=alt.Color('team_color:N', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('batter:N', title='Player'),
            alt.Tooltip('batting_team:N', title='Team'),
            alt.Tooltip('strike_rate:Q', title='Strike Rate', format='.2f'),
            alt.Tooltip('runs_off_bat:Q', title='Total Runs'),
            alt.Tooltip('ball:Q', title='Balls Faced')
        ]
    ).properties(
        width=700,
        height=500,
        title='Top 15 Batters by Strike Rate (min 50 balls)'
    ).interactive()
    
    return chart

def create_boundary_percentage_chart(df, teams, phase=None):
    """Create comprehensive boundary and dot ball analysis with franchise colors"""
    
    if len(teams) != 2:
        return None
    
    team1, team2 = teams[0], teams[1]
    
    # Get team data
    team1_data = df[df['batting_team'] == team1].copy()
    team2_data = df[df['batting_team'] == team2].copy()
    
    if phase:
        team1_data = team1_data[team1_data['phase'] == phase]
        team2_data = team2_data[team2_data['phase'] == phase]
    
    # Calculate stats for both teams
    def get_team_stats(team_data, team_name):
        total = len(team_data)
        if total == 0:
            return None
        
        stats = {
            'team': team_name,
            'sixes': len(team_data[team_data['runs_off_bat'] == 6]),
            'fours': len(team_data[team_data['runs_off_bat'] == 4]),
            'threes': len(team_data[team_data['runs_off_bat'] == 3]),
            'twos': len(team_data[team_data['runs_off_bat'] == 2]),
            'singles': len(team_data[team_data['runs_off_bat'] == 1]),
            'dots': len(team_data[team_data['runs_off_bat'] == 0])
        }
        
        stats['boundaries'] = stats['sixes'] + stats['fours']
        stats['total_balls'] = total
        
        # Calculate percentages
        for key in ['sixes', 'fours', 'threes', 'twos', 'singles', 'dots', 'boundaries']:
            stats[f'{key}_pct'] = round((stats[key] / total) * 100, 1)
        
        return stats
    
    t1_stats = get_team_stats(team1_data, team1)
    t2_stats = get_team_stats(team2_data, team2)
    
    if not t1_stats or not t2_stats:
        return None
    
    # Create dataframe for grouped bar chart
    chart_data = []
    
    # Get franchise colors
    team1_color = TEAM_COLORS.get(team1, '#667eea')
    team2_color = TEAM_COLORS.get(team2, '#764ba2')
    
    categories = [
        ('Sixes', 'sixes_pct', 'sixes'),
        ('Fours', 'fours_pct', 'fours'),
        ('Threes', 'threes_pct', 'threes'),
        ('Twos', 'twos_pct', 'twos'),
        ('Singles', 'singles_pct', 'singles'),
        ('Dots', 'dots_pct', 'dots')
    ]
    
    for label, pct_key, count_key in categories:
        chart_data.append({
            'category': label,
            'team': team1,
            'percentage': t1_stats[pct_key],
            'count': t1_stats[count_key],
            'team_color': team1_color
        })
        chart_data.append({
            'category': label,
            'team': team2,
            'percentage': t2_stats[pct_key],
            'count': t2_stats[count_key],
            'team_color': team2_color
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Split data by team and create separate charts with explicit colors
    team1_df = chart_df[chart_df['team'] == team1].copy()
    team2_df = chart_df[chart_df['team'] == team2].copy()
    
    # Create bar chart for team 1
    bars1 = alt.Chart(team1_df).mark_bar(
        size=45,
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6,
        stroke='#000000',
        strokeWidth=2.5,
        opacity=1.0,
        color=team1_color
    ).encode(
        x=alt.X('category:N',
                title='Ball Type',
                axis=alt.Axis(
                    labelFontSize=14,
                    titleFontSize=16,
                    labelAngle=0,
                    labelFontWeight='bold',
                    labelColor='#2c3e50'
                ),
                sort=['Sixes', 'Fours', 'Threes', 'Twos', 'Singles', 'Dots']),
        y=alt.Y('percentage:Q',
                title='Percentage of Deliveries (%)',
                axis=alt.Axis(
                    labelFontSize=13,
                    titleFontSize=16,
                    gridOpacity=0.3,
                    titleFontWeight='bold',
                    gridDash=[3, 3]
                ),
                scale=alt.Scale(domain=[0, 50])),
        xOffset=alt.value(-25),
        tooltip=[
            alt.Tooltip('team:N', title='🏏 Team'),
            alt.Tooltip('category:N', title='📊 Ball Type'),
            alt.Tooltip('count:Q', title='⚾ Count', format=',d'),
            alt.Tooltip('percentage:Q', title='📈 Percentage', format='.1f')
        ]
    )
    
    # Create bar chart for team 2
    bars2 = alt.Chart(team2_df).mark_bar(
        size=45,
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6,
        stroke='#000000',
        strokeWidth=2.5,
        opacity=1.0,
        color=team2_color
    ).encode(
        x=alt.X('category:N',
                sort=['Sixes', 'Fours', 'Threes', 'Twos', 'Singles', 'Dots']),
        y=alt.Y('percentage:Q',
                scale=alt.Scale(domain=[0, 50])),
        xOffset=alt.value(25),
        tooltip=[
            alt.Tooltip('team:N', title='🏏 Team'),
            alt.Tooltip('category:N', title='📊 Ball Type'),
            alt.Tooltip('count:Q', title='⚾ Count', format=',d'),
            alt.Tooltip('percentage:Q', title='📈 Percentage', format='.1f')
        ]
    )
    
    # Add value labels on bars for both teams
    text1 = alt.Chart(team1_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=11,
        fontWeight='bold',
        color='#000000'
    ).encode(
        x=alt.X('category:N', sort=['Sixes', 'Fours', 'Threes', 'Twos', 'Singles', 'Dots']),
        y=alt.Y('percentage:Q'),
        xOffset=alt.value(-25),
        text=alt.Text('percentage:Q', format='.1f')
    )
    
    text2 = alt.Chart(team2_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=11,
        fontWeight='bold',
        color='#000000'
    ).encode(
        x=alt.X('category:N', sort=['Sixes', 'Fours', 'Threes', 'Twos', 'Singles', 'Dots']),
        y=alt.Y('percentage:Q'),
        xOffset=alt.value(25),
        text=alt.Text('percentage:Q', format='.1f')
    )
    
    # Combine chart
    chart = (bars1 + bars2 + text1 + text2).properties(
        width=900,
        height=500,
        title={
            'text': 'Boundary & Dot Ball Analysis - Team Comparison',
            'subtitle': [
                f'{team1}: {t1_stats["boundaries_pct"]}% boundaries ({t1_stats["sixes"]} sixes, {t1_stats["fours"]} fours) | {t1_stats["dots_pct"]}% dots ({t1_stats["dots"]} balls)',
                f'{team2}: {t2_stats["boundaries_pct"]}% boundaries ({t2_stats["sixes"]} sixes, {t2_stats["fours"]} fours) | {t2_stats["dots_pct"]}% dots ({t2_stats["dots"]} balls)'
            ],
            'fontSize': 20,
            'subtitleFontSize': 12,
            'subtitleColor': '#555555',
            'anchor': 'start',
            'offset': 20
        }
    ).configure_axis(
        labelFontSize=13,
        titleFontSize=16,
        gridOpacity=0.25,
        domainColor='#2c3e50',
        tickColor='#2c3e50'
    ).configure_view(
        strokeWidth=0,
        fill='#fafafa'
    ).interactive()
    
    return chart

def create_runs_over_progression(df, team, phase=None):
    """Create enhanced runs progression over overs using Altair with franchise colors"""
    team_data = df[df['batting_team'] == team].copy()
    if phase:
        team_data = team_data[team_data['phase'] == phase]
    
    # Get team color
    team_color = TEAM_COLORS.get(team, '#667eea')
    
    # Group by over and calculate cumulative runs and wickets
    over_stats = team_data.groupby('over').agg({
        'runs_off_bat': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    over_stats.columns = ['over', 'runs_in_over', 'wickets']
    over_stats['cumulative_runs'] = over_stats['runs_in_over'].cumsum()
    over_stats['cumulative_wickets'] = over_stats['wickets'].cumsum()
    
    # Calculate run rate
    over_stats['run_rate'] = over_stats['runs_in_over']
    over_stats['avg_run_rate'] = over_stats['cumulative_runs'] / (over_stats['over'] + 1)
    
    # Create color scale for bars based on run rate
    over_stats['bar_color'] = over_stats['runs_in_over'].apply(
        lambda x: '#e74c3c' if x >= 15 else '#f39c12' if x >= 10 else '#3498db' if x >= 6 else '#95a5a6'
    )
    
    # Area chart for cumulative runs with gradient
    area = alt.Chart(over_stats).mark_area(
        line={'color': team_color, 'size': 4},
        color=alt.Gradient(
            gradient='linear',
            stops=[
                alt.GradientStop(color=team_color, offset=0),
                alt.GradientStop(color='white', offset=1)
            ],
            x1=0, x2=0, y1=0, y2=1
        ),
        opacity=0.7,
        interpolate='monotone'
    ).encode(
        x=alt.X('over:Q', 
                title='Over Number',
                axis=alt.Axis(
                    labelFontSize=12,
                    titleFontSize=14,
                    tickCount=20,
                    grid=True,
                    gridOpacity=0.2
                ),
                scale=alt.Scale(domain=[0, max(over_stats['over'].max() + 1, 20)])),
        y=alt.Y('cumulative_runs:Q', 
                title='Cumulative Runs',
                axis=alt.Axis(
                    labelFontSize=12,
                    titleFontSize=14,
                    grid=True,
                    gridOpacity=0.3
                )),
        tooltip=[
            alt.Tooltip('over:Q', title='Over', format='d'),
            alt.Tooltip('runs_in_over:Q', title='Runs in Over', format='d'),
            alt.Tooltip('cumulative_runs:Q', title='Total Runs', format='d'),
            alt.Tooltip('wickets:Q', title='Wickets in Over', format='d'),
            alt.Tooltip('cumulative_wickets:Q', title='Total Wickets', format='d'),
            alt.Tooltip('avg_run_rate:Q', title='Avg Run Rate', format='.2f')
        ]
    )
    
    # Bar chart for runs per over with color coding
    bars = alt.Chart(over_stats).mark_bar(
        opacity=0.6,
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        x=alt.X('over:Q'),
        y=alt.Y('runs_in_over:Q', 
                title='Runs per Over',
                axis=alt.Axis(
                    labelFontSize=11,
                    titleFontSize=13,
                    orient='right'
                )),
        color=alt.Color('bar_color:N', 
                       scale=None,
                       legend=None),
        tooltip=[
            alt.Tooltip('over:Q', title='Over', format='d'),
            alt.Tooltip('runs_in_over:Q', title='Runs', format='d')
        ]
    )
    
    # Points for cumulative line
    points = alt.Chart(over_stats).mark_circle(
        size=100,
        opacity=1,
        color=team_color,
        stroke='white',
        strokeWidth=2
    ).encode(
        x=alt.X('over:Q'),
        y=alt.Y('cumulative_runs:Q'),
        tooltip=[
            alt.Tooltip('over:Q', title='Over', format='d'),
            alt.Tooltip('cumulative_runs:Q', title='Total Runs', format='d')
        ]
    )
    
    # Wicket markers
    wicket_overs = over_stats[over_stats['wickets'] > 0].copy()
    wickets_layer = alt.Chart(wicket_overs).mark_point(
        shape='triangle-down',
        size=200,
        filled=True,
        color='#e74c3c',
        stroke='white',
        strokeWidth=2
    ).encode(
        x=alt.X('over:Q'),
        y=alt.Y('cumulative_runs:Q'),
        tooltip=[
            alt.Tooltip('over:Q', title='Over', format='d'),
            alt.Tooltip('wickets:Q', title='Wickets Lost', format='d'),
            alt.Tooltip('cumulative_runs:Q', title='Score', format='d')
        ]
    )
    
    # Text labels for high-scoring overs (15+ runs) - removed for cleaner look
    
    # Combine all layers
    chart = (area + bars + points + wickets_layer).properties(
        width=800,
        height=450,
        title=alt.TitleParams(
            text=f'{team} - Runs Progression',
            fontSize=18,
            fontWeight='bold',
            anchor='start',
            color='#2c3e50'
        )
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelColor='#2c3e50',
        titleColor='#2c3e50',
        gridColor='#e0e0e0'
    ).interactive()
    
    return chart

def create_wicket_timeline(df, bowling_team, phase=None):
    """Create wicket fall timeline using Altair"""
    team_data = df[df['bowling_team'] == bowling_team].copy()
    if phase:
        team_data = team_data[team_data['phase'] == phase]
    
    wickets = team_data[team_data['is_wicket'] == 1].copy()
    
    if wickets.empty:
        return None
    
    wickets['wicket_num'] = range(1, len(wickets) + 1)
    
    chart = alt.Chart(wickets).mark_circle(size=200).encode(
        x=alt.X('over:Q', title='Over', scale=alt.Scale(domain=[0, 20])),
        y=alt.Y('wicket_num:Q', title='Wicket Number'),
        color=alt.Color('wicket_type:N', title='Dismissal Type', scale=alt.Scale(scheme='set2')),
        tooltip=[
            alt.Tooltip('over:Q', title='Over'),
            alt.Tooltip('batter:N', title='Batter Out'),
            alt.Tooltip('bowler:N', title='Bowler'),
            alt.Tooltip('wicket_type:N', title='Dismissal')
        ]
    ).properties(
        width=700,
        height=400,
        title=f'{bowling_team} - Wicket Fall Timeline'
    ).interactive()
    
    return chart

def create_bowler_economy_chart(df, team, phase=None):
    """Create comprehensive bowler economy rate analysis from scratch"""
    # Filter data for bowling team
    team_data = df[df['bowling_team'] == team].copy()
    
    # Apply phase filter if specified
    if phase:
        team_data = team_data[team_data['phase'] == phase]
    
    # Check if we have data
    if len(team_data) == 0:
        return None
    
    # Calculate comprehensive bowling statistics per bowler
    bowler_stats = team_data.groupby('bowler').agg({
        'runs_off_bat': 'sum',
        'extras': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    
    # Calculate additional metrics using direct filtering
    bowler_list = []
    for bowler in bowler_stats['bowler'].unique():
        bowler_balls = team_data[team_data['bowler'] == bowler]
        
        # Count dots (no runs and no extras)
        dots = len(bowler_balls[(bowler_balls['runs_off_bat'] == 0) & (bowler_balls['extras'] == 0)])
        
        # Count boundaries (4s and 6s)
        boundaries = len(bowler_balls[(bowler_balls['runs_off_bat'] == 4) | (bowler_balls['runs_off_bat'] == 6)])
        
        # Count sixes specifically
        sixes = len(bowler_balls[bowler_balls['runs_off_bat'] == 6])
        
        # Count fours specifically
        fours = len(bowler_balls[bowler_balls['runs_off_bat'] == 4])
        
        bowler_list.append({
            'bowler': bowler,
            'dots': dots,
            'boundaries': boundaries,
            'sixes': sixes,
            'fours': fours
        })
    
    # Merge additional statistics
    additional_stats = pd.DataFrame(bowler_list)
    bowler_stats = bowler_stats.merge(additional_stats, on='bowler', how='left')
    
    # Filter bowlers with minimum 24 balls (4 overs)
    bowler_stats = bowler_stats[bowler_stats['ball'] >= 24].copy()
    
    if len(bowler_stats) == 0:
        return None
    
    # Calculate derived metrics
    bowler_stats['overs'] = (bowler_stats['ball'] / 6).round(1)
    bowler_stats['total_runs'] = bowler_stats['runs_off_bat'] + bowler_stats['extras']
    bowler_stats['economy'] = (bowler_stats['total_runs'] / bowler_stats['overs']).round(2)
    
    # Calculate strike rate (balls per wicket)
    bowler_stats['strike_rate'] = bowler_stats.apply(
        lambda x: round(x['ball'] / x['is_wicket'], 1) if x['is_wicket'] > 0 else 999.0, axis=1
    )
    
    # Calculate bowling average (runs per wicket)
    bowler_stats['average'] = bowler_stats.apply(
        lambda x: round(x['total_runs'] / x['is_wicket'], 2) if x['is_wicket'] > 0 else 999.0, axis=1
    )
    
    # Calculate percentages
    bowler_stats['dot_percentage'] = ((bowler_stats['dots'] / bowler_stats['ball']) * 100).round(1)
    bowler_stats['boundary_percentage'] = ((bowler_stats['boundaries'] / bowler_stats['ball']) * 100).round(1)
    
    # Sort by economy rate and select top 12 bowlers
    bowler_stats = bowler_stats.sort_values('economy').head(12)
    
    # Create the base chart
    base = alt.Chart(bowler_stats).encode(
        y=alt.Y('bowler:N', 
                sort=alt.EncodingSortField(field='economy', order='ascending'),
                title='Bowler',
                axis=alt.Axis(labelLimit=200, labelFontSize=11))
    )
    
    # Create horizontal bars with color gradient
    bars = base.mark_bar(
        cornerRadiusTopLeft=10,
        cornerRadiusTopRight=10,
        size=30,
        opacity=0.9
    ).encode(
        x=alt.X('economy:Q', 
                title='Economy Rate (Runs per Over)', 
                scale=alt.Scale(domain=[0, max(15, bowler_stats['economy'].max() + 1)])),
        color=alt.Color('economy:Q',
                       scale=alt.Scale(
                           domain=[4, 6, 8, 10, 12, 15],
                           range=['#00ff00', '#7fff00', '#ffff00', '#ffa500', '#ff6347', '#ff0000']
                       ),
                       legend=alt.Legend(
                           title='Economy Rate',
                           orient='right',
                           titleFontSize=12,
                           labelFontSize=11
                       )),
        tooltip=[
            alt.Tooltip('bowler:N', title='🏏 Bowler'),
            alt.Tooltip('economy:Q', title='💰 Economy', format='.2f'),
            alt.Tooltip('is_wicket:Q', title='🎯 Wickets'),
            alt.Tooltip('average:Q', title='📊 Average', format='.2f'),
            alt.Tooltip('strike_rate:Q', title='⚡ Strike Rate', format='.1f'),
            alt.Tooltip('overs:Q', title='⏱️ Overs', format='.1f'),
            alt.Tooltip('total_runs:Q', title='🏃 Runs Conceded'),
            alt.Tooltip('dot_percentage:Q', title='⚫ Dot %', format='.1f'),
            alt.Tooltip('boundary_percentage:Q', title='🔴 Boundary %', format='.1f'),
            alt.Tooltip('fours:Q', title='4️⃣ Fours'),
            alt.Tooltip('sixes:Q', title='6️⃣ Sixes')
        ]
    )
    
    # Add text labels showing economy rate on bars
    text_labels = base.mark_text(
        align='left',
        baseline='middle',
        dx=5,
        fontSize=12,
        fontWeight='bold',
        color='#333333'
    ).encode(
        x=alt.X('economy:Q'),
        text=alt.Text('economy:Q', format='.2f')
    )
    
    # Add wickets count as secondary text
    wicket_labels = base.mark_text(
        align='left',
        baseline='middle',
        dx=50,
        fontSize=10,
        color='#666666',
        fontStyle='italic'
    ).encode(
        x=alt.X('economy:Q'),
        text=alt.Text('is_wicket:Q', format='W')
    )
    
    # Combine all layers
    chart = (bars + text_labels + wicket_labels).properties(
        width=800,
        height=550,
        title={
            'text': [f'{team} - Bowler Economy Rate Analysis'],
            'subtitle': [
                'Best economy rates ranked (minimum 4 overs bowled)',
                'Lower economy = Better bowling performance'
            ],
            'fontSize': 20,
            'fontWeight': 'bold',
            'subtitleFontSize': 13,
            'subtitleColor': '#555555',
            'anchor': 'start',
            'offset': 20
        }
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=14,
        titleFontWeight='bold',
        gridOpacity=0.3
    ).configure_view(
        strokeWidth=0,
        fill='#fafafa'
    ).configure_legend(
        titleFontSize=12,
        labelFontSize=11,
        symbolSize=200,
        padding=10
    ).interactive()
    
    return chart

# -----------------------------------------------------------------------------
# Manim Cricket Ball Animation (Commented out - not used in deployment)
# -----------------------------------------------------------------------------
# This section contains manim animation code that requires additional dependencies
# and is not used in the current Streamlit deployment
# class CricketBallTrajectory(Scene):
#     """Manim animation showing cricket ball trajectory from stumps view"""
#     
#     def construct(self):
#         # Set background
#         self.camera.background_color = "#1a1a2e"
#         
#         # Title
#         title = Text("Cricket Ball Trajectory - Stumps View", font_size=36, color=WHITE)
#         title.to_edge(UP)
#         self.play(Write(title), run_time=1)
#         self.wait(0.5)
#         
#         # Create pitch outline (stumps view - looking down the pitch)
#         pitch_width = 6
#         pitch_height = 8
#         pitch = Rectangle(
#             width=pitch_width, 
#             height=pitch_height,
#             color="#8B6F47",
#             fill_opacity=0.3,
#             stroke_color=WHITE,
#             stroke_width=2
#         )
#         pitch.shift(DOWN * 0.5)
#         
#         # Stumps at bottom
#         stump_positions = [-0.3, 0, 0.3]
#         stumps = VGroup()
#         for x_pos in stump_positions:
#             stump = Rectangle(
#                 width=0.15,
#                 height=0.8,
#                 color=WHITE,
#                 fill_opacity=1,
#                 stroke_width=2
#             )
#             stump.move_to([x_pos, -pitch_height/2 + 0.4, 0])
#             stumps.add(stump)
#         
#         # Bails on top of stumps
#         bail = Rectangle(width=1, height=0.1, color=WHITE, fill_opacity=1)
#         bail.move_to([0, -pitch_height/2 + 0.85, 0])
#         
#         # Draw pitch and stumps
#         self.play(Create(pitch), run_time=0.8)
#         self.play(Create(stumps), Create(bail), run_time=0.6)
#         
#         # Create grid for line and length zones
#         grid_lines = VGroup()
#         
#         # Vertical lines (line)
#         for x in np.linspace(-pitch_width/2, pitch_width/2, 5):
#             line = Line(
#                 start=[x, -pitch_height/2, 0],
#                 end=[x, pitch_height/2, 0],
#                 color=BLUE_E,
#                 stroke_width=1,
#                 stroke_opacity=0.3
#             )
#             grid_lines.add(line)
#         
#         # Horizontal lines (length)
#         for y in np.linspace(-pitch_height/2, pitch_height/2, 7):
#             line = Line(
#                 start=[-pitch_width/2, y, 0],
#                 end=[pitch_width/2, y, 0],
#                 color=BLUE_E,
#                 stroke_width=1,
#                 stroke_opacity=0.3
#             )
#             grid_lines.add(line)
#         
#         self.play(Create(grid_lines), run_time=0.8)
#         
#         # Zone labels
#         zone_labels = VGroup(
#             Text("Wide", font_size=20, color=RED).move_to([-pitch_width/2 + 0.8, 2, 0]),
#             Text("Off Stump", font_size=20, color=YELLOW).move_to([-1, 2, 0]),
#             Text("Middle", font_size=20, color=GREEN).move_to([0, 2, 0]),
#             Text("Leg Side", font_size=20, color=BLUE).move_to([1.5, 2, 0]),
#         )
#         self.play(FadeIn(zone_labels), run_time=0.6)
#         
#         # Length zones on the right
#         length_labels = VGroup(
#             Text("Full", font_size=18, color=GREEN).move_to([pitch_width/2 + 1.5, -2.5, 0]),
#             Text("Good Length", font_size=18, color=YELLOW).move_to([pitch_width/2 + 1.5, 0, 0]),
#             Text("Short", font_size=18, color=RED).move_to([pitch_width/2 + 1.5, 2.5, 0]),
#         )
#         self.play(FadeIn(length_labels), run_time=0.6)
#         
#         # Animate multiple ball trajectories
#         ball_data = [
#             {"start": [0, 4, 0], "end": [0, -3, 0], "color": GREEN, "label": "Yorker"},
#             {"start": [-1.5, 4, 0], "end": [-1.5, -1, 0], "color": YELLOW, "label": "Off Stump"},
#             {"start": [1, 4, 0], "end": [1.2, 1, 0], "color": BLUE, "label": "Leg Side"},
#             {"start": [-2.5, 4, 0], "end": [-2.5, 2, 0], "color": RED, "label": "Wide"},
#             {"start": [0.5, 4, 0], "end": [0.5, 0, 0], "color": ORANGE, "label": "Good Length"},
#         ]
#         
#         for i, ball_info in enumerate(ball_data):
#             # Create ball
#             ball = Circle(radius=0.15, color=ball_info["color"], fill_opacity=1)
#             ball.set_sheen(-0.4, DR)
#             ball.move_to(ball_info["start"])
#             
#             # Ball label
#             label = Text(ball_info["label"], font_size=16, color=ball_info["color"])
#             label.next_to(ball, UP, buff=0.2)
#             
#             # Trajectory line
#             trajectory = Line(
#                 start=ball_info["start"],
#                 end=ball_info["end"],
#                 color=ball_info["color"],
#                 stroke_width=3,
#                 stroke_opacity=0.5
#             )
#             
#             # Animate
#             self.play(
#                 Create(trajectory),
#                 FadeIn(ball),
#                 FadeIn(label),
#                 run_time=0.4
#             )
#             
#             self.play(
#                 ball.animate.move_to(ball_info["end"]),
#                 label.animate.next_to(ball_info["end"], UP, buff=0.2),
#                 run_time=1.2,
#                 rate_func=rush_into
#             )
#             
#             # Add impact marker
#             impact = Circle(radius=0.2, color=ball_info["color"], stroke_width=4)
#             impact.move_to(ball_info["end"])
#             self.play(
#                 Flash(impact, color=ball_info["color"], flash_radius=0.4),
#                 run_time=0.3
#             )
#             
#             self.wait(0.2)
#         
#         # Final hold
#         self.wait(2)
#         
#         # Fade out
#         self.play(
#             *[FadeOut(mob) for mob in self.mobjects],
#             run_time=1
#         )
# 
# def create_manim_animation(output_path="cricket_animation.mp4"):
#     # Generate Manim animation and return the video path
#     try:
#         import shutil
#         from manim import config, tempconfig
#         
#         # Create output directory if it doesn't exist
#         output_dir = os.path.dirname(output_path) or "."
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         
#         # Use tempconfig context manager to properly set configuration
#         with tempconfig({
#             "pixel_height": 720,
#             "pixel_width": 1280,
#             "frame_rate": 30,
#             "background_color": "#1a1a2e",
#             "output_file": "CricketBallTrajectory",
#             "quality": "medium_quality",
#             "preview": False,
#             "write_to_movie": True,
#         }):
#             # Render the scene
#             scene = CricketBallTrajectory()
#             scene.render()
#             
#             # The rendered file will be in the default media directory
#             # Get the output file path from the scene
#             if hasattr(scene.renderer, 'file_writer') and scene.renderer.file_writer:
#                 output_file = scene.renderer.file_writer.movie_file_path
#                 if output_file and os.path.exists(output_file):
#                     shutil.copy(output_file, output_path)
#                     return output_path
#             
#             # Fallback: search for the video in default locations
#             home_dir = os.path.expanduser("~")
#             possible_dirs = [
#                 os.path.join(home_dir, "media", "videos"),
#                 os.path.join(".", "media", "videos"),
#                 os.path.join(os.getcwd(), "media", "videos"),
#             ]
#             
#             for base_dir in possible_dirs:
#                 if os.path.exists(base_dir):
#                     video_files = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)
#                     if video_files:
#                         # Get the most recent video file
#                         latest_video = max(video_files, key=os.path.getctime)
#                         shutil.copy(latest_video, output_path)
#                         return output_path
#             
#             return None
#             
#     except Exception as e:
#         st.error(f"Error creating animation: {str(e)}")
#         import traceback
#         st.code(traceback.format_exc())
#         return None

# -----------------------------------------------------------------------------
# 3. Three.js 3D Visualization Helper
# -----------------------------------------------------------------------------

def render_threejs_chart(data, chart_type, title, width=600, height=400):
    div_id = f"chart_{uuid.uuid4().hex[:8]}"
    data_json = json.dumps(data)
    
    if chart_type == 'grouped_bar_3d':
        script = f"""
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);
        
        const camera = new THREE.PerspectiveCamera(60, {width}/{height}, 0.1, 1000);
        camera.position.set(15, 15, 15);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize({width}, {height});
        document.getElementById('{div_id}').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        scene.add(directionalLight);
        
        let maxValue = 0;
        data.forEach(cat => {{
            cat.values.forEach(val => {{
                if (val.value > maxValue) maxValue = val.value;
            }});
        }});
        
        const barWidth = 1.5;
        const barDepth = 1.5;
        const spacing = 5;
        const groupSpacing = 2;
        const colors = [0xFDB913, 0x004BA0];
        
        data.forEach((category, catIndex) => {{
            category.values.forEach((val, teamIndex) => {{
                const height = (val.value / maxValue) * 10;
                const geometry = new THREE.BoxGeometry(barWidth, height, barDepth);
                const material = new THREE.MeshPhongMaterial({{ color: colors[teamIndex], shininess: 100 }});
                const bar = new THREE.Mesh(geometry, material);
                
                const xPos = catIndex * spacing - (data.length * spacing / 2);
                const zPos = teamIndex * groupSpacing - 1;
                bar.position.set(xPos, height/2, zPos);
                
                bar.userData = {{
                    team: val.label,
                    phase: category.category,
                    value: val.value.toFixed(2),
                    balls: val.balls || 0,
                    wickets: val.wickets || 0
                }};
                
                scene.add(bar);
            }});
        }});
        
        const gridHelper = new THREE.GridHelper(30, 30, 0x888888, 0xdddddd);
        scene.add(gridHelper);
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        """
    
    elif chart_type == 'bar_3d':
        script = f"""
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);
        
        const camera = new THREE.PerspectiveCamera(60, {width}/{height}, 0.1, 1000);
        camera.position.set(10, 10, 15);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize({width}, {height});
        document.getElementById('{div_id}').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        scene.add(directionalLight);
        
        const maxValue = Math.max(...data.map(d => d.value));
        const barWidth = 1.5;
        const spacing = 3;
        
        data.forEach((item, index) => {{
            const height = (item.value / maxValue) * 10;
            const hue = (index / data.length) * 0.7;
            
            const geometry = new THREE.BoxGeometry(barWidth, height, barWidth);
            const material = new THREE.MeshPhongMaterial({{ 
                color: new THREE.Color().setHSL(hue, 0.7, 0.5),
                shininess: 100
            }});
            const bar = new THREE.Mesh(geometry, material);
            
            bar.position.set((index - data.length/2) * spacing, height/2, 0);
            bar.userData = {{
                player: item.label,
                strikeRate: item.value.toFixed(2),
                balls: item.balls || 0,
                runs: item.runs || 0,
                dismissals: item.dismissals || 0
            }};
            
            scene.add(bar);
        }});
        
        const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0xdddddd);
        scene.add(gridHelper);
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        """
    
    elif chart_type == 'pie_3d':
        script = f"""
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);
        
        const camera = new THREE.PerspectiveCamera(60, {width}/{height}, 0.1, 1000);
        camera.position.set(0, 8, 12);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize({width}, {height});
        document.getElementById('{div_id}').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        scene.add(directionalLight);
        
        const total = data.reduce((sum, item) => sum + item.value, 0);
        let currentAngle = 0;
        const innerRadius = 2;
        const outerRadius = 5;
        const depth = 1;
        
        data.forEach((item, index) => {{
            const angle = (item.value / total) * Math.PI * 2;
            const hue = (index / data.length);
            
            const shape = new THREE.Shape();
            const startAngle = currentAngle;
            const endAngle = currentAngle + angle;
            
            shape.moveTo(outerRadius * Math.cos(startAngle), outerRadius * Math.sin(startAngle));
            shape.absarc(0, 0, outerRadius, startAngle, endAngle, false);
            shape.lineTo(innerRadius * Math.cos(endAngle), innerRadius * Math.sin(endAngle));
            shape.absarc(0, 0, innerRadius, endAngle, startAngle, true);
            
            const geometry = new THREE.ExtrudeGeometry(shape, {{
                depth: depth,
                bevelEnabled: true,
                bevelThickness: 0.1,
                bevelSize: 0.1,
                bevelSegments: 2
            }});
            
            const material = new THREE.MeshPhongMaterial({{ 
                color: new THREE.Color().setHSL(hue, 0.8, 0.6),
                shininess: 100
            }});
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.z = -depth/2;
            mesh.userData = {{
                label: item.label,
                value: item.value,
                percentage: ((item.value / total) * 100).toFixed(1)
            }};
            
            scene.add(mesh);
            currentAngle = endAngle;
        }});
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        """
    else:
        script = ""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ margin: 0; padding: 20px; font-family: sans-serif; }}
            #title {{ text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
            #{div_id} {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div id="title">{title}</div>
        <div id="{div_id}"></div>
        <script>
            const data = {data_json};
            {script}
        </script>
    </body>
    </html>
    """
    return html

def render_pitch_map(data, title, width=800, height=600):
    """Render advanced 3D pitch map with realistic cricket pitch background"""
    div_id = f"pitch_{uuid.uuid4().hex[:8]}"
    data_json = json.dumps(data)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ margin: 0; padding: 15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .pitch-container-{div_id} {{ 
                position: relative;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                overflow: visible;
                isolation: isolate;
            }}
            .pitch-title-{div_id} {{ 
                text-align: center; 
                font-size: 20px; 
                font-weight: bold; 
                margin-bottom: 15px;
                color: white;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .pitch-legend-{div_id} {{ 
                position: absolute; 
                top: 70px; 
                right: 5px; 
                background: rgba(255,255,255,0.98); 
                padding: 10px 12px; 
                border-radius: 8px; 
                font-size: 11px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.5); 
                z-index: 1000;
                border: 2px solid #1e3c72;
                max-width: 130px;
            }}
            .legend-item-{div_id} {{ 
                display: flex; 
                align-items: center; 
                margin: 4px 0; 
                font-weight: 500;
            }}
            .legend-color-{div_id} {{ 
                width: 14px; 
                height: 14px; 
                border-radius: 50%; 
                margin-right: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                flex-shrink: 0;
            }}
            .controls-{div_id} {{
                position: absolute;
                top: 70px;
                left: 5px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                z-index: 1000;
                max-width: 110px;
            }}
            .view-btn-{div_id} {{
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid #1e3c72;
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                font-size: 11px;
                transition: all 0.3s ease;
                color: #1e3c72;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                white-space: nowrap;
            }}
            .view-btn-{div_id}:hover {{
                background: #1e3c72;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(30, 60, 114, 0.6);
            }}
            #{div_id} {{ 
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }}
        </style>
    </head>
    <body>
        <div class="pitch-container-{div_id}">
            <div class="pitch-title-{div_id}">{title}</div>
            <div class="controls-{div_id}">
                <button class="view-btn-{div_id}" onclick="setView_{div_id}('top')">📍 Top View</button>
                <button class="view-btn-{div_id}" onclick="setView_{div_id}('bowler')">🎯 Bowler End</button>
                <button class="view-btn-{div_id}" onclick="setView_{div_id}('batter')">🏏 Batter End</button>
                <button class="view-btn-{div_id}" onclick="setView_{div_id}('side')">👁️ Side View</button>
                <button class="view-btn-{div_id}" onclick="setView_{div_id}('reset')">🔄 Reset</button>
            </div>
            <div class="pitch-legend-{div_id}">
                <div style="font-weight: bold; margin-bottom: 6px; color: #1e3c72; font-size: 12px;">Outcomes</div>
                <div class="legend-item-{div_id}"><div class="legend-color-{div_id}" style="background: #808080;"></div><span>Dot (0)</span></div>
                <div class="legend-item-{div_id}"><div class="legend-color-{div_id}" style="background: #2196f3;"></div><span>1-3 runs</span></div>
                <div class="legend-item-{div_id}"><div class="legend-color-{div_id}" style="background: #00ff00;"></div><span>Four</span></div>
                <div class="legend-item-{div_id}"><div class="legend-color-{div_id}" style="background: #9c27b0;"></div><span>Six</span></div>
                <div class="legend-item-{div_id}"><div class="legend-color-{div_id}" style="background: #ff0000;"></div><span>Wicket</span></div>
            </div>
            <div id="{div_id}"></div>
        </div>
        <script>
        (function() {{
            const pitchData = {data_json};
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);
            // Removed fog for better visibility of stadium
            
            const camera = new THREE.PerspectiveCamera(60, {width}/{height}, 0.1, 1000);
            camera.position.set(0, 18, 22);
            camera.lookAt(0, 0, 11);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize({width}, {height});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.0;
            document.getElementById('{div_id}').appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.minDistance = 10;
            controls.maxDistance = 80;
            controls.maxPolarAngle = Math.PI / 2.1;
            controls.target.set(0, 0, 11);
            
            // Enhanced lighting system
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const mainLight = new THREE.DirectionalLight(0xffffff, 0.9);
            mainLight.position.set(15, 35, 20);
            mainLight.castShadow = true;
            mainLight.shadow.mapSize.width = 4096;
            mainLight.shadow.mapSize.height = 4096;
            mainLight.shadow.camera.near = 0.5;
            mainLight.shadow.camera.far = 100;
            mainLight.shadow.camera.left = -30;
            mainLight.shadow.camera.right = 30;
            mainLight.shadow.camera.top = 30;
            mainLight.shadow.camera.bottom = -30;
            scene.add(mainLight);
            
            const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
            fillLight.position.set(-15, 20, 10);
            scene.add(fillLight);
            
            const backLight = new THREE.DirectionalLight(0xffffff, 0.2);
            backLight.position.set(0, 15, -20);
            scene.add(backLight);
            
            // Cricket Stadium - Circular outfield
            const stadiumRadius = 70;
            
            // Stadium bowl/ground
            const stadiumGeometry = new THREE.CircleGeometry(stadiumRadius, 64);
            const stadiumMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x1a5c1a,
                roughness: 0.85,
                metalness: 0.1
            }});
            const stadium = new THREE.Mesh(stadiumGeometry, stadiumMaterial);
            stadium.rotation.x = -Math.PI / 2;
            stadium.position.set(0, -0.05, 11);
            stadium.receiveShadow = true;
            scene.add(stadium);
            
            // Add stadium grass texture pattern
            const stadiumTexture = document.createElement('canvas');
            stadiumTexture.width = 1024;
            stadiumTexture.height = 1024;
            const stadiumCtx = stadiumTexture.getContext('2d');
            
            // Base green
            stadiumCtx.fillStyle = '#1a5c1a';
            stadiumCtx.fillRect(0, 0, 1024, 1024);
            
            // Grass blades
            for (let i = 0; i < 5000; i++) {{
                const shade = Math.random() * 30 - 15;
                stadiumCtx.fillStyle = `rgb(${{26 + shade}},${{92 + shade * 1.5}},${{26 + shade}})`;
                stadiumCtx.fillRect(Math.random() * 1024, Math.random() * 1024, 2, 2);
            }}
            
            // Mowing pattern - stripes
            stadiumCtx.globalAlpha = 0.15;
            for (let i = 0; i < 20; i++) {{
                if (i % 2 === 0) {{
                    stadiumCtx.fillStyle = '#0d4a0d';
                }} else {{
                    stadiumCtx.fillStyle = '#236b23';
                }}
                const stripeWidth = 1024 / 20;
                stadiumCtx.fillRect(i * stripeWidth, 0, stripeWidth, 1024);
            }}
            stadiumCtx.globalAlpha = 1.0;
            
            const texture = new THREE.CanvasTexture(stadiumTexture);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(4, 4);
            stadium.material.map = texture;
            stadium.material.needsUpdate = true;
            
            // Inner circle (30-yard circle)
            const innerCircleGeometry = new THREE.RingGeometry(27, 27.3, 64);
            const innerCircleMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.9,
                emissive: 0xffffff,
                emissiveIntensity: 0.1
            }});
            const innerCircle = new THREE.Mesh(innerCircleGeometry, innerCircleMaterial);
            innerCircle.rotation.x = -Math.PI / 2;
            innerCircle.position.set(0, 0, 11);
            innerCircle.receiveShadow = true;
            scene.add(innerCircle);
            
            // Boundary rope
            const boundaryGeometry = new THREE.RingGeometry(stadiumRadius - 0.5, stadiumRadius, 64);
            const boundaryMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.7,
                emissive: 0xffffff,
                emissiveIntensity: 0.2
            }});
            const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
            boundary.rotation.x = -Math.PI / 2;
            boundary.position.set(0, 0.02, 11);
            scene.add(boundary);
            
            // Stadium boundary markers (advertising boards simulation)
            const markerCount = 32;
            const sponsorLogos = ['VIVO', 'BCCI', 'TATA', 'CRED', 'DREAM11', 'PayTM', 'MRF', 'ARAMCO'];
            for (let i = 0; i < markerCount; i++) {{
                const angle = (i / markerCount) * Math.PI * 2;
                const radius = stadiumRadius - 2;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius + 11;
                
                // Advertising board
                const markerGeometry = new THREE.BoxGeometry(4, 2, 0.3);
                const hue = (i / markerCount) * 360;
                const markerMaterial = new THREE.MeshStandardMaterial({{ 
                    color: new THREE.Color(`hsl(${{hue}}, 80%, 50%)`),
                    roughness: 0.3,
                    metalness: 0.4,
                    emissive: new THREE.Color(`hsl(${{hue}}, 80%, 35%)`),
                    emissiveIntensity: 0.4
                }});
                const marker = new THREE.Mesh(markerGeometry, markerMaterial);
                marker.position.set(x, 1, z);
                marker.lookAt(0, 1, 11);
                marker.castShadow = true;
                scene.add(marker);
                
                // Add sponsor text on boards
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 256;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = `hsl(${{hue}}, 80%, 50%)`;
                ctx.fillRect(0, 0, 512, 256);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 80px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(sponsorLogos[i % sponsorLogos.length], 256, 128);
                
                const texture = new THREE.CanvasTexture(canvas);
                marker.material.map = texture;
                marker.material.needsUpdate = true;
            }}
            
            // Stadium stands with audience
            const standSections = 8;
            for (let section = 0; section < standSections; section++) {{
                const startAngle = (section / standSections) * Math.PI * 2;
                const endAngle = ((section + 1) / standSections) * Math.PI * 2;
                const rows = 12;
                const seatsPerRow = 40;
                
                for (let row = 0; row < rows; row++) {{
                    const rowRadius = stadiumRadius + 3 + row * 1.2;
                    const rowHeight = 0.5 + row * 0.8;
                    
                    for (let seat = 0; seat < seatsPerRow; seat++) {{
                        const angle = startAngle + (seat / seatsPerRow) * (endAngle - startAngle);
                        const x = Math.cos(angle) * rowRadius;
                        const z = Math.sin(angle) * rowRadius + 11;
                        
                        // Audience member (small colorful boxes representing people)
                        if (Math.random() > 0.15) {{ // 85% occupancy
                            const personGeometry = new THREE.BoxGeometry(0.4, 0.6, 0.3);
                            const shirtColors = [
                                0x0000ff, 0xff0000, 0x00ff00, 0xffff00, 
                                0xff00ff, 0x00ffff, 0xffa500, 0x800080,
                                0xffffff, 0x000000
                            ];
                            const personMaterial = new THREE.MeshStandardMaterial({{ 
                                color: shirtColors[Math.floor(Math.random() * shirtColors.length)],
                                roughness: 0.7,
                                metalness: 0.1
                            }});
                            const person = new THREE.Mesh(personGeometry, personMaterial);
                            person.position.set(x, rowHeight, z);
                            person.lookAt(0, 0, 11);
                            person.castShadow = true;
                            scene.add(person);
                            
                            // Random celebration hands up
                            if (Math.random() > 0.7) {{
                                const handGeometry = new THREE.BoxGeometry(0.15, 0.4, 0.1);
                                const handMaterial = new THREE.MeshStandardMaterial({{ 
                                    color: 0xffdbac,
                                    roughness: 0.8
                                }});
                                const hand = new THREE.Mesh(handGeometry, handMaterial);
                                const handOffset = Math.random() > 0.5 ? 0.25 : -0.25;
                                hand.position.set(x + handOffset * Math.cos(angle), rowHeight + 0.5, z + handOffset * Math.sin(angle));
                                scene.add(hand);
                            }}
                        }}
                    }}
                }}
            }}
            
            // Team banners/flags in stands
            const bannerPositions = [
                {{ angle: 0, text: '{title.split("-")[0].strip() if "-" in title else "IPL"}' }},
                {{ angle: Math.PI / 2, text: 'IPL 2024' }},
                {{ angle: Math.PI, text: 'CRICKET' }},
                {{ angle: 3 * Math.PI / 2, text: 'FANS' }}
            ];
            
            bannerPositions.forEach(banner => {{
                const bannerRadius = stadiumRadius + 8;
                const x = Math.cos(banner.angle) * bannerRadius;
                const z = Math.sin(banner.angle) * bannerRadius + 11;
                
                const bannerGeometry = new THREE.PlaneGeometry(6, 3);
                const bannerCanvas = document.createElement('canvas');
                bannerCanvas.width = 512;
                bannerCanvas.height = 256;
                const bannerCtx = bannerCanvas.getContext('2d');
                
                // Gradient background
                const gradient = bannerCtx.createLinearGradient(0, 0, 512, 256);
                gradient.addColorStop(0, '#667eea');
                gradient.addColorStop(1, '#764ba2');
                bannerCtx.fillStyle = gradient;
                bannerCtx.fillRect(0, 0, 512, 256);
                
                // Border
                bannerCtx.strokeStyle = 'white';
                bannerCtx.lineWidth = 8;
                bannerCtx.strokeRect(0, 0, 512, 256);
                
                // Text
                bannerCtx.fillStyle = 'white';
                bannerCtx.font = 'bold 90px Arial';
                bannerCtx.textAlign = 'center';
                bannerCtx.textBaseline = 'middle';
                bannerCtx.fillText(banner.text, 256, 128);
                
                const bannerTexture = new THREE.CanvasTexture(bannerCanvas);
                const bannerMaterial = new THREE.MeshStandardMaterial({{ 
                    map: bannerTexture,
                    side: THREE.DoubleSide,
                    roughness: 0.6,
                    metalness: 0.2
                }});
                const bannerMesh = new THREE.Mesh(bannerGeometry, bannerMaterial);
                bannerMesh.position.set(x, 8, z);
                bannerMesh.lookAt(0, 8, 11);
                scene.add(bannerMesh);
                
                // Banner pole
                const poleGeometry = new THREE.CylinderGeometry(0.1, 0.1, 10, 8);
                const poleMaterial = new THREE.MeshStandardMaterial({{ color: 0xaaaaaa }});
                const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                pole.position.set(x, 5, z);
                scene.add(pole);
            }});
            
            // Confetti particles (celebration)
            const confettiCount = 200;
            const confettiGeometry = new THREE.BufferGeometry();
            const confettiPositions = [];
            const confettiColors = [];
            
            for (let i = 0; i < confettiCount; i++) {{
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * 30 + 20;
                confettiPositions.push(
                    Math.cos(angle) * radius,
                    Math.random() * 25 + 15,
                    Math.sin(angle) * radius + 11
                );
                
                const color = new THREE.Color();
                color.setHSL(Math.random(), 1.0, 0.5);
                confettiColors.push(color.r, color.g, color.b);
            }}
            
            confettiGeometry.setAttribute('position', new THREE.Float32BufferAttribute(confettiPositions, 3));
            confettiGeometry.setAttribute('color', new THREE.Float32BufferAttribute(confettiColors, 3));
            
            const confettiMaterial = new THREE.PointsMaterial({{
                size: 0.3,
                vertexColors: true,
                transparent: true,
                opacity: 0.8
            }});
            
            const confetti = new THREE.Points(confettiGeometry, confettiMaterial);
            scene.add(confetti);
            
            // Floodlight towers (4 corners)
            const floodlightPositions = [
                {{ x: 50, z: -30 }},
                {{ x: -50, z: -30 }},
                {{ x: 50, z: 52 }},
                {{ x: -50, z: 52 }}
            ];
            
            floodlightPositions.forEach(pos => {{
                // Tower pole
                const poleGeometry = new THREE.CylinderGeometry(0.5, 0.8, 40, 16);
                const poleMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0x808080,
                    roughness: 0.6,
                    metalness: 0.7
                }});
                const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                pole.position.set(pos.x, 20, pos.z);
                pole.castShadow = true;
                scene.add(pole);
                
                // Light fixture on top
                const lightGeometry = new THREE.BoxGeometry(3, 2, 1);
                const lightMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0xffff00,
                    roughness: 0.3,
                    metalness: 0.5,
                    emissive: 0xffff88,
                    emissiveIntensity: 0.8
                }});
                const lightFixture = new THREE.Mesh(lightGeometry, lightMaterial);
                lightFixture.position.set(pos.x, 41, pos.z);
                lightFixture.lookAt(0, 0, 11);
                scene.add(lightFixture);
            }});
            
            // Cricket pitch - Realistic rectangular pitch (3.05m width x 20.12m length = 22 yards)
            // Reduced size: Using scale factor of 0.7 for better visibility
            const pitchWidth = 3.05 * 0.7;  // 2.135m
            const pitchLength = 20.12 * 0.7; // 14.084m (~15.4 yards visual)
            
            const pitchGeometry = new THREE.PlaneGeometry(pitchWidth, pitchLength);
            const pitchCanvas = document.createElement('canvas');
            pitchCanvas.width = 256;
            pitchCanvas.height = 1024;
            const pitchCtx = pitchCanvas.getContext('2d');
            
            // Base color - light brown/tan (cricket pitch color)
            const gradient = pitchCtx.createLinearGradient(0, 0, 0, 1024);
            gradient.addColorStop(0, '#c9a875');
            gradient.addColorStop(0.5, '#b89665');
            gradient.addColorStop(1, '#c9a875');
            pitchCtx.fillStyle = gradient;
            pitchCtx.fillRect(0, 0, 256, 1024);
            
            // Add dirt/clay texture
            for (let i = 0; i < 5000; i++) {{
                const shade = Math.random() * 40 - 20;
                pitchCtx.fillStyle = `rgb(${{201 + shade}},${{168 + shade}},${{117 + shade}})`;
                pitchCtx.fillRect(Math.random() * 256, Math.random() * 1024, 2, 2);
            }}
            
            // Worn areas in the middle (darker patches where bowlers land)
            pitchCtx.fillStyle = 'rgba(160, 130, 80, 0.4)';
            for (let i = 0; i < 6; i++) {{
                const y = 400 + Math.random() * 250;
                const x = 90 + Math.random() * 75;
                pitchCtx.fillRect(x, y, 30 + Math.random() * 25, 40 + Math.random() * 30);
            }}
            
            // Edge striping pattern (subtle)
            pitchCtx.fillStyle = 'rgba(200, 160, 110, 0.2)';
            pitchCtx.fillRect(0, 0, 20, 1024);
            pitchCtx.fillRect(236, 0, 20, 1024);
            
            const pitchTexture = new THREE.CanvasTexture(pitchCanvas);
            const pitchMaterial = new THREE.MeshStandardMaterial({{ 
                map: pitchTexture,
                roughness: 0.85,
                metalness: 0.0
            }});
            const pitch = new THREE.Mesh(pitchGeometry, pitchMaterial);
            pitch.rotation.x = -Math.PI / 2;
            pitch.position.set(0, 0.005, 11);
            pitch.receiveShadow = true;
            pitch.castShadow = false;
            scene.add(pitch);
            
            // Pitch markings - white creases (adjusted for new pitch size)
            const creaseMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.7,
                emissive: 0xffffff,
                emissiveIntensity: 0.3
            }});
            
            // Bowling end crease (far end - bowler's end)
            const bowlingCreaseGeometry = new THREE.PlaneGeometry(pitchWidth + 0.1, 0.06);
            const bowlingCrease = new THREE.Mesh(bowlingCreaseGeometry, creaseMaterial);
            bowlingCrease.rotation.x = -Math.PI / 2;
            bowlingCrease.position.set(0, 0.01, 11 - pitchLength / 2);
            bowlingCrease.receiveShadow = true;
            scene.add(bowlingCrease);
            
            // Batting end crease (near end - batter's end)
            const battingCrease = new THREE.Mesh(bowlingCreaseGeometry.clone(), creaseMaterial);
            battingCrease.rotation.x = -Math.PI / 2;
            battingCrease.position.set(0, 0.01, 11 + pitchLength / 2);
            battingCrease.receiveShadow = true;
            scene.add(battingCrease);
            
            // Popping creases (1.22m / 4 feet in front of stumps, scaled)
            const poppingCreaseOffset = 1.22 * 0.7;
            const poppingCreaseGeometry = new THREE.PlaneGeometry(pitchWidth + 0.1, 0.05);
            
            const bowlingPoppingCrease = new THREE.Mesh(poppingCreaseGeometry, creaseMaterial);
            bowlingPoppingCrease.rotation.x = -Math.PI / 2;
            bowlingPoppingCrease.position.set(0, 0.01, 11 - pitchLength / 2 + poppingCreaseOffset);
            scene.add(bowlingPoppingCrease);
            
            const battingPoppingCrease = new THREE.Mesh(poppingCreaseGeometry.clone(), creaseMaterial);
            battingPoppingCrease.rotation.x = -Math.PI / 2;
            battingPoppingCrease.position.set(0, 0.01, 11 + pitchLength / 2 - poppingCreaseOffset);
            scene.add(battingPoppingCrease);
            
            // Return creases (perpendicular lines at both ends)
            const returnCreaseLength = 2.44 * 0.7;
            const returnCreaseGeometry = new THREE.PlaneGeometry(0.05, returnCreaseLength);
            const returnCreaseX = pitchWidth / 2;
            
            // Bowling end return creases
            [-returnCreaseX, returnCreaseX].forEach(x => {{
                const returnCrease = new THREE.Mesh(returnCreaseGeometry, creaseMaterial);
                returnCrease.rotation.x = -Math.PI / 2;
                returnCrease.position.set(x, 0.01, 11 - pitchLength / 2 + returnCreaseLength / 2);
                scene.add(returnCrease);
            }});
            
            // Batting end return creases
            [-returnCreaseX, returnCreaseX].forEach(x => {{
                const returnCrease = new THREE.Mesh(returnCreaseGeometry.clone(), creaseMaterial);
                returnCrease.rotation.x = -Math.PI / 2;
                returnCrease.position.set(x, 0.01, 11 + pitchLength / 2 - returnCreaseLength / 2);
                scene.add(returnCrease);
            }});
            
            // Stumps - realistic wooden stumps at both ends (28 inches / 0.71m high, scaled)
            const stumpHeight = 0.71 * 0.7;
            const stumpRadius = 0.022 * 0.7;
            const stumpSpacing = 0.115 * 0.7; // Distance between stumps (4.5 inches)
            
            const stumpMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.6,
                metalness: 0.15,
                emissive: 0x5d2e0f,
                emissiveIntensity: 0.1
            }});
            
            const stumpPositions = [-stumpSpacing, 0, stumpSpacing]; // Off, Middle, Leg
            
            // BOWLING END STUMPS (far end)
            stumpPositions.forEach(x => {{
                const stump = new THREE.Mesh(
                    new THREE.CylinderGeometry(stumpRadius, stumpRadius, stumpHeight, 16), 
                    stumpMaterial
                );
                stump.position.set(x, stumpHeight / 2, 11 - pitchLength / 2);
                stump.castShadow = true;
                stump.receiveShadow = true;
                scene.add(stump);
                
                // Add stump groove detail
                const grooveGeometry = new THREE.CylinderGeometry(stumpRadius * 0.95, stumpRadius * 0.95, stumpHeight * 0.1, 16);
                const grooveMaterial = new THREE.MeshStandardMaterial({{ color: 0x6d3d1a, roughness: 0.8 }});
                const groove = new THREE.Mesh(grooveGeometry, grooveMaterial);
                groove.position.set(x, stumpHeight * 0.3, 11 - pitchLength / 2);
                scene.add(groove);
            }});
            
            // BATTING END STUMPS (near end)
            stumpPositions.forEach(x => {{
                const stump = new THREE.Mesh(
                    new THREE.CylinderGeometry(stumpRadius, stumpRadius, stumpHeight, 16), 
                    stumpMaterial
                );
                stump.position.set(x, stumpHeight / 2, 11 + pitchLength / 2);
                stump.castShadow = true;
                stump.receiveShadow = true;
                scene.add(stump);
                
                // Add stump groove detail
                const grooveGeometry = new THREE.CylinderGeometry(stumpRadius * 0.95, stumpRadius * 0.95, stumpHeight * 0.1, 16);
                const grooveMaterial = new THREE.MeshStandardMaterial({{ color: 0x6d3d1a, roughness: 0.8 }});
                const groove = new THREE.Mesh(grooveGeometry, grooveMaterial);
                groove.position.set(x, stumpHeight * 0.3, 11 + pitchLength / 2);
                scene.add(groove);
            }});
            
            // Bails on top of stumps (two bails connecting three stumps)
            const bailLength = stumpSpacing;
            const bailRadius = 0.012 * 0.7;
            const bailHeight = stumpHeight + bailRadius;
            
            const bailMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xD2691E,
                roughness: 0.5,
                metalness: 0.3,
                emissive: 0x8B4513,
                emissiveIntensity: 0.1
            }});
            
            // Bowling end bails
            [-stumpSpacing / 2, stumpSpacing / 2].forEach(x => {{
                const bail = new THREE.Mesh(
                    new THREE.CylinderGeometry(bailRadius, bailRadius, bailLength, 12),
                    bailMaterial
                );
                bail.rotation.z = Math.PI / 2;
                bail.position.set(x, bailHeight, 11 - pitchLength / 2);
                bail.castShadow = true;
                scene.add(bail);
            }});
            
            // Batting end bails
            [-stumpSpacing / 2, stumpSpacing / 2].forEach(x => {{
                const bail = new THREE.Mesh(
                    new THREE.CylinderGeometry(bailRadius, bailRadius, bailLength, 12),
                    bailMaterial
                );
                bail.rotation.z = Math.PI / 2;
                bail.position.set(x, bailHeight, 11 + pitchLength / 2);
                bail.castShadow = true;
                scene.add(bail);
            }});
            
            // Add text labels for ends
            const createTextLabel = (text, position) => {{
                const canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 64;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fillRect(0, 0, 256, 64);
                ctx.fillStyle = '#1e3c72';
                ctx.font = 'bold 32px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(text, 128, 32);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.position.copy(position);
                sprite.scale.set(3, 0.75, 1);
                return sprite;
            }};
            
            const bowlingLabel = createTextLabel('BOWLING END', new THREE.Vector3(0, 1.2, 11 - pitchLength / 2 - 1));
            scene.add(bowlingLabel);
            
            const battingLabel = createTextLabel('BATTING END', new THREE.Vector3(0, 1.2, 11 + pitchLength / 2 + 1));
            scene.add(battingLabel);
            
            // Ball landing positions with enhanced materials (mapped to new pitch dimensions)
            const colorMap = {{ 
                'red': 0xff0000, 
                'purple': 0x9c27b0, 
                'green': 0x00ff00, 
                'blue': 0x2196f3, 
                'gray': 0x808080 
            }};
            
            pitchData.forEach(ball => {{
                const radius = ball.size * 0.02;
                const geometry = new THREE.SphereGeometry(radius, 20, 20);
                const material = new THREE.MeshStandardMaterial({{ 
                    color: colorMap[ball.color],
                    roughness: 0.3,
                    metalness: 0.5,
                    emissive: colorMap[ball.color],
                    emissiveIntensity: 0.5
                }});
                const sphere = new THREE.Mesh(geometry, material);
                
                // Map ball positions to the rectangular pitch
                // ball.x ranges from -1.2 to 1.2 (width)
                // ball.y ranges from 0 to 22 (length)
                
                // Scale x to fit within pitch width (pitchWidth / 2.4 to normalize -1.2 to 1.2 range)
                const xPos = ball.x * (pitchWidth / 2.6);
                
                // Map y from 0-22 range to bowling end to batting end
                // 0 = bowling end (11 - pitchLength/2), 22 = batting end (11 + pitchLength/2)
                const yScale = pitchLength / 22;
                const zPos = (11 - pitchLength / 2) + (ball.y * yScale);
                
                sphere.position.set(xPos, radius + 0.015, zPos);
                sphere.castShadow = true;
                sphere.receiveShadow = true;
                sphere.userData = {{ 
                    batter: ball.batter, 
                    bowler: ball.bowler, 
                    runs: ball.runs, 
                    wicket: ball.wicket 
                }};
                scene.add(sphere);
            }});
            
            // View preset functions
            window.setView_{div_id} = function(view) {{
                let targetPos, targetLookAt;
                switch(view) {{
                    case 'top':
                        targetPos = {{ x: 0, y: 35, z: 11 }};
                        targetLookAt = {{ x: 0, y: 0, z: 11 }};
                        break;
                    case 'bowler':
                        targetPos = {{ x: 0, y: 8, z: -10 }};
                        targetLookAt = {{ x: 0, y: 0, z: 11 }};
                        break;
                    case 'batter':
                        targetPos = {{ x: 0, y: 8, z: 32 }};
                        targetLookAt = {{ x: 0, y: 0, z: 11 }};
                        break;
                    case 'side':
                        targetPos = {{ x: 20, y: 12, z: 11 }};
                        targetLookAt = {{ x: 0, y: 0, z: 11 }};
                        break;
                    case 'reset':
                        targetPos = {{ x: 0, y: 18, z: 22 }};
                        targetLookAt = {{ x: 0, y: 0, z: 11 }};
                        break;
                }}
                
                const startPos = {{ x: camera.position.x, y: camera.position.y, z: camera.position.z }};
                const startTime = Date.now();
                const duration = 1200;
                
                function animateCamera() {{
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;
                    
                    camera.position.x = startPos.x + (targetPos.x - startPos.x) * eased;
                    camera.position.y = startPos.y + (targetPos.y - startPos.y) * eased;
                    camera.position.z = startPos.z + (targetPos.z - startPos.z) * eased;
                    
                    controls.target.set(targetLookAt.x, targetLookAt.y, targetLookAt.z);
                    controls.update();
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animateCamera);
                    }}
                }}
                animateCamera();
            }};
            
            // Animation loop with confetti
            let confettiTime = 0;
            function animate() {{
                requestAnimationFrame(animate);
                
                // Animate confetti falling
                confettiTime += 0.016;
                const positions = confetti.geometry.attributes.position.array;
                for (let i = 0; i < confettiCount; i++) {{
                    const idx = i * 3;
                    positions[idx + 1] -= 0.05; // Fall down
                    
                    // Sway left-right
                    positions[idx] += Math.sin(confettiTime + i) * 0.02;
                    
                    // Reset if below ground
                    if (positions[idx + 1] < 0) {{
                        const angle = Math.random() * Math.PI * 2;
                        const radius = Math.random() * 30 + 20;
                        positions[idx] = Math.cos(angle) * radius;
                        positions[idx + 1] = Math.random() * 25 + 15;
                        positions[idx + 2] = Math.sin(angle) * radius + 11;
                    }}
                }}
                confetti.geometry.attributes.position.needsUpdate = true;
                
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
    </body>
    </html>
    """
    return html

def render_wagon_wheel(data, title, width=600, height=600):
    """Render wagon wheel visualization with realistic cricket stadium using Three.js"""
    data_json = json.dumps(data)
    div_id = f"wagon_wheel_{uuid.uuid4().hex[:8]}"
    unique_id = uuid.uuid4().hex[:8]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ 
                margin: 0; 
                padding: 0; 
                font-family: 'Segoe UI', sans-serif;
            }}
            .wagon-container {{ 
                position: relative;
                border: 2px solid #ddd; 
                border-radius: 10px;
                overflow: hidden;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .wagon-title {{
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                color: white;
                padding: 10px;
                text-transform: uppercase;
                letter-spacing: 2px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .legend-box {{
                position: absolute;
                top: 60px;
                right: 20px;
                background: rgba(0,0,0,0.85);
                padding: 12px 16px;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                backdrop-filter: blur(10px);
                border: 2px solid rgba(255,255,255,0.2);
                z-index: 100;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 6px 0;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin-right: 10px;
                border: 2px solid white;
            }}
        </style>
    </head>
    <body>
        <div class="wagon-container">
            <div class="wagon-title">{title}</div>
            <div id="{div_id}"></div>
            <div class="legend-box">
                <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 6px;">SHOT TYPES</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff0000;"></div>
                    <span>Boundaries (4s & 6s)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9800;"></div>
                    <span>Twos (2 runs)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196f3;"></div>
                    <span>Threes (3 runs)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #00ff00;"></div>
                    <span>Singles (1 run)</span>
                </div>
            </div>
        </div>
        <script>
        (function() {{
            const wagonData = {data_json};
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);
            
            const camera = new THREE.PerspectiveCamera(50, {width}/{height}, 0.1, 1000);
            camera.position.set(0, 85, 5);
            camera.lookAt(0, 0, 0);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize({width}, {height});
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('{div_id}').appendChild(renderer.domElement);
            
            // Enhanced Lighting System
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 0.9);
            sunLight.position.set(50, 120, 50);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.left = -100;
            sunLight.shadow.camera.right = 100;
            sunLight.shadow.camera.top = 100;
            sunLight.shadow.camera.bottom = -100;
            scene.add(sunLight);
            
            const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
            fillLight.position.set(-50, 80, -50);
            scene.add(fillLight);
            
            // Create procedural grass texture
            const grassCanvas = document.createElement('canvas');
            grassCanvas.width = 1024;
            grassCanvas.height = 1024;
            const grassCtx = grassCanvas.getContext('2d');
            
            // Base grass with mowing pattern
            for (let i = 0; i < 20; i++) {{
                grassCtx.fillStyle = i % 2 === 0 ? '#157015' : '#1a7a1a';
                grassCtx.fillRect(0, i * 51.2, 1024, 51.2);
            }}
            
            // Add realistic grass texture
            for (let i = 0; i < 8000; i++) {{
                const x = Math.random() * 1024;
                const y = Math.random() * 1024;
                const brightness = 90 + Math.random() * 50;
                grassCtx.fillStyle = `rgba(20, ${{brightness}}, 20, 0.6)`;
                grassCtx.fillRect(x, y, 2, 2);
            }}
            
            const grassTexture = new THREE.CanvasTexture(grassCanvas);
            grassTexture.wrapS = THREE.RepeatWrapping;
            grassTexture.wrapT = THREE.RepeatWrapping;
            
            // Circular stadium ground (70m radius - regulation size)
            const groundGeometry = new THREE.CircleGeometry(70, 64);
            const groundMaterial = new THREE.MeshStandardMaterial({{ 
                map: grassTexture,
                roughness: 0.85,
                metalness: 0.1
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            scene.add(ground);
            
            // 30-yard circle (regulation inner circle)
            const innerCircleGeometry = new THREE.RingGeometry(27.43, 27.73, 64);
            const innerCircleMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.6,
                metalness: 0.2
            }});
            const innerCircle = new THREE.Mesh(innerCircleGeometry, innerCircleMaterial);
            innerCircle.rotation.x = -Math.PI / 2;
            innerCircle.position.y = 0.05;
            scene.add(innerCircle);
            
            // Boundary rope (white)
            const boundaryGeometry = new THREE.RingGeometry(69.5, 70, 64);
            const boundaryMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.4,
                metalness: 0.3
            }});
            const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
            boundary.rotation.x = -Math.PI / 2;
            boundary.position.y = 0.1;
            scene.add(boundary);
            
            // Advertising boards around boundary (32 boards)
            const adColors = [
                0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 
                0xff00ff, 0x00ffff, 0xff8800, 0x8800ff
            ];
            
            for (let i = 0; i < 32; i++) {{
                const angle = (i / 32) * Math.PI * 2;
                const x = Math.cos(angle) * 68;
                const z = Math.sin(angle) * 68;
                
                const boardGeometry = new THREE.BoxGeometry(3.5, 1.8, 0.3);
                const boardMaterial = new THREE.MeshStandardMaterial({{ 
                    color: adColors[i % adColors.length],
                    roughness: 0.4,
                    metalness: 0.5,
                    emissive: adColors[i % adColors.length],
                    emissiveIntensity: 0.2
                }});
                const board = new THREE.Mesh(boardGeometry, boardMaterial);
                board.position.set(x, 0.9, z);
                board.lookAt(0, 0.9, 0);
                board.castShadow = true;
                scene.add(board);
            }}
            
            // Create realistic pitch texture with wear patterns
            const pitchCanvas = document.createElement('canvas');
            pitchCanvas.width = 256;
            pitchCanvas.height = 2048;
            const pitchCtx = pitchCanvas.getContext('2d');
            
            // Base pitch color (tan/brown)
            pitchCtx.fillStyle = '#c9a875';
            pitchCtx.fillRect(0, 0, 256, 2048);
            
            // Add dirt particles for realism
            for (let i = 0; i < 10000; i++) {{
                const x = Math.random() * 256;
                const y = Math.random() * 2048;
                const shade = 170 + Math.random() * 50;
                pitchCtx.fillStyle = `rgb(${{shade}}, ${{shade * 0.75}}, ${{shade * 0.55}})`;
                pitchCtx.fillRect(x, y, 1, 1);
            }}
            
            // Add worn patches in center (where bowlers land)
            for (let i = 0; i < 8; i++) {{
                const y = 900 + Math.random() * 300;
                pitchCtx.fillStyle = 'rgba(150, 120, 85, 0.4)';
                pitchCtx.fillRect(50 + Math.random() * 30, y, 120 + Math.random() * 40, 50);
            }}
            
            const pitchTexture = new THREE.CanvasTexture(pitchCanvas);
            
            // Cricket pitch (22 yards = 20.12m length, 3.05m width)
            const pitchGeometry = new THREE.PlaneGeometry(3.05, 20.12);
            const pitchMaterial = new THREE.MeshStandardMaterial({{ 
                map: pitchTexture,
                roughness: 0.92,
                metalness: 0.08
            }});
            const pitch = new THREE.Mesh(pitchGeometry, pitchMaterial);
            pitch.rotation.x = -Math.PI / 2;
            pitch.position.y = 0.15;
            pitch.receiveShadow = true;
            scene.add(pitch);
            
            // Cricket stumps at striker's end (center)
            const stumpGeometry = new THREE.CylinderGeometry(0.022, 0.022, 0.71, 16);
            const stumpMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.4,
                metalness: 0.2
            }});
            
            // Three stumps with proper spacing (11cm between stumps)
            [-0.11, 0, 0.11].forEach(xPos => {{
                const stump = new THREE.Mesh(stumpGeometry, stumpMaterial);
                stump.position.set(xPos, 0.355, 0);
                stump.castShadow = true;
                scene.add(stump);
            }});
            
            // Bails on top of stumps
            const bailGeometry = new THREE.CylinderGeometry(0.01, 0.01, 0.11, 8);
            const bailMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.4,
                metalness: 0.2
            }});
            
            [-0.055, 0.055].forEach(xPos => {{
                const bail = new THREE.Mesh(bailGeometry, bailMaterial);
                bail.rotation.z = Math.PI / 2;
                bail.position.set(xPos, 0.71, 0);
                scene.add(bail);
            }});
            
            // Crease line at striker's end
            const creaseGeometry = new THREE.PlaneGeometry(3, 0.05);
            const creaseMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.6
            }});
            const crease = new THREE.Mesh(creaseGeometry, creaseMaterial);
            crease.rotation.x = -Math.PI / 2;
            crease.position.y = 0.16;
            scene.add(crease);
            
            // Wagon wheel shot lines and balls
            const colorMap = {{
                'red': 0xff0000,      // Boundaries (4s & 6s)
                'orange': 0xff9800,  // Twos
                'blue': 0x2196f3,    // Threes
                'green': 0x00ff00    // Singles
            }};
            
            wagonData.forEach(shot => {{
                // Shot line from stumps to landing point
                const lineMaterial = new THREE.LineBasicMaterial({{ 
                    color: colorMap[shot.color],
                    linewidth: 2,
                    opacity: 0.65,
                    transparent: true
                }});
                const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0.4, 0),
                    new THREE.Vector3(shot.x, 0.4, shot.y)
                ]);
                const line = new THREE.Line(lineGeometry, lineMaterial);
                scene.add(line);
                
                // Ball at landing point
                const radius = shot.size * 0.12;
                const ballGeometry = new THREE.SphereGeometry(radius, 20, 20);
                const ballMaterial = new THREE.MeshStandardMaterial({{ 
                    color: colorMap[shot.color],
                    roughness: 0.3,
                    metalness: 0.7,
                    emissive: colorMap[shot.color],
                    emissiveIntensity: 0.4
                }});
                const ball = new THREE.Mesh(ballGeometry, ballMaterial);
                ball.position.set(shot.x, radius + 0.2, shot.y);
                ball.castShadow = true;
                scene.add(ball);
            }});
            
            // Interactive OrbitControls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 25;
            controls.maxDistance = 150;
            controls.maxPolarAngle = Math.PI / 2.1;
            controls.target.set(0, 0, 0);
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
    </body>
    </html>
    """
    return html

def render_bowling_length_map(df, team, phase=None, bowler_type=None, unique_id=""):
    """Render 3D bowling length visualization with zones and percentages"""
    pitch_data = generate_pitch_map_data_complete(df, team=team, phase=phase, bowler_type=bowler_type)
    
    if not pitch_data:
        return "<p>No data available for bowling length map</p>"
    
    div_id = f"bowling_length_{unique_id}"
    data_json = json.dumps(pitch_data)
    
    title_parts = [f"{team} - Bowling Length Analysis"]
    if phase:
        title_parts.append(f"({phase})")
    if bowler_type and bowler_type != 'All Types':
        title_parts.append(f"vs {bowler_type}")
    title = " ".join(title_parts)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ 
                margin: 0; 
                padding: 20px; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .bowling-container {{ 
                position: relative; 
                text-align: center;
                background: rgba(255,255,255,0.05);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            .bowling-title {{ 
                text-align: center; 
                font-size: 26px; 
                font-weight: bold; 
                margin-bottom: 20px;
                color: white;
                text-transform: uppercase;
                letter-spacing: 3px;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            }}
            #{div_id} {{ 
                border: 3px solid rgba(255,255,255,0.3); 
                border-radius: 12px; 
                display: inline-block;
                box-shadow: 0 10px 40px rgba(0,0,0,0.4);
                background: #000;
            }}
            .stats-overlay {{
                position: absolute;
                top: 90px;
                right: 30px;
                background: rgba(0,0,0,0.95);
                padding: 12px 16px;
                border-radius: 10px;
                color: white;
                font-size: 11px;
                min-width: 160px;
                border: 2px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                display: none;
                transition: all 0.3s ease;
            }}
            .stats-overlay.show {{
                display: block;
                animation: slideIn 0.3s ease-out;
            }}
            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateX(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            .toggle-stats-btn {{
                position: absolute;
                top: 90px;
                right: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 10px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                z-index: 100;
            }}
            .toggle-stats-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }}
            .toggle-views-btn {{
                position: absolute;
                top: 90px;
                left: 30px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border: none;
                color: white;
                padding: 10px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
                z-index: 100;
            }}
            .toggle-views-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(240, 147, 251, 0.5);
            }}
            .zone-stat {{
                margin: 6px 0;
                padding: 8px 10px;
                background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-left: 3px solid;
            }}
            .zone-name {{
                font-weight: bold;
                text-transform: uppercase;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            .zone-percentage {{
                font-size: 20px;
                font-weight: bold;
            }}
            .short {{ color: #ff6b6b; border-color: #ff6b6b; }}
            .length {{ color: #ffd93d; border-color: #ffd93d; }}
            .full {{ color: #6bcf7f; border-color: #6bcf7f; }}
            .yorker {{ color: #4dabf7; border-color: #4dabf7; }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 12px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }}
            .view-controls {{
                position: absolute;
                top: 90px;
                left: 30px;
                background: rgba(0,0,0,0.9);
                padding: 12px 14px;
                border-radius: 10px;
                color: white;
                font-size: 11px;
                border: 2px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                display: none;
                z-index: 10;
            }}
            .view-controls.show {{
                display: block;
                animation: slideInLeft 0.3s ease-out;
            }}
            @keyframes slideInLeft {{
                from {{
                    opacity: 0;
                    transform: translateX(-20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            .view-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 8px 12px;
                margin: 3px 0;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
                font-size: 11px;
                transition: all 0.3s ease;
                box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
            }}
            .view-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
            }}
            .controls-title {{
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 12px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }}
        </style>
    </head>
    <body>
        <div class="bowling-container">
            <div class="bowling-title">{title}</div>
            <div id="{div_id}"></div>
            <button class="toggle-views-btn" onclick="toggleViews_{unique_id}()">📐 Views</button>
            <button class="toggle-stats-btn" onclick="toggleStats_{unique_id}()">📊 Statistics</button>
            <div class="view-controls" id="view-controls-{unique_id}">
                <div class="controls-title">📐 VIEWS</div>
                <button class="view-btn" onclick="setTopView_{unique_id}()">📍 Top</button>
                <button class="view-btn" onclick="setBowlerView_{unique_id}()">🎯 Bowler</button>
                <button class="view-btn" onclick="setBatterView_{unique_id}()">🏏 Batter</button>
                <button class="view-btn" onclick="setSideView_{unique_id}()">👁️ Side</button>
                <button class="view-btn" onclick="resetView_{unique_id}()">🔄 Reset</button>
            </div>
            <div class="stats-overlay" id="stats-overlay-{unique_id}">
                <div class="legend-title">📊 Bowling Length %</div>
                <div id="zone-stats-{unique_id}"></div>
            </div>
        </div>
        
        <script>
        (function() {{
            const pitchData = {data_json};
            
            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);
            
            // Camera setup - wider FOV to show complete pitch
            const camera = new THREE.PerspectiveCamera(60, 900/700, 0.1, 500);
            camera.position.set(0, 18, 22);
            camera.lookAt(0, 0, 11);
            
            // Renderer setup
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(900, 700);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('{div_id}').appendChild(renderer.domElement);
            
            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.target.set(0, 0, 11);
            controls.minDistance = 10;
            controls.maxDistance = 80;
            controls.maxPolarAngle = Math.PI / 2.1;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
            sunLight.position.set(20, 30, 20);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.left = -50;
            sunLight.shadow.camera.right = 50;
            sunLight.shadow.camera.top = 50;
            sunLight.shadow.camera.bottom = -50;
            sunLight.shadow.camera.far = 100;
            scene.add(sunLight);
            
            const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
            fillLight.position.set(-20, 15, -15);
            scene.add(fillLight);
            
            // Cricket Stadium - Full circular outfield
            const stadiumRadius = 70;
            
            // Stadium ground with grass texture
            const groundGeometry = new THREE.CircleGeometry(stadiumRadius, 64);
            const groundMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x1a5c1a,
                roughness: 0.85
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.position.set(0, -0.1, 11);
            ground.receiveShadow = true;
            scene.add(ground);
            
            // Create grass texture
            const grassCanvas = document.createElement('canvas');
            grassCanvas.width = 1024;
            grassCanvas.height = 1024;
            const grassCtx = grassCanvas.getContext('2d');
            
            // Base green
            grassCtx.fillStyle = '#1a5c1a';
            grassCtx.fillRect(0, 0, 1024, 1024);
            
            // Add grass texture
            for (let i = 0; i < 5000; i++) {{
                const shade = Math.random() * 30 - 15;
                grassCtx.fillStyle = `rgb(${{26 + shade}},${{92 + shade * 1.5}},${{26 + shade}})`;
                grassCtx.fillRect(Math.random() * 1024, Math.random() * 1024, 2, 2);
            }}
            
            // Mowing pattern stripes
            grassCtx.globalAlpha = 0.15;
            for (let i = 0; i < 20; i++) {{
                grassCtx.fillStyle = i % 2 === 0 ? '#0d4a0d' : '#236b23';
                grassCtx.fillRect(i * 51.2, 0, 51.2, 1024);
            }}
            grassCtx.globalAlpha = 1.0;
            
            const grassTexture = new THREE.CanvasTexture(grassCanvas);
            grassTexture.wrapS = THREE.RepeatWrapping;
            grassTexture.wrapT = THREE.RepeatWrapping;
            grassTexture.repeat.set(4, 4);
            ground.material.map = grassTexture;
            ground.material.needsUpdate = true;
            
            // 30-yard inner circle
            const innerCircleGeometry = new THREE.RingGeometry(27, 27.3, 64);
            const innerCircleMaterial = new THREE.MeshBasicMaterial({{ 
                color: 0xffffff,
                side: THREE.DoubleSide
            }});
            const innerCircle = new THREE.Mesh(innerCircleGeometry, innerCircleMaterial);
            innerCircle.rotation.x = -Math.PI / 2;
            innerCircle.position.set(0, -0.05, 11);
            scene.add(innerCircle);
            
            // Boundary rope
            const boundaryGeometry = new THREE.RingGeometry(stadiumRadius - 0.5, stadiumRadius, 64);
            const boundaryMaterial = new THREE.MeshBasicMaterial({{ 
                color: 0xffffff,
                side: THREE.DoubleSide
            }});
            const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
            boundary.rotation.x = -Math.PI / 2;
            boundary.position.set(0, -0.04, 11);
            scene.add(boundary);
            
            // Advertising boards around boundary
            const markerCount = 32;
            for (let i = 0; i < markerCount; i++) {{
                const angle = (i / markerCount) * Math.PI * 2;
                const radius = stadiumRadius - 2;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius + 11;
                
                const markerGeometry = new THREE.BoxGeometry(3, 1.5, 0.2);
                const hue = (i / markerCount) * 360;
                const markerMaterial = new THREE.MeshStandardMaterial({{ 
                    color: new THREE.Color(`hsl(${{hue}}, 70%, 50%)`),
                    roughness: 0.5,
                    metalness: 0.3,
                    emissive: new THREE.Color(`hsl(${{hue}}, 70%, 30%)`),
                    emissiveIntensity: 0.3
                }});
                const marker = new THREE.Mesh(markerGeometry, markerMaterial);
                marker.position.set(x, 0.75, z);
                marker.lookAt(0, 0.75, 11);
                marker.castShadow = true;
                scene.add(marker);
            }}
            
            // Floodlight towers at corners
            const floodlightPositions = [
                {{ x: 50, z: -30 }},
                {{ x: -50, z: -30 }},
                {{ x: 50, z: 52 }},
                {{ x: -50, z: 52 }}
            ];
            
            floodlightPositions.forEach(pos => {{
                // Tower pole
                const poleGeometry = new THREE.CylinderGeometry(0.5, 0.8, 40, 16);
                const poleMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0x808080,
                    roughness: 0.6,
                    metalness: 0.7
                }});
                const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                pole.position.set(pos.x, 20, pos.z);
                pole.castShadow = true;
                scene.add(pole);
                
                // Light fixture
                const lightGeometry = new THREE.BoxGeometry(3, 2, 1);
                const lightMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0xffff00,
                    roughness: 0.3,
                    metalness: 0.5,
                    emissive: 0xffff88,
                    emissiveIntensity: 0.8
                }});
                const lightFixture = new THREE.Mesh(lightGeometry, lightMaterial);
                lightFixture.position.set(pos.x, 41, pos.z);
                lightFixture.lookAt(0, 0, 11);
                scene.add(lightFixture);
            }});
            
            // Cricket pitch - realistic tan/brown surface in center
            const pitchGeometry = new THREE.PlaneGeometry(2.8, 23);
            
            // Create pitch texture
            const pitchCanvas = document.createElement('canvas');
            pitchCanvas.width = 256;
            pitchCanvas.height = 2048;
            const pitchCtx = pitchCanvas.getContext('2d');
            
            // Base tan color
            pitchCtx.fillStyle = '#c9a875';
            pitchCtx.fillRect(0, 0, 256, 2048);
            
            // Add dirt texture
            for (let i = 0; i < 8000; i++) {{
                const shade = Math.random() * 40 - 20;
                pitchCtx.fillStyle = `rgb(${{201 + shade}},${{168 + shade}},${{117 + shade}})`;
                pitchCtx.fillRect(Math.random() * 256, Math.random() * 2048, 3, 3);
            }}
            
            // Worn patches
            pitchCtx.fillStyle = 'rgba(160, 130, 80, 0.3)';
            for (let i = 0; i < 5; i++) {{
                const y = 800 + Math.random() * 400;
                pitchCtx.fillRect(60 + Math.random() * 130, y, 40 + Math.random() * 30, 60 + Math.random() * 40);
            }}
            
            const pitchTexture = new THREE.CanvasTexture(pitchCanvas);
            const pitchMaterial = new THREE.MeshStandardMaterial({{ 
                map: pitchTexture,
                roughness: 0.85
            }});
            const pitch = new THREE.Mesh(pitchGeometry, pitchMaterial);
            pitch.rotation.x = -Math.PI / 2;
            pitch.position.y = -0.03;
            pitch.position.z = 11;
            pitch.receiveShadow = true;
            scene.add(pitch);
            
            // Pitch creases (white lines)
            const creaseMaterial = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
            
            // Bowling creases
            const creaseGeometry = new THREE.PlaneGeometry(2.9, 0.08);
            const crease1 = new THREE.Mesh(creaseGeometry, creaseMaterial);
            crease1.rotation.x = -Math.PI / 2;
            crease1.position.set(0, 0.01, 0);
            scene.add(crease1);
            
            const crease2 = new THREE.Mesh(creaseGeometry, creaseMaterial);
            crease2.rotation.x = -Math.PI / 2;
            crease2.position.set(0, 0.01, 22);
            scene.add(crease2);
            
            // Stumps at both ends
            const stumpMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.8
            }});
            
            const stumpPositions = [-0.115, 0, 0.115];
            stumpPositions.forEach(x => {{
                // Bowler end
                const stump1 = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.022, 0.022, 0.71, 12),
                    stumpMaterial
                );
                stump1.position.set(x, 0.36, 0);
                stump1.castShadow = true;
                scene.add(stump1);
                
                // Batter end
                const stump2 = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.022, 0.022, 0.71, 12),
                    stumpMaterial
                );
                stump2.position.set(x, 0.36, 22);
                stump2.castShadow = true;
                scene.add(stump2);
            }});
            
            // Define bowling length zones
            const zones = [
                {{ name: 'SHORT', start: 4, end: 10, color: 0xff6b6b, label: 'SHORT', yPos: 7 }},
                {{ name: 'LENGTH', start: 10, end: 16, color: 0xffd93d, label: 'LENGTH', yPos: 13 }},
                {{ name: 'FULL', start: 16, end: 20, color: 0x6bcf7f, label: 'FULL', yPos: 18 }},
                {{ name: 'YORKER', start: 20, end: 22, color: 0x4dabf7, label: 'YORKER', yPos: 21 }}
            ];
            
            // Create zone boxes
            zones.forEach(zone => {{
                const zoneLength = zone.end - zone.start;
                const zoneGeometry = new THREE.BoxGeometry(3, 0.15, zoneLength);
                const zoneMaterial = new THREE.MeshStandardMaterial({{ 
                    color: zone.color,
                    transparent: true,
                    opacity: 0.6,
                    roughness: 0.5
                }});
                const zoneMesh = new THREE.Mesh(zoneGeometry, zoneMaterial);
                zoneMesh.position.set(0, 0.08, zone.start + zoneLength/2);
                zoneMesh.castShadow = true;
                zoneMesh.receiveShadow = true;
                scene.add(zoneMesh);
                
                // Zone labels
                const canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 128;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'white';
                ctx.font = 'bold 56px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(zone.label, 128, 64);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ 
                    map: texture,
                    transparent: true
                }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.position.set(4.2, 1.5, zone.yPos);
                sprite.scale.set(3, 1.5, 1);
                scene.add(sprite);
            }});
            
            // Add balls
            const colorMap = {{
                'red': 0xff0000,
                'purple': 0x9c27b0,
                'green': 0x00ff00,
                'blue': 0x2196f3,
                'gray': 0x808080
            }};
            
            pitchData.forEach(ball => {{
                const radius = ball.size * 0.02;
                const ballGeometry = new THREE.SphereGeometry(radius, 12, 12);
                const ballMaterial = new THREE.MeshStandardMaterial({{ 
                    color: colorMap[ball.color],
                    roughness: 0.4,
                    metalness: 0.3,
                    emissive: colorMap[ball.color],
                    emissiveIntensity: 0.3
                }});
                const ballMesh = new THREE.Mesh(ballGeometry, ballMaterial);
                ballMesh.position.set(ball.x, radius + 0.15, ball.y);
                ballMesh.castShadow = true;
                scene.add(ballMesh);
            }});
            
            // Calculate zone statistics
            const zoneCounts = {{ SHORT: 0, LENGTH: 0, FULL: 0, YORKER: 0 }};
            pitchData.forEach(ball => {{
                const y = ball.y;
                if (y >= 4 && y < 10) zoneCounts.SHORT++;
                else if (y >= 10 && y < 16) zoneCounts.LENGTH++;
                else if (y >= 16 && y < 20) zoneCounts.FULL++;
                else if (y >= 20 && y <= 22) zoneCounts.YORKER++;
            }});
            
            const total = pitchData.length;
            const statsData = [
                {{ name: 'SHORT', count: zoneCounts.SHORT, class: 'short' }},
                {{ name: 'LENGTH', count: zoneCounts.LENGTH, class: 'length' }},
                {{ name: 'FULL', count: zoneCounts.FULL, class: 'full' }},
                {{ name: 'YORKER', count: zoneCounts.YORKER, class: 'yorker' }}
            ];
            
            const statsHtml = statsData.map(stat => {{
                const percentage = total > 0 ? ((stat.count / total) * 100).toFixed(0) : 0;
                return `
                    <div class="zone-stat ${{stat.class}}">
                        <span class="zone-name">${{stat.name}}</span>
                        <span class="zone-percentage">${{percentage}}%</span>
                    </div>
                `;
            }}).join('');
            
            document.getElementById('zone-stats-{unique_id}').innerHTML = statsHtml;
            
            // Camera animation helper
            function animateCamera(targetPos, targetLookAt) {{
                const startPos = camera.position.clone();
                const startTarget = controls.target.clone();
                const startTime = Date.now();
                const duration = 1000;
                
                function animate() {{
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    
                    camera.position.lerpVectors(startPos, targetPos, eased);
                    controls.target.lerpVectors(startTarget, targetLookAt, eased);
                    controls.update();
                    
                    if (progress < 1) requestAnimationFrame(animate);
                }}
                animate();
            }}
            
            // View functions
            window.setTopView_{unique_id} = () => animateCamera(
                new THREE.Vector3(0, 40, 11),
                new THREE.Vector3(0, 0, 11)
            );
            
            window.setBowlerView_{unique_id} = () => animateCamera(
                new THREE.Vector3(0, 8, -8),
                new THREE.Vector3(0, 0, 11)
            );
            
            window.setBatterView_{unique_id} = () => animateCamera(
                new THREE.Vector3(0, 8, 30),
                new THREE.Vector3(0, 0, 11)
            );
            
            window.setSideView_{unique_id} = () => animateCamera(
                new THREE.Vector3(25, 12, 11),
                new THREE.Vector3(0, 0, 11)
            );
            
            window.resetView_{unique_id} = () => animateCamera(
                new THREE.Vector3(0, 18, 22),
                new THREE.Vector3(0, 0, 11)
            );
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Toggle statistics overlay
            window.toggleStats_{unique_id} = function() {{
                const statsOverlay = document.getElementById('stats-overlay-{unique_id}');
                statsOverlay.classList.toggle('show');
            }};
            
            // Toggle view controls
            window.toggleViews_{unique_id} = function() {{
                const viewControls = document.getElementById('view-controls-{unique_id}');
                viewControls.classList.toggle('show');
            }};
            
            // Toggle view controls
            window.toggleViews_{unique_id} = function() {{
                const viewControls = document.getElementById('view-controls-{unique_id}');
                viewControls.classList.toggle('show');
            }};
        }})();
        </script>
    </body>
    </html>
    """
    return html
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #1a1a1a; }}
            .bowling-container {{ position: relative; text-align: center; }}
            .bowling-title {{ 
                text-align: center; 
                font-size: 24px; 
                font-weight: bold; 
                margin-bottom: 20px;
                color: white;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            #{div_id} {{ 
                border: 2px solid #333; 
                border-radius: 8px; 
                display: inline-block;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            }}
            .stats-overlay {{
                position: absolute;
                top: 80px;
                right: 40px;
                background: rgba(0,0,0,0.85);
                padding: 20px;
                border-radius: 10px;
                color: white;
                font-size: 14px;
                min-width: 200px;
                border: 2px solid #444;
            }}
            .zone-stat {{
                margin: 12px 0;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .zone-name {{
                font-weight: bold;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 1px;
            }}
            .zone-percentage {{
                font-size: 24px;
                font-weight: bold;
            }}
            .short {{ color: #ff6b6b; }}
            .length {{ color: #ffd93d; }}
            .full {{ color: #6bcf7f; }}
            .yorker {{ color: #4dabf7; }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 15px;
                font-size: 16px;
                border-bottom: 2px solid #666;
                padding-bottom: 10px;
            }}
            .view-controls {{
                position: absolute;
                top: 80px;
                left: 20px;
                background: rgba(0,0,0,0.85);
                padding: 15px;
                border-radius: 10px;
                color: white;
                font-size: 12px;
                border: 2px solid #444;
            }}
            .view-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
                transition: all 0.3s;
            }}
            .view-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .controls-title {{
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 14px;
                border-bottom: 2px solid #666;
                padding-bottom: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="bowling-container">
            <div class="bowling-title">{title}</div>
            <div id="{div_id}"></div>
            <div class="view-controls">
                <div class="controls-title">VIEW ANGLES</div>
                <button class="view-btn" onclick="setTopView_{unique_id}()">📐 Top View</button>
                <button class="view-btn" onclick="setBowlerView_{unique_id}()">🎯 Bowler End</button>
                <button class="view-btn" onclick="setBatterView_{unique_id}()">🏏 Batter End</button>
                <button class="view-btn" onclick="setSideView_{unique_id}()">👁️ Side View</button>
                <button class="view-btn" onclick="resetView_{unique_id}()">🔄 Reset</button>
            </div>
            <div class="stats-overlay">
                <div class="legend-title">BOWLING LENGTH %</div>
                <div id="zone-stats-{unique_id}"></div>
            </div>
        </div>
        
        <script>
        (function() {{
            const pitchData = {data_json};
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);
            scene.fog = new THREE.Fog(0x87ceeb, 100, 200);
            
            const camera = new THREE.PerspectiveCamera(50, 900/700, 0.1, 1000);
            camera.position.set(0, 15, 25);
            camera.lookAt(0, 0, 11);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(900, 700);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.outputEncoding = THREE.sRGBEncoding;
            document.getElementById('{div_id}').appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.target.set(0, 0, 11);
            controls.maxPolarAngle = Math.PI / 2.1;
            controls.minPolarAngle = 0;
            controls.minDistance = 10;
            controls.maxDistance = 80;
            controls.enablePan = true;
            controls.panSpeed = 0.8;
            controls.rotateSpeed = 0.6;
            
            // Enhanced Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
            mainLight.position.set(15, 25, 15);
            mainLight.castShadow = true;
            mainLight.shadow.mapSize.width = 2048;
            mainLight.shadow.mapSize.height = 2048;
            mainLight.shadow.camera.near = 0.5;
            mainLight.shadow.camera.far = 150;
            mainLight.shadow.camera.left = -80;
            mainLight.shadow.camera.right = 80;
            mainLight.shadow.camera.top = 80;
            mainLight.shadow.camera.bottom = -80;
            scene.add(mainLight);
            
            const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
            fillLight.position.set(-15, 15, -10);
            scene.add(fillLight);
            
            const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
            backLight.position.set(0, 10, -20);
            scene.add(backLight);
            
            // Cricket Stadium - Circular outfield
            const stadiumRadius = 70;
            
            // Stadium bowl/ground
            const stadiumGeometry = new THREE.CircleGeometry(stadiumRadius, 64);
            const stadiumMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x1a5c1a,
                roughness: 0.85,
                metalness: 0.1
            }});
            const stadium = new THREE.Mesh(stadiumGeometry, stadiumMaterial);
            stadium.rotation.x = -Math.PI / 2;
            stadium.position.set(0, -0.08, 11);
            stadium.receiveShadow = true;
            scene.add(stadium);
            
            // Add stadium grass texture pattern
            const stadiumTexture = document.createElement('canvas');
            stadiumTexture.width = 1024;
            stadiumTexture.height = 1024;
            const stadiumCtx = stadiumTexture.getContext('2d');
            
            // Base green
            stadiumCtx.fillStyle = '#1a5c1a';
            stadiumCtx.fillRect(0, 0, 1024, 1024);
            
            // Grass blades
            for (let i = 0; i < 5000; i++) {{
                const shade = Math.random() * 30 - 15;
                stadiumCtx.fillStyle = `rgb(${{26 + shade}},${{92 + shade * 1.5}},${{26 + shade}})`;
                stadiumCtx.fillRect(Math.random() * 1024, Math.random() * 1024, 2, 2);
            }}
            
            // Mowing pattern - stripes
            stadiumCtx.globalAlpha = 0.15;
            for (let i = 0; i < 20; i++) {{
                if (i % 2 === 0) {{
                    stadiumCtx.fillStyle = '#0d4a0d';
                }} else {{
                    stadiumCtx.fillStyle = '#236b23';
                }}
                const stripeWidth = 1024 / 20;
                stadiumCtx.fillRect(i * stripeWidth, 0, stripeWidth, 1024);
            }}
            stadiumCtx.globalAlpha = 1.0;
            
            const stadTexture = new THREE.CanvasTexture(stadiumTexture);
            stadTexture.wrapS = THREE.RepeatWrapping;
            stadTexture.wrapT = THREE.RepeatWrapping;
            stadTexture.repeat.set(4, 4);
            stadium.material.map = stadTexture;
            stadium.material.needsUpdate = true;
            
            // Inner circle (30-yard circle)
            const innerCircleGeometry = new THREE.RingGeometry(27, 27.3, 64);
            const innerCircleMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.9,
                emissive: 0xffffff,
                emissiveIntensity: 0.1
            }});
            const innerCircle = new THREE.Mesh(innerCircleGeometry, innerCircleMaterial);
            innerCircle.rotation.x = -Math.PI / 2;
            innerCircle.position.set(0, -0.05, 11);
            innerCircle.receiveShadow = true;
            scene.add(innerCircle);
            
            // Boundary rope
            const boundaryGeometry = new THREE.RingGeometry(stadiumRadius - 0.5, stadiumRadius, 64);
            const boundaryMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.7,
                emissive: 0xffffff,
                emissiveIntensity: 0.2
            }});
            const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
            boundary.rotation.x = -Math.PI / 2;
            boundary.position.set(0, -0.03, 11);
            scene.add(boundary);
            
            // Stadium boundary markers (advertising boards)
            const markerCount = 32;
            for (let i = 0; i < markerCount; i++) {{
                const angle = (i / markerCount) * Math.PI * 2;
                const radius = stadiumRadius - 2;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius + 11;
                
                const markerGeometry = new THREE.BoxGeometry(3, 1.5, 0.2);
                const hue = (i / markerCount) * 360;
                const markerMaterial = new THREE.MeshStandardMaterial({{ 
                    color: new THREE.Color(`hsl(${{hue}}, 70%, 50%)`),
                    roughness: 0.5,
                    metalness: 0.3,
                    emissive: new THREE.Color(`hsl(${{hue}}, 70%, 30%)`),
                    emissiveIntensity: 0.3
                }});
                const marker = new THREE.Mesh(markerGeometry, markerMaterial);
                marker.position.set(x, 0.75, z);
                marker.lookAt(0, 0.75, 11);
                marker.castShadow = true;
                scene.add(marker);
            }}
            
            // Floodlight towers (4 corners)
            const floodlightPositions = [
                {{ x: 50, z: -30 }},
                {{ x: -50, z: -30 }},
                {{ x: 50, z: 52 }},
                {{ x: -50, z: 52 }}
            ];
            
            floodlightPositions.forEach(pos => {{
                // Tower pole
                const poleGeometry = new THREE.CylinderGeometry(0.5, 0.8, 40, 16);
                const poleMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0x808080,
                    roughness: 0.6,
                    metalness: 0.7
                }});
                const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                pole.position.set(pos.x, 20, pos.z);
                pole.castShadow = true;
                scene.add(pole);
                
                // Light fixture on top
                const lightGeometry = new THREE.BoxGeometry(3, 2, 1);
                const lightMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0xffff00,
                    roughness: 0.3,
                    metalness: 0.5,
                    emissive: 0xffff88,
                    emissiveIntensity: 0.8
                }});
                const lightFixture = new THREE.Mesh(lightGeometry, lightMaterial);
                lightFixture.position.set(pos.x, 41, pos.z);
                lightFixture.lookAt(0, 0, 11);
                scene.add(lightFixture);
            }});
            
            // Cricket pitch - realistic tan/brown with detailed texture
            const pitchGeometry = new THREE.PlaneGeometry(2.6, 22.5);
            const pitchCanvas = document.createElement('canvas');
            pitchCanvas.width = 256;
            pitchCanvas.height = 2048;
            const pitchCtx = pitchCanvas.getContext('2d');
            
            // Base color - light brown/tan
            pitchCtx.fillStyle = '#c9a875';
            pitchCtx.fillRect(0, 0, 256, 2048);
            
            // Add dirt/clay texture
            for (let i = 0; i < 8000; i++) {{
                const shade = Math.random() * 40 - 20;
                pitchCtx.fillStyle = `rgb(${{201 + shade}},${{168 + shade}},${{117 + shade}})`;
                pitchCtx.fillRect(Math.random() * 256, Math.random() * 2048, 3, 3);
            }}
            
            // Worn areas (darker patches in middle)
            pitchCtx.fillStyle = 'rgba(160, 130, 80, 0.3)';
            for (let i = 0; i < 5; i++) {{
                const y = 800 + Math.random() * 400;
                pitchCtx.fillRect(60 + Math.random() * 130, y, 40 + Math.random() * 30, 60 + Math.random() * 40);
            }}
            
            // Add some cracks
            pitchCtx.strokeStyle = 'rgba(140, 110, 70, 0.4)';
            pitchCtx.lineWidth = 2;
            for (let i = 0; i < 15; i++) {{
                pitchCtx.beginPath();
                const startX = Math.random() * 256;
                const startY = 600 + Math.random() * 800;
                pitchCtx.moveTo(startX, startY);
                pitchCtx.lineTo(startX + Math.random() * 40 - 20, startY + Math.random() * 60);
                pitchCtx.stroke();
            }}
            
            const pitchTexture = new THREE.CanvasTexture(pitchCanvas);
            const pitchMaterial = new THREE.MeshStandardMaterial({{ 
                map: pitchTexture,
                roughness: 0.8,
                metalness: 0.0
            }});
            const pitch = new THREE.Mesh(pitchGeometry, pitchMaterial);
            pitch.rotation.x = -Math.PI / 2;
            pitch.position.set(0, -0.04, 11);
            pitch.receiveShadow = true;
            pitch.castShadow = false;
            scene.add(pitch);
            
            // Pitch markings - white creases
            const creaseMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.7,
                emissive: 0xffffff,
                emissiveIntensity: 0.2
            }});
            
            // Bowling creases
            const creaseGeometry = new THREE.PlaneGeometry(2.7, 0.08);
            const crease1 = new THREE.Mesh(creaseGeometry, creaseMaterial);
            crease1.rotation.x = -Math.PI / 2;
            crease1.position.set(0, -0.03, 0);
            crease1.receiveShadow = true;
            scene.add(crease1);
            
            const crease2 = new THREE.Mesh(creaseGeometry, creaseMaterial);
            crease2.rotation.x = -Math.PI / 2;
            crease2.position.set(0, -0.03, 22);
            crease2.receiveShadow = true;
            scene.add(crease2);
            
            // Popping creases (4 feet in front of stumps)
            const poppingCreaseGeometry = new THREE.PlaneGeometry(2.7, 0.06);
            const poppingCrease1 = new THREE.Mesh(poppingCreaseGeometry, creaseMaterial);
            poppingCrease1.rotation.x = -Math.PI / 2;
            poppingCrease1.position.set(0, -0.03, 1.22);
            scene.add(poppingCrease1);
            
            const poppingCrease2 = new THREE.Mesh(poppingCreaseGeometry, creaseMaterial);
            poppingCrease2.rotation.x = -Math.PI / 2;
            poppingCrease2.position.set(0, -0.03, 20.78);
            scene.add(poppingCrease2);
            
            // Return creases (perpendicular lines)
            const returnCreaseGeometry = new THREE.PlaneGeometry(0.06, 2.44);
            for (let x of [-1.35, 1.35]) {{
                const returnCrease1 = new THREE.Mesh(returnCreaseGeometry, creaseMaterial);
                returnCrease1.rotation.x = -Math.PI / 2;
                returnCrease1.position.set(x, -0.03, 0);
                scene.add(returnCrease1);
                
                const returnCrease2 = new THREE.Mesh(returnCreaseGeometry, creaseMaterial);
                returnCrease2.rotation.x = -Math.PI / 2;
                returnCrease2.position.set(x, -0.03, 22);
                scene.add(returnCrease2);
            }}
            
            // Define length zones with 3D appearance
            const zones = [
                {{ name: 'YORKER', start: 20, end: 22, color: 0x4dabf7, label: 'YORKER', yPos: 21 }},
                {{ name: 'FULL', start: 16, end: 20, color: 0x6bcf7f, label: 'FULL', yPos: 18 }},
                {{ name: 'LENGTH', start: 10, end: 16, color: 0xffd93d, label: 'LENGTH', yPos: 13 }},
                {{ name: 'SHORT', start: 4, end: 10, color: 0xff6b6b, label: 'SHORT', yPos: 7 }}
            ];
            
            // Create 3D zone blocks (semi-transparent to show pitch)
            zones.forEach(zone => {{
                const height = zone.end - zone.start;
                const geometry = new THREE.BoxGeometry(2.8, 0.12, height);
                const material = new THREE.MeshStandardMaterial({{ 
                    color: zone.color,
                    transparent: true,
                    opacity: 0.5,
                    roughness: 0.5,
                    metalness: 0.2
                }});
                const zoneMesh = new THREE.Mesh(geometry, material);
                zoneMesh.position.set(0, 0.06, zone.start + height/2);
                zoneMesh.receiveShadow = true;
                zoneMesh.castShadow = false;
                scene.add(zoneMesh);
                
                // Zone labels
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 128;
                context.fillStyle = 'white';
                context.font = 'bold 48px Arial';
                context.textAlign = 'center';
                context.fillText(zone.label, 128, 80);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ 
                    map: texture,
                    transparent: true,
                    opacity: 0.9
                }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.position.set(-2.5, 0.5, zone.yPos);
                sprite.scale.set(2, 1, 1);
                scene.add(sprite);
            }});
            
            // Create realistic wooden stumps
            const stumpMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.7,
                metalness: 0.1
            }});
            
            const stumpPositions = [-0.115, 0, 0.115];
            
            // Bowler's end stumps
            for (let x of stumpPositions) {{
                const stumpGeometry = new THREE.CylinderGeometry(0.022, 0.022, 0.71, 16);
                const stump = new THREE.Mesh(stumpGeometry, stumpMaterial);
                stump.position.set(x, 0.355, 0);
                stump.castShadow = true;
                stump.receiveShadow = true;
                scene.add(stump);
            }}
            
            // Batter's end stumps
            for (let x of stumpPositions) {{
                const stumpGeometry = new THREE.CylinderGeometry(0.022, 0.022, 0.71, 16);
                const stump = new THREE.Mesh(stumpGeometry, stumpMaterial);
                stump.position.set(x, 0.355, 22);
                stump.castShadow = true;
                stump.receiveShadow = true;
                scene.add(stump);
            }}
            
            // Bails on top of stumps
            const bailMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.6,
                metalness: 0.2
            }});
            
            for (let i = 0; i < 2; i++) {{
                const x = i === 0 ? -0.0575 : 0.0575;
                
                // Bowler's end bails
                const bail1 = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.012, 0.012, 0.115, 16),
                    bailMaterial
                );
                bail1.rotation.z = Math.PI / 2;
                bail1.position.set(x, 0.73, 0);
                bail1.castShadow = true;
                scene.add(bail1);
                
                // Batter's end bails
                const bail2 = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.012, 0.012, 0.115, 16),
                    bailMaterial
                );
                bail2.rotation.z = Math.PI / 2;
                bail2.position.set(x, 0.73, 22);
                bail2.castShadow = true;
                scene.add(bail2);
            }}
            
            // Pitch center line (visual guide)
            const lineGeometry = new THREE.BoxGeometry(0.08, 0.01, 22);
            const lineMaterial = new THREE.MeshStandardMaterial({{ color: 0x9b59b6 }});
            const centerLine = new THREE.Mesh(lineGeometry, lineMaterial);
            centerLine.position.set(0, 0.16, 11);
            scene.add(centerLine);
            
            // Crease lines (white)
            const creaseMaterial = new THREE.MeshStandardMaterial({{ color: 0xffffff }});
            for (let z of [0, 22]) {{
                const creaseGeometry = new THREE.BoxGeometry(2.8, 0.02, 0.1);
                const crease = new THREE.Mesh(creaseGeometry, creaseMaterial);
                crease.position.set(0, 0.16, z);
                scene.add(crease);
            }}
            
            // Create balls
            const colorMap = {{
                'red': 0xff0000,
                'purple': 0x9c27b0,
                'green': 0x00ff00,
                'blue': 0x2196f3,
                'gray': 0x808080
            }};
            
            pitchData.forEach(ball => {{
                const radius = ball.size * 0.025;
                const geometry = new THREE.SphereGeometry(radius, 12, 12);
                const material = new THREE.MeshStandardMaterial({{ 
                    color: colorMap[ball.color],
                    roughness: 0.5,
                    metalness: 0.3,
                    emissive: colorMap[ball.color],
                    emissiveIntensity: 0.3
                }});
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(ball.x, radius + 0.15, ball.y);
                sphere.castShadow = true;
                scene.add(sphere);
            }});
            
            // Calculate zone percentages
            const zoneCounts = {{
                'YORKER': pitchData.filter(d => d.y >= 20 && d.y <= 22).length,
                'FULL': pitchData.filter(d => d.y >= 16 && d.y < 20).length,
                'LENGTH': pitchData.filter(d => d.y >= 10 && d.y < 16).length,
                'SHORT': pitchData.filter(d => d.y >= 4 && d.y < 10).length
            }};
            
            const total = pitchData.length;
            const statsHtml = [
                {{ name: 'YORKER', count: zoneCounts.YORKER, class: 'yorker' }},
                {{ name: 'FULL', count: zoneCounts.FULL, class: 'full' }},
                {{ name: 'LENGTH', count: zoneCounts.LENGTH, class: 'length' }},
                {{ name: 'SHORT', count: zoneCounts.SHORT, class: 'short' }}
            ].map(stat => {{
                const percentage = total > 0 ? ((stat.count / total) * 100).toFixed(0) : 0;
                return `
                    <div class="zone-stat">
                        <span class="zone-name ${{stat.class}}\">${{stat.name}}</span>
                        <span class="zone-percentage ${{stat.class}}\">${{percentage}}%</span>
                    </div>
                `;
            }}).join('');
            
            document.getElementById('zone-stats-{unique_id}').innerHTML = statsHtml;
            
            // Camera animation helper
            function animateCamera(targetPos, targetLookAt, duration = 1000) {{
                const startPos = camera.position.clone();
                const startTarget = controls.target.clone();
                const startTime = Date.now();
                
                function animate() {{
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    
                    camera.position.lerpVectors(startPos, targetPos, eased);
                    controls.target.lerpVectors(startTarget, targetLookAt, eased);
                    controls.update();
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animate);
                    }}
                }}
                animate();
            }}
            
            // View angle functions
            window.setTopView_{unique_id} = function() {{
                animateCamera(
                    new THREE.Vector3(0, 35, 11),
                    new THREE.Vector3(0, 0, 11)
                );
            }};
            
            window.setBowlerView_{unique_id} = function() {{
                animateCamera(
                    new THREE.Vector3(0, 12, -8),
                    new THREE.Vector3(0, 0, 11)
                );
            }};
            
            window.setBatterView_{unique_id} = function() {{
                animateCamera(
                    new THREE.Vector3(0, 12, 30),
                    new THREE.Vector3(0, 0, 11)
                );
            }};
            
            window.setSideView_{unique_id} = function() {{
                animateCamera(
                    new THREE.Vector3(25, 15, 11),
                    new THREE.Vector3(0, 0, 11)
                );
            }};
            
            window.resetView_{unique_id} = function() {{
                animateCamera(
                    new THREE.Vector3(0, 15, 25),
                    new THREE.Vector3(0, 0, 11)
                );
            }};
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
    </body>
    </html>
    """
    return html

def render_stumps_view(data, title, width=500, height=600):
    """Render stumps view visualization as 3D with interactive controls like Bowling Length Analysis"""
    data_json = json.dumps(data)
    div_id = f"stumps_view_{uuid.uuid4().hex[:8]}"
    unique_id = uuid.uuid4().hex[:8]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ 
                margin: 0; 
                padding: 20px; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .stumps-container {{ 
                position: relative; 
                text-align: center;
                background: rgba(255,255,255,0.05);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                overflow: visible;
            }}
            .stumps-title {{ 
                text-align: center; 
                font-size: 22px; 
                font-weight: bold; 
                margin-bottom: 15px;
                color: white;
                text-transform: uppercase;
                letter-spacing: 2px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .stats-overlay {{
                position: absolute;
                top: 90px;
                right: 10px;
                background: rgba(0,0,0,0.95);
                padding: 10px 12px;
                border-radius: 10px;
                color: white;
                font-size: 10px;
                min-width: 140px;
                max-width: 180px;
                border: 2px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                display: none;
                transition: all 0.3s ease;
                z-index: 1000;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }}
            .stats-overlay.show {{
                display: block;
                animation: slideIn 0.3s ease-out;
            }}
            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateX(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            .toggle-stats-btn {{
                position: absolute;
                top: 90px;
                right: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 8px 14px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 11px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                z-index: 999;
            }}
            .toggle-stats-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }}
            .toggle-views-btn {{
                position: absolute;
                top: 90px;
                left: 10px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border: none;
                color: white;
                padding: 8px 14px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 11px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
                z-index: 999;
            }}
            .toggle-views-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(240, 147, 251, 0.5);
            }}
            .zone-stat {{
                margin: 5px 0;
                padding: 6px 8px;
                background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-left: 3px solid;
            }}
            .zone-name {{
                font-weight: bold;
                text-transform: uppercase;
                font-size: 9px;
                letter-spacing: 0.5px;
            }}
            .zone-percentage {{
                font-size: 16px;
                font-weight: bold;
            }}
            .wickets {{ color: #ff6b6b; border-color: #ff6b6b; }}
            .boundaries {{ color: #9c27b0; border-color: #9c27b0; }}
            .singles {{ color: #00ff00; border-color: #00ff00; }}
            .twos {{ color: #2196f3; border-color: #2196f3; }}
            .dots {{ color: #808080; border-color: #808080; }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 11px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .view-controls {{
                position: absolute;
                top: 90px;
                left: 10px;
                background: rgba(0,0,0,0.95);
                padding: 10px 12px;
                border-radius: 10px;
                color: white;
                font-size: 10px;
                border: 2px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                display: none;
                z-index: 1000;
                max-width: 150px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }}
            .view-controls.show {{
                display: block;
                animation: slideInLeft 0.3s ease-out;
            }}
            @keyframes slideInLeft {{
                from {{
                    opacity: 0;
                    transform: translateX(-20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            .view-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 6px 10px;
                margin: 3px 0;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
                font-size: 10px;
                transition: all 0.3s ease;
                box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
            }}
            .view-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
            }}
            .controls-title {{
                font-weight: bold;
                margin-bottom: 6px;
                font-size: 11px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        </style>
    </head>
    <body>
        <div class="stumps-container">
            <div class="stumps-title">{title}</div>
            <div id="{div_id}"></div>
            <button class="toggle-views-btn" onclick="toggleViews_{unique_id}()">📐 Views</button>
            <button class="toggle-stats-btn" onclick="toggleStats_{unique_id}()">📊 Statistics</button>
            <div class="view-controls" id="view-controls-{unique_id}">
                <div class="controls-title">📐 VIEWS</div>
                <button class="view-btn" onclick="setFrontView_{unique_id}()">📍 Front</button>
                <button class="view-btn" onclick="setTopView_{unique_id}()">🎯 Top</button>
                <button class="view-btn" onclick="setSideView_{unique_id}()">👁️ Side</button>
                <button class="view-btn" onclick="resetView_{unique_id}()">🔄 Reset</button>
            </div>
            <div class="stats-overlay" id="stats-overlay-{unique_id}">
                <div class="legend-title">📊 Ball Statistics</div>
                <div id="zone-stats-{unique_id}"></div>
            </div>
        </div>
        
        <script>
        (function() {{
            const stumpsData = {data_json};
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);
            
            const camera = new THREE.PerspectiveCamera(45, {width}/{height}, 0.1, 1000);
            camera.position.set(0, 8, 25);
            camera.lookAt(0, 1.5, 0);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize({width}, {height});
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('{div_id}').appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 10;
            controls.maxDistance = 50;
            controls.maxPolarAngle = Math.PI / 2;
            controls.target.set(0, 1.5, 0);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
            sunLight.position.set(5, 15, 10);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            scene.add(sunLight);
            
            const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
            fillLight.position.set(-5, 8, 5);
            scene.add(fillLight);
            
            // Ground plane - Cricket field
            const groundGeometry = new THREE.CircleGeometry(20, 64);
            const groundMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x1a7a1a,
                roughness: 0.8,
                metalness: 0.1
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            scene.add(ground);
            
            // Boundary rope (circular)
            const boundaryRadius = 19.5;
            const ropeSegments = 120;
            const ropeMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.6,
                metalness: 0.3
            }});
            
            for (let i = 0; i < ropeSegments; i++) {{
                const angle1 = (i / ropeSegments) * Math.PI * 2;
                const angle2 = ((i + 1) / ropeSegments) * Math.PI * 2;
                
                const ropeGeometry = new THREE.CylinderGeometry(0.08, 0.08, 0.5, 8);
                const rope = new THREE.Mesh(ropeGeometry, ropeMaterial);
                
                const midAngle = (angle1 + angle2) / 2;
                rope.position.set(
                    Math.cos(midAngle) * boundaryRadius,
                    0.3,
                    Math.sin(midAngle) * boundaryRadius
                );
                rope.rotation.z = Math.PI / 2;
                rope.rotation.y = midAngle;
                scene.add(rope);
            }}
            
            // Sight screens at both ends
            const sightScreenMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x000000,
                roughness: 0.3,
                metalness: 0.1
            }});
            
            // Bowling end sight screen
            const bowlingSightScreen = new THREE.Mesh(
                new THREE.PlaneGeometry(8, 5),
                sightScreenMaterial
            );
            bowlingSightScreen.position.set(0, 2.5, -22);
            scene.add(bowlingSightScreen);
            
            // Batting end sight screen
            const battingSightScreen = new THREE.Mesh(
                new THREE.PlaneGeometry(8, 5),
                sightScreenMaterial
            );
            battingSightScreen.position.set(0, 2.5, 22);
            battingSightScreen.rotation.y = Math.PI;
            scene.add(battingSightScreen);
            
            // Floodlight towers (4 corners)
            const towerPositions = [
                {{ x: 17, z: 17 }},
                {{ x: -17, z: 17 }},
                {{ x: 17, z: -17 }},
                {{ x: -17, z: -17 }}
            ];
            
            const towerMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x333333,
                roughness: 0.7,
                metalness: 0.5
            }});
            
            towerPositions.forEach(pos => {{
                // Tower pole
                const tower = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.15, 0.2, 12, 12),
                    towerMaterial
                );
                tower.position.set(pos.x, 6, pos.z);
                scene.add(tower);
                
                // Floodlight platform
                const platform = new THREE.Mesh(
                    new THREE.BoxGeometry(1.5, 0.3, 1.5),
                    towerMaterial
                );
                platform.position.set(pos.x, 12, pos.z);
                scene.add(platform);
                
                // Lights on platform
                const lightsMaterial = new THREE.MeshStandardMaterial({{ 
                    color: 0xffff99,
                    emissive: 0xffff00,
                    emissiveIntensity: 0.8
                }});
                
                for (let i = -1; i <= 1; i++) {{
                    for (let j = -1; j <= 1; j++) {{
                        const lightBulb = new THREE.Mesh(
                            new THREE.SphereGeometry(0.12, 8, 8),
                            lightsMaterial
                        );
                        lightBulb.position.set(
                            pos.x + i * 0.4,
                            12.2,
                            pos.z + j * 0.4
                        );
                        scene.add(lightBulb);
                    }}
                }}
            }});
            
            // Sponsor boards at strategic positions
            const sponsors = ['VIVO IPL', 'DREAM11', 'CRED', 'TATA'];
            const boardPositions = [
                {{ angle: 0, distance: 20.5 }},
                {{ angle: Math.PI / 2, distance: 20.5 }},
                {{ angle: Math.PI, distance: 20.5 }},
                {{ angle: 3 * Math.PI / 2, distance: 20.5 }}
            ];
            
            boardPositions.forEach((pos, i) => {{
                const boardGeometry = new THREE.PlaneGeometry(3.5, 1.5);
                
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 256;
                const ctx = canvas.getContext('2d');
                
                // Gradient background
                const gradient = ctx.createLinearGradient(0, 0, 512, 0);
                gradient.addColorStop(0, '#0066cc');
                gradient.addColorStop(1, '#004499');
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, 512, 256);
                
                // Sponsor text
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 70px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
                ctx.shadowBlur = 8;
                ctx.fillText(sponsors[i], 256, 128);
                
                const texture = new THREE.CanvasTexture(canvas);
                const boardMaterial = new THREE.MeshStandardMaterial({{ 
                    map: texture,
                    roughness: 0.4
                }});
                
                const board = new THREE.Mesh(boardGeometry, boardMaterial);
                board.position.set(
                    Math.cos(pos.angle) * pos.distance,
                    3,
                    Math.sin(pos.angle) * pos.distance
                );
                board.rotation.y = -pos.angle + Math.PI;
                scene.add(board);
            }});
            
            // Cricket pitch dimensions (22 yards = 20.12m length, 3.05m width)
            const pitchLength = 20.12;
            const pitchWidth = 3.05;
            
            // Pitch surface
            const pitchGeometry = new THREE.PlaneGeometry(pitchWidth, pitchLength);
            const pitchMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xd4a574,
                roughness: 0.9,
                metalness: 0.1
            }});
            const pitch = new THREE.Mesh(pitchGeometry, pitchMaterial);
            pitch.rotation.x = -Math.PI / 2;
            pitch.position.y = 0.01;
            pitch.receiveShadow = true;
            scene.add(pitch);
            
            // Stump specifications (standard cricket dimensions)
            const stumpMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.4,
                metalness: 0.2
            }});
            const stumpHeight = 0.71; // 71cm
            const stumpRadius = 0.02; // 2cm radius
            const stumpSpacing = 0.11; // 11cm between stumps
            const stumpPositions = [-stumpSpacing, 0, stumpSpacing];
            
            // Batting stumps at near end (z = pitchLength/2)
            const battingStumpZ = pitchLength / 2;
            stumpPositions.forEach(x => {{
                const stump = new THREE.Mesh(
                    new THREE.CylinderGeometry(stumpRadius, stumpRadius, stumpHeight, 16), 
                    stumpMaterial
                );
                stump.position.set(x, stumpHeight / 2, battingStumpZ);
                stump.castShadow = true;
                scene.add(stump);
            }});
            
            // Bails on batting stumps
            const bailMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x8B4513,
                roughness: 0.4,
                metalness: 0.2
            }});
            const bailLength = 0.11;
            [-stumpSpacing/2, stumpSpacing/2].forEach(x => {{
                const bail = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.01, 0.01, bailLength, 8),
                    bailMaterial
                );
                bail.rotation.z = Math.PI / 2;
                bail.position.set(x, stumpHeight, battingStumpZ);
                scene.add(bail);
            }});
            
            // Bowling stumps at far end (z = -pitchLength/2)
            const bowlingStumpZ = -pitchLength / 2;
            stumpPositions.forEach(x => {{
                const stump = new THREE.Mesh(
                    new THREE.CylinderGeometry(stumpRadius, stumpRadius, stumpHeight, 16), 
                    stumpMaterial
                );
                stump.position.set(x, stumpHeight / 2, bowlingStumpZ);
                stump.castShadow = true;
                scene.add(stump);
            }});
            
            // Bails on bowling stumps
            [-stumpSpacing/2, stumpSpacing/2].forEach(x => {{
                const bail = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.01, 0.01, bailLength, 8),
                    bailMaterial
                );
                bail.rotation.z = Math.PI / 2;
                bail.position.set(x, stumpHeight, bowlingStumpZ);
                scene.add(bail);
            }});
            
            // Crease lines
            const lineMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xffffff,
                roughness: 0.6
            }});
            
            // Batting crease (at batting end)
            const battingCrease = new THREE.Mesh(
                new THREE.PlaneGeometry(pitchWidth, 0.05),
                lineMaterial
            );
            battingCrease.rotation.x = -Math.PI / 2;
            battingCrease.position.set(0, 0.02, battingStumpZ);
            scene.add(battingCrease);
            
            // Bowling crease (at bowling end)
            const bowlingCrease = new THREE.Mesh(
                new THREE.PlaneGeometry(pitchWidth, 0.05),
                lineMaterial
            );
            bowlingCrease.rotation.x = -Math.PI / 2;
            bowlingCrease.position.set(0, 0.02, bowlingStumpZ);
            scene.add(bowlingCrease);
            
            // Wide line markers (dashed effect with small rectangles)
            // Positioned at approximately 1.2m from center on each side
            const wideLineMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xff0000,
                transparent: true,
                opacity: 0.6
            }});
            
            const wideLineX = 1.2;
            for (let z = -pitchLength/2; z < pitchLength/2; z += 0.4) {{
                // Left wide line
                const leftWide = new THREE.Mesh(
                    new THREE.PlaneGeometry(0.03, 0.2),
                    wideLineMaterial
                );
                leftWide.rotation.x = -Math.PI / 2;
                leftWide.position.set(-wideLineX, 0.02, z);
                scene.add(leftWide);
                
                // Right wide line
                const rightWide = new THREE.Mesh(
                    new THREE.PlaneGeometry(0.03, 0.2),
                    wideLineMaterial
                );
                rightWide.rotation.x = -Math.PI / 2;
                rightWide.position.set(wideLineX, 0.02, z);
                scene.add(rightWide);
            }}
            
            // Color map for balls
            const colorMap = {{
                'red': 0xff0000,
                'purple': 0x9c27b0,
                'green': 0x00ff00,
                'blue': 0x2196f3,
                'gray': 0x808080
            }};
            
            // Statistics
            let stats = {{
                total: stumpsData.length,
                wickets: 0,
                boundaries: 0,
                singles: 0,
                twosThrees: 0,
                dots: 0
            }};
            
            // Draw balls with better distribution across pitch
            stumpsData.forEach(ball => {{
                // Update stats
                switch(ball.color) {{
                    case 'red': stats.wickets++; break;
                    case 'purple': stats.boundaries++; break;
                    case 'green': stats.singles++; break;
                    case 'blue': stats.twosThrees++; break;
                    case 'gray': stats.dots++; break;
                }}
                
                // Enhanced size calculation - make balls much more visible and vary by type
                // Wickets (size 12) -> 0.28, Boundaries (14) -> 0.32, Fours (10) -> 0.24, Singles (6) -> 0.16, Dots (4) -> 0.12
                const radius = ball.size * 0.023;  // Increased from 0.012 to 0.023 for better visibility
                const ballGeometry = new THREE.SphereGeometry(radius, 20, 20);  // Higher segments for smoother appearance
                const ballMaterial = new THREE.MeshStandardMaterial({{ 
                    color: colorMap[ball.color],
                    roughness: 0.3,
                    metalness: 0.6,
                    emissive: colorMap[ball.color],
                    emissiveIntensity: 0.3  // Increased glow for better visibility
                }});
                const sphere = new THREE.Mesh(ballGeometry, ballMaterial);
                
                // Position mapping for better visualization:
                // ball.x: horizontal position (-2 to 2) - maps to line (wide left to wide right)
                // ball.y: vertical position along pitch (0 to 5) - maps to length (bowling end to batting end)
                // ball.z: height above ground
                
                // Map x coordinate: -2 to 2 range maps to -1.4 to 1.4 on pitch (within 3.05m width)
                const xPos = ball.x * 0.7;
                
                // Map y coordinate: 0 to 5 range maps along full pitch length (20.12m)
                // bowling end (-10.06) to batting end (+10.06)
                const zPos = (ball.y * 4.024) - 10.06;
                
                // Height: slightly above pitch surface with slight random variation
                const yPos = radius + 0.05 + (Math.random() * 0.02);  // Lift balls slightly higher
                
                sphere.position.set(xPos, yPos, zPos);
                sphere.castShadow = true;
                sphere.receiveShadow = true;  // Add shadows for better depth perception
                scene.add(sphere);
            }});
            
            // Display statistics
            const statsHtml = `
                <div class="zone-stat wickets">
                    <span class="zone-name">Wickets</span>
                    <span class="zone-percentage">${{stats.wickets}}</span>
                </div>
                <div class="zone-stat boundaries">
                    <span class="zone-name">Boundaries</span>
                    <span class="zone-percentage">${{stats.boundaries}}</span>
                </div>
                <div class="zone-stat singles">
                    <span class="zone-name">Singles</span>
                    <span class="zone-percentage">${{stats.singles}}</span>
                </div>
                <div class="zone-stat twos">
                    <span class="zone-name">2s/3s</span>
                    <span class="zone-percentage">${{stats.twosThrees}}</span>
                </div>
                <div class="zone-stat dots">
                    <span class="zone-name">Dot Balls</span>
                    <span class="zone-percentage">${{stats.dots}}</span>
                </div>
                <div class="zone-stat" style="border-top: 2px solid rgba(255,255,255,0.3); margin-top: 10px; padding-top: 10px;">
                    <span class="zone-name">Total</span>
                    <span class="zone-percentage">${{stats.total}}</span>
                </div>
            `;
            document.getElementById('zone-stats-{unique_id}').innerHTML = statsHtml;
            
            // Camera animation function
            function animateCamera(targetPos, targetLookAt, duration = 1000) {{
                const startPos = {{
                    x: camera.position.x,
                    y: camera.position.y,
                    z: camera.position.z
                }};
                const startTime = Date.now();
                
                function animate() {{
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = progress < 0.5 
                        ? 2 * progress * progress 
                        : -1 + (4 - 2 * progress) * progress;
                    
                    camera.position.x = startPos.x + (targetPos.x - startPos.x) * eased;
                    camera.position.y = startPos.y + (targetPos.y - startPos.y) * eased;
                    camera.position.z = startPos.z + (targetPos.z - startPos.z) * eased;
                    
                    controls.target.set(targetLookAt.x, targetLookAt.y, targetLookAt.z);
                    controls.update();
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animate);
                    }}
                }}
                animate();
            }}
            
            // View preset functions
            window.setFrontView_{unique_id} = () => animateCamera(
                {{ x: 0, y: 8, z: 25 }},
                {{ x: 0, y: 1.5, z: 0 }}
            );
            
            window.setTopView_{unique_id} = () => animateCamera(
                {{ x: 0, y: 30, z: 0 }},
                {{ x: 0, y: 0, z: 0 }}
            );
            
            window.setSideView_{unique_id} = () => animateCamera(
                {{ x: 25, y: 8, z: 0 }},
                {{ x: 0, y: 1.5, z: 0 }}
            );
            
            window.resetView_{unique_id} = () => animateCamera(
                {{ x: 0, y: 8, z: 25 }},
                {{ x: 0, y: 1.5, z: 0 }}
            );
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Toggle statistics overlay
            window.toggleStats_{unique_id} = function() {{
                const statsOverlay = document.getElementById('stats-overlay-{unique_id}');
                statsOverlay.classList.toggle('show');
            }};
            
            // Toggle view controls
            window.toggleViews_{unique_id} = function() {{
                const viewControls = document.getElementById('view-controls-{unique_id}');
                viewControls.classList.toggle('show');
            }};
        }})();
        </script>
    </body>
    </html>
    """
    return html
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #{div_id}-container {{ 
                position: relative; 
                border: 2px solid #ddd; 
                border-radius: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 10px;
            }}
            #{div_id} {{ 
                background: white;
                border-radius: 8px;
            }}
            .title-{div_id} {{
                text-align: center;
                color: white;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .stats-{div_id} {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                font-size: 11px;
                max-width: 220px;
            }}
            .stats-{div_id} h4 {{
                margin: 0 0 10px 0;
                font-size: 13px;
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 5px;
            }}
            .stat-row {{
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 3px 0;
                border-bottom: 1px solid #eee;
            }}
            .stat-label {{ font-weight: 600; }}
            .stat-value {{ color: #667eea; font-weight: bold; }}
            .chart-overlay {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 12px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                max-width: 200px;
            }}
            .chart-overlay h4 {{
                margin: 0 0 8px 0;
                font-size: 12px;
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="{div_id}-container">
            <div class="title-{div_id}">{title}</div>
            <canvas id="{div_id}" width="{width}" height="{height}"></canvas>
            <div class="chart-overlay" id="chart-{div_id}"></div>
            <div class="stats-{div_id}" id="stats-{div_id}"></div>
        </div>
        <script>
        (function() {{
            const stumpsData = {data_json};
            const canvas = document.getElementById('{div_id}');
            const ctx = canvas.getContext('2d');
            const width = {width};
            const height = {height};
            
            // Background
            ctx.fillStyle = '#f0f8ff';
            ctx.fillRect(0, 0, width, height);
            
            // Draw pitch area
            const pitchX = width * 0.1;
            const pitchY = height * 0.1;
            const pitchWidth = width * 0.8;
            const pitchHeight = height * 0.8;
            
            ctx.fillStyle = '#d4a574';
            ctx.fillRect(pitchX, pitchY, pitchWidth, pitchHeight);
            
            // Draw wide line markers (left and right)
            const wideLineLeft = pitchX + pitchWidth * 0.15;
            const wideLineRight = pitchX + pitchWidth * 0.85;
            
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.lineWidth = 3;
            ctx.setLineDash([10, 5]);
            
            // Left wide line
            ctx.beginPath();
            ctx.moveTo(wideLineLeft, pitchY);
            ctx.lineTo(wideLineLeft, pitchY + pitchHeight);
            ctx.stroke();
            
            // Right wide line
            ctx.beginPath();
            ctx.moveTo(wideLineRight, pitchY);
            ctx.lineTo(wideLineRight, pitchY + pitchHeight);
            ctx.stroke();
            
            ctx.setLineDash([]);
            
            // Draw zone boundaries
            const offStumpLine = pitchX + pitchWidth * 0.4;
            const legStumpLine = pitchX + pitchWidth * 0.6;
            
            ctx.strokeStyle = 'rgba(100, 100, 255, 0.3)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            
            // Off stump zone line
            ctx.beginPath();
            ctx.moveTo(offStumpLine, pitchY);
            ctx.lineTo(offStumpLine, pitchY + pitchHeight);
            ctx.stroke();
            
            // Leg stump zone line
            ctx.beginPath();
            ctx.moveTo(legStumpLine, pitchY);
            ctx.lineTo(legStumpLine, pitchY + pitchHeight);
            ctx.stroke();
            
            ctx.setLineDash([]);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.15)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                // Vertical lines (line)
                const x = pitchX + (pitchWidth / 10) * i;
                ctx.beginPath();
                ctx.moveTo(x, pitchY);
                ctx.lineTo(x, pitchY + pitchHeight);
                ctx.stroke();
                
                // Horizontal lines (length)
                const y = pitchY + (pitchHeight / 10) * i;
                ctx.beginPath();
                ctx.moveTo(pitchX, y);
                ctx.lineTo(pitchX + pitchWidth, y);
                ctx.stroke();
            }}
            
            // Draw BATTING stumps at bottom center
            const battingStumpCenterX = width / 2;
            const battingStumpY = height * 0.85;
            const stumpWidth = 6;
            const stumpHeight = 40;
            const stumpSpacing = 15;
            
            ctx.fillStyle = '#8B4513';
            for (let i = -1; i <= 1; i++) {{
                ctx.fillRect(
                    battingStumpCenterX + (i * stumpSpacing) - stumpWidth / 2,
                    battingStumpY - stumpHeight,
                    stumpWidth,
                    stumpHeight
                );
            }}
            
            // Draw bails on batting stumps
            ctx.fillStyle = '#8B4513';
            ctx.fillRect(
                battingStumpCenterX - stumpSpacing - stumpWidth,
                battingStumpY - stumpHeight - 3,
                stumpSpacing * 2 + stumpWidth * 2,
                3
            );
            
            // Draw BOWLING stumps at top center
            const bowlingStumpCenterX = width / 2;
            const bowlingStumpY = height * 0.15;
            
            ctx.fillStyle = '#8B4513';
            for (let i = -1; i <= 1; i++) {{
                ctx.fillRect(
                    bowlingStumpCenterX + (i * stumpSpacing) - stumpWidth / 2,
                    bowlingStumpY,
                    stumpWidth,
                    stumpHeight
                );
            }}
            
            // Draw bails on bowling stumps
            ctx.fillStyle = '#8B4513';
            ctx.fillRect(
                bowlingStumpCenterX - stumpSpacing - stumpWidth,
                bowlingStumpY,
                stumpSpacing * 2 + stumpWidth * 2,
                3
            );
            
            // Label the stumps
            ctx.font = 'bold 12px Arial';
            ctx.fillStyle = '#333';
            ctx.fillText('Bowler End', bowlingStumpCenterX - 35, bowlingStumpY - 10);
            ctx.fillText('Batter End', battingStumpCenterX - 35, battingStumpY + 20);
            
            // Draw batting crease line
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(pitchX, battingStumpY);
            ctx.lineTo(pitchX + pitchWidth, battingStumpY);
            ctx.stroke();
            
            // Draw bowling crease line
            ctx.beginPath();
            ctx.moveTo(pitchX, bowlingStumpY + stumpHeight);
            ctx.lineTo(pitchX + pitchWidth, bowlingStumpY + stumpHeight);
            ctx.stroke();
            
            // Color mapping
            const colorMap = {{
                'red': '#ff0000',
                'purple': '#9c27b0',
                'green': '#00ff00',
                'blue': '#2196f3',
                'gray': '#808080'
            }};
            
            // Calculate statistics and line/length zones
            let stats = {{
                total: stumpsData.length,
                wickets: 0,
                boundaries: 0,
                singles: 0,
                twosThrees: 0,
                dots: 0,
                lineZones: {{ wide_left: 0, off: 0, middle: 0, leg: 0, wide_right: 0 }},
                lengthZones: {{ full: 0, good: 0, short: 0 }}
            }};
            
            // Draw balls and collect stats
            stumpsData.forEach(ball => {{
                // Update stats
                switch(ball.color) {{
                    case 'red': stats.wickets++; break;
                    case 'purple': stats.boundaries++; break;
                    case 'green': stats.singles++; break;
                    case 'blue': stats.twosThrees++; break;
                    case 'gray': stats.dots++; break;
                }}
                
                // Calculate line zones (-3 to 3 range)
                if (ball.x < -1.5) stats.lineZones.wide_left++;
                else if (ball.x < -0.5) stats.lineZones.off++;
                else if (ball.x < 0.5) stats.lineZones.middle++;
                else if (ball.x < 1.5) stats.lineZones.leg++;
                else stats.lineZones.wide_right++;
                
                // Calculate length zones (z: -2 to 2 range)
                if (ball.z > 0.8) stats.lengthZones.short++;
                else if (ball.z > -0.5) stats.lengthZones.good++;
                else stats.lengthZones.full++;
                
                // Convert coordinates to canvas space
                // x: -3 to 3 -> horizontal position
                // z: -2 to 2 -> vertical position (height)
                const canvasX = pitchX + pitchWidth / 2 + (ball.x / 3) * (pitchWidth / 2);
                const canvasY = pitchY + pitchHeight - ((ball.z + 2) / 4) * pitchHeight;
                
                const radius = ball.size * 2;
                
                // Draw ball shadow
                ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
                ctx.beginPath();
                ctx.arc(canvasX + 2, canvasY + 2, radius, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw ball
                ctx.fillStyle = colorMap[ball.color] || '#808080';
                ctx.beginPath();
                ctx.arc(canvasX, canvasY, radius, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw ball outline
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }});
            
            // Draw mini bar chart on canvas (bottom left)
            const chartX = pitchX;
            const chartY = pitchY + pitchHeight - 120;
            const chartWidth = 150;
            const chartHeight = 100;
            const barSpacing = 4;
            const barWidth = (chartWidth - barSpacing * 4) / 5;
            
            // Chart background
            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 2;
            ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
            
            // Chart title
            ctx.fillStyle = '#333';
            ctx.font = 'bold 11px Arial';
            ctx.fillText('Ball Distribution', chartX + 5, chartY + 12);
            
            // Draw bars
            const maxCount = Math.max(stats.wickets, stats.boundaries, stats.singles, stats.twosThrees, stats.dots);
            const barData = [
                {{ count: stats.wickets, color: '#ff0000', label: 'W' }},
                {{ count: stats.boundaries, color: '#9c27b0', label: 'B' }},
                {{ count: stats.singles, color: '#00ff00', label: '1' }},
                {{ count: stats.twosThrees, color: '#2196f3', label: '2/3' }},
                {{ count: stats.dots, color: '#808080', label: 'D' }}
            ];
            
            barData.forEach((bar, index) => {{
                const barHeight = maxCount > 0 ? (bar.count / maxCount) * 60 : 0;
                const x = chartX + 10 + index * (barWidth + barSpacing);
                const y = chartY + chartHeight - 25 - barHeight;
                
                // Draw bar
                ctx.fillStyle = bar.color;
                ctx.fillRect(x, y, barWidth, barHeight);
                
                // Draw count on top of bar
                ctx.fillStyle = '#333';
                ctx.font = 'bold 9px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(bar.count, x + barWidth / 2, y - 3);
                
                // Draw label
                ctx.fillStyle = '#333';
                ctx.font = '9px Arial';
                ctx.fillText(bar.label, x + barWidth / 2, chartY + chartHeight - 10);
            }});
            
            ctx.textAlign = 'left';
            
            // Draw legend
            const legendX = 20;
            let legendY = 30;
            const legendSize = 10;
            const legendSpacing = 25;
            
            ctx.font = 'bold 14px Arial';
            ctx.fillStyle = '#333';
            ctx.fillText('Ball Types:', legendX, legendY);
            legendY += 20;
            
            ctx.font = '12px Arial';
            const legend = [
                {{ color: 'red', label: 'Wicket' }},
                {{ color: 'purple', label: 'Boundary (4/6)' }},
                {{ color: 'green', label: 'Single' }},
                {{ color: 'blue', label: 'Two/Three' }},
                {{ color: 'gray', label: 'Dot Ball' }}
            ];
            
            legend.forEach(item => {{
                ctx.fillStyle = colorMap[item.color];
                ctx.beginPath();
                ctx.arc(legendX + legendSize, legendY, legendSize, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.fillStyle = '#333';
                ctx.fillText(item.label, legendX + legendSize * 3, legendY + 4);
                legendY += legendSpacing;
            }});
            
            // Draw zone labels with wide markers
            ctx.font = 'bold 12px Arial';
            ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
            ctx.fillText('WIDE LINE', pitchX + 5, pitchY + 15);
            ctx.fillText('WIDE LINE', pitchX + pitchWidth - 65, pitchY + 15);
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.font = 'bold 11px Arial';
            ctx.fillText('Wide Zone', pitchX + 10, pitchY + pitchHeight / 2);
            ctx.fillText('Off Stump', pitchX + pitchWidth * 0.25, pitchY + pitchHeight * 0.3);
            ctx.fillText('Middle', pitchX + pitchWidth / 2 - 20, pitchY + pitchHeight * 0.3);
            ctx.fillText('Leg Side', pitchX + pitchWidth * 0.65, pitchY + pitchHeight * 0.3);
            ctx.fillText('Wide Zone', pitchX + pitchWidth - 55, pitchY + pitchHeight / 2);
            
            ctx.font = 'bold 11px Arial';
            ctx.fillText('Full', pitchX + pitchWidth + 10, pitchY + pitchHeight * 0.75);
            ctx.fillText('Good', pitchX + pitchWidth + 10, pitchY + pitchHeight * 0.5);
            ctx.fillText('Short', pitchX + pitchWidth + 10, pitchY + pitchHeight * 0.25);
            
            // Display line and length zone statistics in overlay
            const chartDiv = document.getElementById('chart-{div_id}');
            chartDiv.innerHTML = `
                <h4>📍 Line Distribution</h4>
                <div style="font-size: 10px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Wide Left:</span>
                        <span style="font-weight: bold; color: #ff0000;">${{stats.lineZones.wide_left}}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Off Stump:</span>
                        <span style="font-weight: bold; color: #2196f3;">${{stats.lineZones.off}}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Middle:</span>
                        <span style="font-weight: bold; color: #00ff00;">${{stats.lineZones.middle}}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Leg Side:</span>
                        <span style="font-weight: bold; color: #9c27b0;">${{stats.lineZones.leg}}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Wide Right:</span>
                        <span style="font-weight: bold; color: #ff0000;">${{stats.lineZones.wide_right}}</span>
                    </div>
                </div>
                <h4 style="margin-top: 10px;">📏 Length Distribution</h4>
                <div style="font-size: 10px;">
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Full:</span>
                        <span style="font-weight: bold; color: #00ff00;">${{stats.lengthZones.full}} (${{((stats.lengthZones.full/stats.total)*100).toFixed(0)}}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Good Length:</span>
                        <span style="font-weight: bold; color: #ffd700;">${{stats.lengthZones.good}} (${{((stats.lengthZones.good/stats.total)*100).toFixed(0)}}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                        <span>Short:</span>
                        <span style="font-weight: bold; color: #ff0000;">${{stats.lengthZones.short}} (${{((stats.lengthZones.short/stats.total)*100).toFixed(0)}}%)</span>
                    </div>
                </div>
            `;
            
            // Display statistics in the stats box with visual bars
            const statsDiv = document.getElementById('stats-{div_id}');
            
            const getPercentage = (value) => ((value / stats.total) * 100).toFixed(1);
            const getBarWidth = (value) => Math.max((value / stats.total) * 100, 2);
            
            statsDiv.innerHTML = `
                <h4>📊 Ball Statistics</h4>
                <div class="stat-row">
                    <span class="stat-label">Total Balls:</span>
                    <span class="stat-value">${{stats.total}}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Wickets:</span>
                    <span class="stat-value" style="color: #ff0000;">${{stats.wickets}} (${{getPercentage(stats.wickets)}}%)</span>
                </div>
                <div style="background: linear-gradient(90deg, #ff0000 ${{getBarWidth(stats.wickets)}}%, #eee ${{getBarWidth(stats.wickets)}}%); height: 4px; border-radius: 2px; margin: 2px 0 8px 0;"></div>
                
                <div class="stat-row">
                    <span class="stat-label">Boundaries:</span>
                    <span class="stat-value" style="color: #9c27b0;">${{stats.boundaries}} (${{getPercentage(stats.boundaries)}}%)</span>
                </div>
                <div style="background: linear-gradient(90deg, #9c27b0 ${{getBarWidth(stats.boundaries)}}%, #eee ${{getBarWidth(stats.boundaries)}}%); height: 4px; border-radius: 2px; margin: 2px 0 8px 0;"></div>
                
                <div class="stat-row">
                    <span class="stat-label">Singles:</span>
                    <span class="stat-value" style="color: #00ff00;">${{stats.singles}} (${{getPercentage(stats.singles)}}%)</span>
                </div>
                <div style="background: linear-gradient(90deg, #00ff00 ${{getBarWidth(stats.singles)}}%, #eee ${{getBarWidth(stats.singles)}}%); height: 4px; border-radius: 2px; margin: 2px 0 8px 0;"></div>
                
                <div class="stat-row">
                    <span class="stat-label">2s/3s:</span>
                    <span class="stat-value" style="color: #2196f3;">${{stats.twosThrees}} (${{getPercentage(stats.twosThrees)}}%)</span>
                </div>
                <div style="background: linear-gradient(90deg, #2196f3 ${{getBarWidth(stats.twosThrees)}}%, #eee ${{getBarWidth(stats.twosThrees)}}%); height: 4px; border-radius: 2px; margin: 2px 0 8px 0;"></div>
                
                <div class="stat-row">
                    <span class="stat-label">Dot Balls:</span>
                    <span class="stat-value" style="color: #808080;">${{stats.dots}} (${{getPercentage(stats.dots)}}%)</span>
                </div>
                <div style="background: linear-gradient(90deg, #808080 ${{getBarWidth(stats.dots)}}%, #eee ${{getBarWidth(stats.dots)}}%); height: 4px; border-radius: 2px; margin: 2px 0 8px 0;"></div>
                
                <div class="stat-row" style="border-top: 2px solid #667eea; margin-top: 8px; padding-top: 8px;">
                    <span class="stat-label">Dot Ball %:</span>
                    <span class="stat-value">${{((stats.dots / stats.total) * 100).toFixed(1)}}%</span>
                </div>
            `;
        }})();
        </script>
    </body>
    </html>
    """
    return html

def render_advanced_pitch_viz(data, title, width=1200, height=450):
    """Render advanced 4-panel pitch visualization with heat maps"""
    import json
    
    if not data:
        return "<p>No data available</p>"
    
    data_json = json.dumps(data)
    div_id = f"advanced_pitch_{uuid.uuid4().hex[:8]}"
    
    # Separate wickets and non-wickets
    wickets = [d for d in data if d['wicket'] == 1]
    hitting = [d for d in data if d['wicket'] == 0]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            .viz-container-{div_id} {{ 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 15px; 
                margin: 20px 0;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }}
            .viz-item-{div_id} {{ 
                border: 2px solid #e0e0e0; 
                border-radius: 10px; 
                padding: 10px;
                background: white;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .viz-item-{div_id}:hover {{ 
                transform: translateY(-4px);
                box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                border-color: #667eea;
            }}
            .viz-subtitle-{div_id} {{ 
                text-align: center; 
                font-size: 14px; 
                font-weight: bold; 
                margin-bottom: 10px;
                color: #333;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        </style>
    </head>
    <body>
        <div class="viz-container-{div_id}">
            <div class="viz-item-{div_id}">
                <div class="viz-subtitle-{div_id}">Wickets</div>
                <div id="plot1_{div_id}"></div>
            </div>
            <div class="viz-item-{div_id}">
                <div class="viz-subtitle-{div_id}">Hitting</div>
                <div id="plot2_{div_id}"></div>
            </div>
            <div class="viz-item-{div_id}">
                <div class="viz-subtitle-{div_id}">Density Heat Map</div>
                <div id="plot3_{div_id}"></div>
            </div>
            <div class="viz-item-{div_id}">
                <div class="viz-subtitle-{div_id}">Combined</div>
                <div id="plot4_{div_id}"></div>
            </div>
        </div>
        
        <script>
        (function() {{
            const allData = {data_json};
            const wickets = allData.filter(d => d.wicket === 1);
            const hitting = allData.filter(d => d.wicket === 0);
            
            const commonLayout = {{
                width: 280,
                height: 380,
                margin: {{ t: 5, r: 10, b: 25, l: 30 }},
                xaxis: {{ 
                    range: [-1.5, 1.5], 
                    showgrid: true, 
                    gridcolor: 'rgba(255, 255, 255, 0.3)',
                    gridwidth: 1,
                    zeroline: false,
                    title: '',
                    tickfont: {{ size: 9 }}
                }},
                yaxis: {{ 
                    range: [0, 22], 
                    showgrid: true,
                    gridcolor: 'rgba(255, 255, 255, 0.3)',
                    gridwidth: 1,
                    zeroline: false,
                    title: '',
                    tickfont: {{ size: 9 }}
                }},
                plot_bgcolor: '#d4a574',
                paper_bgcolor: 'white',
                showlegend: false,
                images: [
                    // Pitch center strip (lighter color for worn area)
                    {{
                        source: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMSIgaGVpZ2h0PSIxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNjNDk1NjQiLz48L3N2Zz4=',
                        xref: 'x',
                        yref: 'y',
                        x: -0.5,
                        y: 22,
                        sizex: 1,
                        sizey: 22,
                        sizing: 'stretch',
                        opacity: 0.6,
                        layer: 'below'
                    }}
                ]
            }};
            
            // Cricket stumps (3 vertical lines for each end)
            const stumpHeight = 0.3;
            
            // Batting end stumps (bottom)
            const battingStumps = [
                {{ x: [-0.11, -0.11], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [0, 0], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [0.11, 0.11], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                // Bails
                {{ x: [-0.11, 0.11], y: [stumpHeight, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 3 }}, hoverinfo: 'skip' }}
            ];
            
            // Bowling end stumps (top)
            const bowlingStumps = [
                {{ x: [-0.11, -0.11], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [0, 0], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [0.11, 0.11], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 4 }}, hoverinfo: 'skip' }},
                // Bails
                {{ x: [-0.11, 0.11], y: [22 - stumpHeight, 22 - stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 3 }}, hoverinfo: 'skip' }}
            ];
            
            // Crease lines
            const creaseLines = [
                // Batting crease (bottom)
                {{ x: [-0.6, 0.6], y: [0, 0], mode: 'lines', line: {{ color: 'white', width: 3 }}, hoverinfo: 'skip' }},
                // Bowling crease (top)
                {{ x: [-0.6, 0.6], y: [22, 22], mode: 'lines', line: {{ color: 'white', width: 3 }}, hoverinfo: 'skip' }},
                // Popping crease markers (dashed)
                {{ x: [-0.6, 0.6], y: [0.5, 0.5], mode: 'lines', line: {{ color: 'white', width: 2, dash: 'dot' }}, hoverinfo: 'skip' }},
                {{ x: [-0.6, 0.6], y: [21.5, 21.5], mode: 'lines', line: {{ color: 'white', width: 2, dash: 'dot' }}, hoverinfo: 'skip' }}
            ];
            
            // Plot 1: Wickets
            const wicketsTrace = {{
                x: wickets.map(d => d.x),
                y: wickets.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 7,
                    color: '#ef5350',
                    opacity: 0.7,
                    line: {{ width: 1, color: 'white' }}
                }},
                hovertemplate: '<b>WICKET</b><br>%{{text}}<extra></extra>',
                text: wickets.map(d => d.batter)
            }};
            
            Plotly.newPlot('plot1_{div_id}', [wicketsTrace, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false}});
            
            // Plot 2: Hitting
            const hittingTrace = {{
                x: hitting.map(d => d.x),
                y: hitting.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 5,
                    color: '#2196f3',
                    opacity: 0.6,
                    line: {{ width: 0.5, color: 'white' }}
                }},
                hovertemplate: '<b>%{{text}} run(s)</b><extra></extra>',
                text: hitting.map(d => d.runs)
            }};
            
            Plotly.newPlot('plot2_{div_id}', [hittingTrace, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false}});
            
            // Plot 3: Heat map
            const heatmapTrace = {{
                x: allData.map(d => d.x),
                y: allData.map(d => d.y),
                type: 'histogram2dcontour',
                colorscale: [
                    [0, '#d4a574'],
                    [0.2, '#c49564'],
                    [0.4, '#ffeb3b'],
                    [0.6, '#ff9800'],
                    [0.8, '#ff5722'],
                    [1, '#b71c1c']
                ],
                showscale: true,
                colorbar: {{
                    len: 0.6,
                    thickness: 8,
                    x: 1.02,
                    tickfont: {{ size: 8 }}
                }},
                contours: {{
                    coloring: 'heatmap',
                    showlabels: false
                }},
                hoverinfo: 'skip'
            }};
            
            Plotly.newPlot('plot3_{div_id}', [heatmapTrace, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false}});
            
            // Plot 4: Combined
            const wicketsCombined = {{
                x: wickets.map(d => d.x),
                y: wickets.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 7,
                    color: '#ef5350',
                    opacity: 0.8
                }},
                hovertemplate: '<b>WICKET</b><br>%{{text}}<extra></extra>',
                text: wickets.map(d => d.batter)
            }};
            
            const hittingCombined = {{
                x: hitting.map(d => d.x),
                y: hitting.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 5,
                    color: '#ff9800',
                    opacity: 0.5
                }},
                hovertemplate: '<b>%{{text}} runs</b><extra></extra>',
                text: hitting.map(d => d.runs)
            }};
            
            Plotly.newPlot('plot4_{div_id}', [hittingCombined, wicketsCombined, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false}});
        }})();
        </script>
                    gridwidth: 1,
                    zeroline: false,
                    title: {{ text: 'Length (yards)', font: {{ size: 11, color: '#666' }} }},
                    tickfont: {{ size: 10 }}
                }},
                plot_bgcolor: '#d4a574',
                paper_bgcolor: 'white',
                showlegend: false,
                images: [pitchGradient],
                hovermode: 'closest'
            }};
            
            // Cricket stumps with enhanced visualization
            const stumpHeight = 0.35;
            
            // Batting end stumps (bottom)
            const battingStumps = [
                {{ x: [-0.11, -0.11], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [0, 0], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [0.11, 0.11], y: [0, stumpHeight], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [-0.11, 0.11], y: [stumpHeight, stumpHeight], mode: 'lines', line: {{ color: '#D2691E', width: 4 }}, hoverinfo: 'skip' }}
            ];
            
            // Bowling end stumps (top)
            const bowlingStumps = [
                {{ x: [-0.11, -0.11], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [0, 0], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [0.11, 0.11], y: [22 - stumpHeight, 22], mode: 'lines', line: {{ color: '#8B4513', width: 5 }}, hoverinfo: 'skip' }},
                {{ x: [-0.11, 0.11], y: [22 - stumpHeight, 22 - stumpHeight], mode: 'lines', line: {{ color: '#D2691E', width: 4 }}, hoverinfo: 'skip' }}
            ];
            
            // Enhanced crease lines
            const creaseLines = [
                {{ x: [-0.8, 0.8], y: [0, 0], mode: 'lines', line: {{ color: 'white', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [-0.8, 0.8], y: [22, 22], mode: 'lines', line: {{ color: 'white', width: 4 }}, hoverinfo: 'skip' }},
                {{ x: [-0.8, 0.8], y: [1.22, 1.22], mode: 'lines', line: {{ color: 'rgba(255, 255, 255, 0.7)', width: 2, dash: 'dash' }}, hoverinfo: 'skip' }},
                {{ x: [-0.8, 0.8], y: [20.78, 20.78], mode: 'lines', line: {{ color: 'rgba(255, 255, 255, 0.7)', width: 2, dash: 'dash' }}, hoverinfo: 'skip' }}
            ];
            
            // Plot 1: Wickets - Enhanced with size variation
            const wicketsTrace = {{
                x: wickets.map(d => d.x),
                y: wickets.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 11,
                    color: '#ef5350',
                    opacity: 0.75,
                    line: {{ width: 2, color: 'white' }},
                    symbol: 'x'
                }},
                hovertemplate: '<b style="color:#ef5350">WICKET!</b><br>' +
                               '<b>Batter:</b> %{{text}}<br>' +
                               '<b>Position:</b> (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                text: wickets.map(d => d.batter),
                name: 'Wickets'
            }};
            
            Plotly.newPlot('plot1_{div_id}', [wicketsTrace, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false, responsive: true}});
            
            // Plot 2: Scoring - Color coded by runs
            const scoringData = hitting.map(d => ({{
                x: d.x,
                y: d.y,
                runs: d.runs,
                batter: d.batter,
                color: d.runs >= 6 ? '#9c27b0' : d.runs === 4 ? '#4caf50' : d.runs > 0 ? '#2196f3' : '#bdbdbd',
                size: d.runs >= 6 ? 13 : d.runs === 4 ? 10 : d.runs > 0 ? 7 : 5
            }}));
            
            // Group by runs for better visualization
            const boundaries6 = scoringData.filter(d => d.runs >= 6);
            const boundaries4 = scoringData.filter(d => d.runs === 4);
            const singles = scoringData.filter(d => d.runs > 0 && d.runs < 4);
            const dotBalls = scoringData.filter(d => d.runs === 0);
            
            const traces2 = [
                {{
                    x: boundaries6.map(d => d.x),
                    y: boundaries6.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 13, color: '#9c27b0', opacity: 0.8, line: {{ width: 2, color: 'white' }} }},
                    hovertemplate: '<b style="color:#9c27b0">SIX!</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: boundaries6.map(d => d.batter),
                    name: 'Sixes'
                }},
                {{
                    x: boundaries4.map(d => d.x),
                    y: boundaries4.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 10, color: '#4caf50', opacity: 0.75, line: {{ width: 1.5, color: 'white' }} }},
                    hovertemplate: '<b style="color:#4caf50">FOUR!</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: boundaries4.map(d => d.batter),
                    name: 'Fours'
                }},
                {{
                    x: singles.map(d => d.x),
                    y: singles.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 7, color: '#2196f3', opacity: 0.65 }},
                    hovertemplate: '<b>%{{customdata}} run(s)</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: singles.map(d => d.batter),
                    customdata: singles.map(d => d.runs),
                    name: 'Singles/Doubles'
                }},
                {{
                    x: dotBalls.map(d => d.x),
                    y: dotBalls.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 5, color: '#bdbdbd', opacity: 0.4 }},
                    hovertemplate: '<b>Dot Ball</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: dotBalls.map(d => d.batter),
                    name: 'Dots'
                }}
            ];
            
            Plotly.newPlot('plot2_{div_id}', [...traces2, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false, responsive: true}});
            
            // Plot 3: Enhanced density heat map with contours
            const heatmapTrace = {{
                x: allData.map(d => d.x),
                y: allData.map(d => d.y),
                type: 'histogram2dcontour',
                colorscale: [
                    [0, 'rgba(212, 165, 116, 0.3)'],
                    [0.25, '#ffeb3b'],
                    [0.5, '#ff9800'],
                    [0.75, '#ff5722'],
                    [1, '#b71c1c']
                ],
                showscale: true,
                colorbar: {{
                    title: {{ text: 'Density', font: {{ size: 10 }} }},
                    len: 0.7,
                    thickness: 12,
                    x: 1.02,
                    tickfont: {{ size: 9 }}
                }},
                contours: {{
                    coloring: 'heatmap',
                    showlabels: true,
                    labelfont: {{ size: 8, color: 'white' }}
                }},
                ncontours: 20,
                hovertemplate: '<b>Density Zone</b><br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>'
            }};
            
            Plotly.newPlot('plot3_{div_id}', [heatmapTrace, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false, responsive: true}});
            
            // Plot 4: Combined view with all outcomes
            const combinedTraces = [
                {{
                    x: hitting.map(d => d.x),
                    y: hitting.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: hitting.map(d => d.runs >= 6 ? 11 : d.runs === 4 ? 8 : d.runs > 0 ? 6 : 4),
                        color: hitting.map(d => d.runs >= 6 ? '#9c27b0' : d.runs === 4 ? '#4caf50' : d.runs > 0 ? '#2196f3' : '#e0e0e0'),
                        opacity: 0.6,
                        line: {{ width: 0.5, color: 'white' }}
                    }},
                    hovertemplate: '<b>%{{customdata}} run(s)</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: hitting.map(d => d.batter),
                    customdata: hitting.map(d => d.runs),
                    name: 'Runs'
                }},
                {{
                    x: wickets.map(d => d.x),
                    y: wickets.map(d => d.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 12,
                        color: '#ef5350',
                        opacity: 0.85,
                        symbol: 'x',
                        line: {{ width: 2, color: 'white' }}
                    }},
                    hovertemplate: '<b style="color:#ef5350">WICKET!</b><br><b>Batter:</b> %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                    text: wickets.map(d => d.batter),
                    name: 'Wickets'
                }}
            ];
            
            Plotly.newPlot('plot4_{div_id}', [...combinedTraces, ...creaseLines, ...battingStumps, ...bowlingStumps], commonLayout, {{displayModeBar: false, responsive: true}});
        }})();
        </script>
    </body>
    </html>
    """
    return html

def render_player_stats_cards(stats_df, title):
    """Render player statistics cards"""
    if stats_df.empty:
        return "<p>No player statistics available</p>"
    
    div_id = f"player_stats_{uuid.uuid4().hex[:8]}"
    
    # Get team color
    team = stats_df['batting_team'].iloc[0] if 'batting_team' in stats_df.columns else None
    team_color = TEAM_COLORS.get(team, '#667eea') if team else '#667eea'
    
    players_data = []
    for idx, row in stats_df.iterrows():
        players_data.append({
            'name': str(row['batter']),
            'runs': int(row['runs_off_bat']),
            'balls': int(row['ball']),
            'sr': float(row['strike_rate']),
            'avg': float(row['average']),
            'fours': int(row['fours']),
            'sixes': int(row['sixes']),
            'dismissals': int(row['is_wicket']),
            'highest': int(row['highest_score'])
        })
    
    data_json = json.dumps(players_data)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .stats-container-{div_id} {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                gap: 12px;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 8px;
            }}
            .player-card-{div_id} {{
                background: linear-gradient(135deg, {team_color} 0%, {team_color}dd 100%);
                border-radius: 10px;
                padding: 16px;
                color: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                transition: transform 0.2s;
                position: relative;
            }}
            .player-card-{div_id}:hover {{
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }}
            .player-rank-{div_id} {{
                position: absolute;
                top: 8px;
                right: 8px;
                background: rgba(255,255,255,0.3);
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-weight: bold;
            }}
            .player-name-{div_id} {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 12px;
                padding-right: 45px;
            }}
            .main-stats-{div_id} {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 12px;
                padding: 10px;
                background: rgba(255,255,255,0.15);
                border-radius: 6px;
            }}
            .stat-item-{div_id} {{
                text-align: center;
            }}
            .stat-value-{div_id} {{
                font-size: 18px;
                font-weight: 700;
                display: block;
                line-height: 1.1;
            }}
            .stat-label-{div_id} {{
                font-size: 9px;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            .boundaries-{div_id} {{
                display: flex;
                gap: 8px;
                margin-bottom: 10px;
            }}
            .boundary-badge-{div_id} {{
                flex: 1;
                background: rgba(255,255,255,0.2);
                padding: 6px;
                border-radius: 5px;
                text-align: center;
            }}
            .boundary-value-{div_id} {{
                font-size: 18px;
                font-weight: bold;
                display: block;
            }}
            .boundary-label-{div_id} {{
                font-size: 10px;
                opacity: 0.9;
            }}
            .secondary-stats-{div_id} {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 6px;
            }}
            .secondary-stat-{div_id} {{
                background: rgba(255,255,255,0.1);
                padding: 6px;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                font-size: 11px;
            }}
        </style>
    </head>
    <body>
        <div class="stats-container-{div_id}" id="container-{div_id}"></div>
        
        <script>
            (function() {{
                const players = {data_json};
                const container = document.getElementById('container-{div_id}');
                
                const solidColors = [
                    {{ bg: '#059669', accent: '#047857' }},
                    {{ bg: '#7c3aed', accent: '#6d28d9' }},
                    {{ bg: '#ea580c', accent: '#c2410c' }},
                    {{ bg: '#0891b2', accent: '#0e7490' }},
                    {{ bg: '#db2777', accent: '#be185d' }},
                    {{ bg: '#65a30d', accent: '#4d7c0f' }},
                    {{ bg: '#4f46e5', accent: '#4338ca' }},
                    {{ bg: '#0d9488', accent: '#0f766e' }}
                ];
                
                players.forEach((player, index) => {{
                    const card = document.createElement('div');
                    card.className = 'player-card-{div_id}';
                    const colorScheme = solidColors[index % solidColors.length];
                    card.style.background = colorScheme.bg;
                    card.style.borderTop = `4px solid ${{colorScheme.accent}}`;
                    card.style.boxShadow = `0 4px 15px ${{colorScheme.accent}}40`;
                    
                    card.innerHTML = `
                        <div class="player-rank-{div_id}">${{index + 1}}</div>
                        <div class="player-name-{div_id}">${{player.name}}</div>
                        
                        <div class="main-stats-{div_id}">
                            <div class="stat-item-{div_id}">
                                <span class="stat-value-{div_id}">${{player.runs}}</span>
                                <span class="stat-label-{div_id}">Runs</span>
                            </div>
                            <div class="stat-item-{div_id}">
                                <span class="stat-value-{div_id}">${{player.balls}}</span>
                                <span class="stat-label-{div_id}">Balls</span>
                            </div>
                            <div class="stat-item-{div_id}">
                                <span class="stat-value-{div_id}">${{player.sr.toFixed(1)}}</span>
                                <span class="stat-label-{div_id}">SR</span>
                            </div>
                        </div>
                        
                        <div class="boundaries-{div_id}">
                            <div class="boundary-badge-{div_id}">
                                <span class="boundary-value-{div_id}">${{player.fours}}</span>
                                <span class="boundary-label-{div_id}">4s</span>
                            </div>
                            <div class="boundary-badge-{div_id}">
                                <span class="boundary-value-{div_id}">${{player.sixes}}</span>
                                <span class="boundary-label-{div_id}">6s</span>
                            </div>
                        </div>
                        
                        <div class="secondary-stats-{div_id}">
                            <div class="secondary-stat-{div_id}">
                                <span>Avg</span>
                                <strong>${{player.avg.toFixed(1)}}</strong>
                            </div>
                            <div class="secondary-stat-{div_id}">
                                <span>Out</span>
                                <strong>${{player.dismissals}}</strong>
                            </div>
                        </div>
                    `;
                    
                    container.appendChild(card);
                }});
            }})();
        </script>
    </body>
    </html>
    """
    return html

# -----------------------------------------------------------------------------
# 4. Main Application
# -----------------------------------------------------------------------------

# Enhanced Header with Branding
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: rgba(255, 255, 255, 0.1); border-radius: 20px; backdrop-filter: blur(10px); margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
    <h1 style="font-size: 4rem; margin: 0; background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; text-shadow: 0 4px 20px rgba(255, 255, 255, 0.3);">
        🏏 IPL Analytics Pro
    </h1>
    <p style="font-size: 1.3rem; color: rgba(255, 255, 255, 0.9); margin-top: 0.5rem; font-weight: 500; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);">
        Professional Cricket Intelligence Platform
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 12px; backdrop-filter: blur(5px);">
            <span style="font-weight: 700; color: #fff;">📊 Advanced Analytics</span>
        </div>
        <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 12px; backdrop-filter: blur(5px);">
            <span style="font-weight: 700; color: #fff;">🎯 Interactive Visualizations</span>
        </div>
        <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 12px; backdrop-filter: blur(5px);">
            <span style="font-weight: 700; color: #fff;">⚡ Real-time Insights</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #667eea; margin-top: 0;">✨ Key Features</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; color: #1e293b;">
        <div>📊 <strong>Phase Analysis</strong> - Run rates across match phases</div>
        <div>🎯 <strong>Advanced Pitch Maps</strong> - 4-panel analysis with heat maps</div>
        <div>🎯 <strong>Stumps View</strong> - Line & length behind bowler</div>
        <div>⚾ <strong>Wagon Wheel</strong> - Shot directions & scoring zones</div>
        <div>📊 <strong>Player Statistics</strong> - Beautiful performance cards</div>
        <div>📈 <strong>Matchup Analysis</strong> - Batter vs bowler insights</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading Data..."):
    df = load_data()

# Sidebar Controls
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 1.5rem;">
    <h2 style="color: white; margin: 0; font-size: 1.8rem; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);">⚙️ Control Panel</h2>
</div>
""", unsafe_allow_html=True)

# Season/Year Filter
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
    <h3 style="color: white; margin: 0; font-size: 1.1rem;">📅 Season Selection</h3>
</div>
""", unsafe_allow_html=True)
filter_mode = "Overall Statistics (2008-Present)"  # Default value
selected_seasons = []  # Default value

if 'season' in df.columns:
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    available_seasons = sorted(df['season'].dropna().unique())
    
    filter_mode = st.sidebar.radio(
        "Filter Mode",
        ["Overall Statistics (2008-Present)", "Specific Season(s)"],
        help="Choose to view all-time stats or filter by specific seasons"
    )
    
    if filter_mode == "Overall Statistics (2008-Present)":
        st.sidebar.info(f"📊 Analyzing data from **{int(min(available_seasons))}** to **{int(max(available_seasons))}**\n\n**Total Seasons:** {len(available_seasons)}")
        filtered_df = df.copy()
    else:
        selected_seasons = st.sidebar.multiselect(
            "Select Season(s)",
            options=available_seasons,
            default=[max(available_seasons)] if available_seasons else [],
            help="Select one or multiple seasons to analyze"
        )
        
        if selected_seasons:
            filtered_df = df[df['season'].isin(selected_seasons)]
            st.sidebar.success(f"✅ Filtered to {len(selected_seasons)} season(s)")
        else:
            st.sidebar.warning("⚠️ No season selected. Showing all data.")
            filtered_df = df.copy()
else:
    filtered_df = df.copy()
    st.sidebar.info("Season data not available in dataset")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
    <h3 style="color: white; margin: 0; font-size: 1.1rem;">🏏 Team Selection</h3>
</div>
""", unsafe_allow_html=True)
# Remove NaN values from teams list
teams = sorted([t for t in filtered_df['batting_team'].unique() if pd.notna(t)])

if len(teams) < 2:
    st.sidebar.error("⚠️ Not enough teams in selected data. Please adjust filters.")
    st.stop()

team1 = st.sidebar.selectbox("Team 1", teams, index=0, help="Select first team for comparison")

# Filter team2 options to exclude team1
team2_options = [t for t in teams if t != team1]
if team2_options:
    team2 = st.sidebar.selectbox("Team 2", team2_options, index=0, help="Select second team for comparison")
else:
    team2 = st.sidebar.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, help="Select second team for comparison")

# Validation check
if team1 == team2:
    st.sidebar.warning("⚠️ Same team selected for both. Results may be identical.")


st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
    <h3 style="color: white; margin: 0; font-size: 1.1rem;">🎯 Bowler Analysis</h3>
</div>
""", unsafe_allow_html=True)
bowler_types = ['All Types', 'Right-Arm Pace', 'Left-Arm Pace', 'Right-Arm Leg Spin', 'Right-Arm Off Spin', 'Left-Arm Orthodox', 'Left-Arm Wrist Spin']
bowler_type = st.sidebar.selectbox("Bowler Type", bowler_types, help="Filter analysis by bowler type")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
    <h3 style="color: white; margin: 0; font-size: 1.1rem;">⏱️ Match Phase Filter</h3>
</div>
""", unsafe_allow_html=True)
phase_options = ['All Phases', 'Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
selected_phase = st.sidebar.selectbox(
    "Match Phase",
    phase_options,
    help="Filter analysis by match phase"
)
phase_filter = None if selected_phase == 'All Phases' else selected_phase

# Update df to be filtered_df for all subsequent analysis
df = filtered_df

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
    <h3 style="color: white; margin: 0; font-size: 1.1rem;">ℹ️ About</h3>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
    <p style="margin: 0; font-weight: 600; color: #667eea; font-size: 1.1rem;">IPL Analytics Pro</p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #475569;">
        Comprehensive cricket analytics with:<br>
        • 3D visualizations<br>
        • Interactive pitch maps<br>
        • Player performance metrics<br>
        • Advanced statistics
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; font-style: italic;">
        Data Source: Cricsheet
    </p>
</div>
""", unsafe_allow_html=True)

# Display current filter summary
st.markdown("---")
if 'season' in df.columns and filter_mode == "Overall Statistics (2008-Present)":
    st.info(f"""
    ### 📊 Currently Analyzing: **Overall Statistics**
    - **Period**: {int(df['season'].min())} to {int(df['season'].max())}
    - **Total Seasons**: {len(df['season'].unique())}
    - **Total Matches**: {df['match_id'].nunique():,}
    - **Total Balls**: {len(df):,}
    - **Teams Compared**: {team1} vs {team2}
    - **Bowler Type**: {bowler_type}
    - **Match Phase**: {selected_phase}
    - **{team1} Balls**: {len(df[df['batting_team'] == team1]):,}
    - **{team2} Balls**: {len(df[df['batting_team'] == team2]):,}
    """)
elif 'season' in df.columns and filter_mode == "Specific Season(s)":
    season_list = ', '.join(map(str, sorted(selected_seasons))) if selected_seasons else "None"
    st.info(f"""
    ### 📊 Currently Analyzing: **{season_list}**
    - **Total Matches**: {df['match_id'].nunique():,}
    - **Total Balls**: {len(df):,}
    - **Teams Compared**: {team1} vs {team2}
    - **Bowler Type**: {bowler_type}
    - **Match Phase**: {selected_phase}
    - **{team1} Balls**: {len(df[df['batting_team'] == team1]):,}
    - **{team2} Balls**: {len(df[df['batting_team'] == team2]):,}
    """)
else:
    st.info(f"""
    ### 📊 Currently Analyzing
    - **Total Matches**: {df['match_id'].nunique():,}
    - **Total Balls**: {len(df):,}
    - **Teams Compared**: {team1} vs {team2}
    - **Bowler Type**: {bowler_type}
    - **Match Phase**: {selected_phase}
    - **{team1} Balls**: {len(df[df['batting_team'] == team1]):,}
    - **{team2} Balls**: {len(df[df['batting_team'] == team2]):,}
    """)

# 1. Phase Analysis
st.markdown("---")
st.markdown("## 📊 Phase Analysis: Run Rate Comparison")
st.markdown(f"**{team1}** vs **{team2}** - Interactive bar chart visualization showing run rates across match phases")

t1_stats = calculate_run_rate_by_phase(df, team1)
t2_stats = calculate_run_rate_by_phase(df, team2)
phases = ['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(f"{team1} Avg Run Rate", f"{t1_stats['run_rate'].mean():.2f}", help="Average run rate across all phases")
with col_b:
    st.metric(f"{team2} Avg Run Rate", f"{t2_stats['run_rate'].mean():.2f}", help="Average run rate across all phases")
with col_c:
    diff = t1_stats['run_rate'].mean() - t2_stats['run_rate'].mean()
    st.metric("Difference", f"{abs(diff):.2f}", delta=f"{diff:.2f}")

st.markdown("")

# Create comparison dataframe
comparison_data = []
for phase in phases:
    t1_phase = t1_stats[t1_stats['phase'] == phase]
    t2_phase = t2_stats[t2_stats['phase'] == phase]
    
    if not t1_phase.empty:
        comparison_data.append({
            'Phase': phase,
            'Team': team1,
            'Run Rate': float(t1_phase['run_rate'].values[0])
        })
    if not t2_phase.empty:
        comparison_data.append({
            'Phase': phase,
            'Team': team2,
            'Run Rate': float(t2_phase['run_rate'].values[0])
        })

comp_df = pd.DataFrame(comparison_data)

if not comp_df.empty:
    import plotly.graph_objects as go
    
    # Get team colors
    team1_color = TEAM_COLORS.get(team1, '#667eea')
    team2_color = TEAM_COLORS.get(team2, '#f093fb')
    
    fig = go.Figure()
    
    # Add bars for each team
    for team in [team1, team2]:
        team_data = comp_df[comp_df['Team'] == team]
        color = team1_color if team == team1 else team2_color
        
        fig.add_trace(go.Bar(
            name=team,
            x=team_data['Phase'],
            y=team_data['Run Rate'],
            marker=dict(
                color=color,
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=team_data['Run Rate'].apply(lambda x: f'{x:.2f}'),
            textposition='outside',
            textfont=dict(size=14, color='black', family='Arial Black'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Phase: %{x}<br>' +
                         'Run Rate: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Calculate proper y-axis range
    max_run_rate = comp_df['Run Rate'].max()
    y_range = [0, max_run_rate * 1.15]  # Add 15% padding at top for text labels
    
    fig.update_layout(
        barmode='group',
        title=dict(
            text='Run Rate Comparison by Phase',
            font=dict(size=20, color='#333', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='Match Phase',
                font=dict(size=16, color='#333', family='Arial')
            ),
            tickfont=dict(size=13, color='#333'),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(
                text='Run Rate',
                font=dict(size=16, color='#333', family='Arial')
            ),
            tickfont=dict(size=13, color='#333'),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            range=y_range,
            zeroline=True
        ),
        plot_bgcolor='rgba(248,249,250,1)',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#333'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ddd',
            borderwidth=1,
            font=dict(size=13)
        ),
        height=450,
        margin=dict(l=60, r=40, t=100, b=60),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 1b. Pitch Maps
st.markdown("---")
st.markdown("## 🎯 Pitch Maps - Ball Landing Positions")
st.markdown("Interactive 3D visualization showing where balls landed on the pitch and their outcomes")

# Color legend bar
st.markdown("""
<div style="display: flex; gap: 15px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin: 15px 0; justify-content: center; flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #e74c3c; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Wickets</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #9b59b6; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Sixes (6)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #2ecc71; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Fours (4)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #3498db; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Singles/Doubles (1-3)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #95a5a6; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Dot Balls (0)</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Understanding Pitch Maps", expanded=False):
    st.markdown("""
    **Color Coding:**
    - 🔴 **Red**: Wickets
    - 🟣 **Purple**: Sixes (6 runs)
    - 🟢 **Green**: Fours (4 runs)
    - 🔵 **Blue**: Singles/Doubles (1-3 runs)
    - ⚫ **Gray**: Dot balls (0 runs)
    
    **How to interact:**
    - Rotate: Click and drag
    - Zoom: Scroll wheel
    - Pan: Right-click and drag
    """)

# Use sidebar phase filter
phase_selection = None if selected_phase == 'All Phases' else selected_phase

st.markdown("")
p1, p2 = st.columns(2)

with p1:
    st.markdown(f"**{team1} Pitch Map**")
    pitch1_data = generate_pitch_map_data(df, team=team1, phase=phase_selection, bowler_type=bowler_type)
    if pitch1_data:
        components.html(render_pitch_map(pitch1_data, f"{team1} - {selected_phase}", 500, 550), height=600)
    else:
        st.info("No pitch data available")

with p2:
    st.markdown(f"**{team2} Pitch Map**")
    pitch2_data = generate_pitch_map_data(df, team=team2, phase=phase_selection, bowler_type=bowler_type)
    if pitch2_data:
        components.html(render_pitch_map(pitch2_data, f"{team2} - {selected_phase}", 500, 550), height=600)
    else:
        st.info("No pitch data available")

# 2. Matchup Analysis
st.markdown("---")
st.markdown(f"## 🎯 Player Matchups vs {bowler_type}")
st.markdown(f"Interactive comparison of top batters' performance against {bowler_type} bowling")

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"### {team1} Top Batters")
    batters1_df = get_top_batters(df, team1, n=5)
    if not batters1_df.empty:
        batters1 = batters1_df['batter'].tolist()
        m1_data = []
        for b in batters1:
            s = calculate_player_matchup(df, b, bowler_type)
            if s: 
                m1_data.append({
                    'Player': str(b), 
                    'Strike Rate': float(s['strike_rate']),
                    'Balls': int(s['balls_faced']),
                    'Runs': int(s['runs_scored']),
                    'Dismissals': int(s['dismissals']),
                    'Average': float(s['average'])
                })
        
        if m1_data:
            matchup_df = pd.DataFrame(m1_data)
            
            # Create enhanced Plotly chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Get team color
            team1_color = TEAM_COLORS.get(team1, '#667eea')
            
            # Add bar chart for Strike Rate
            fig.add_trace(go.Bar(
                x=matchup_df['Player'],
                y=matchup_df['Strike Rate'],
                name='Strike Rate',
                marker=dict(
                    color=team1_color,
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=matchup_df['Strike Rate'].round(1),
                textposition='outside',
                textfont=dict(size=12, color='black', family='Arial Black'),
                hovertemplate='<b>%{{x}}</b><br>' +
                             'Strike Rate: %{{y:.1f}}<br>' +
                             '<extra></extra>',
                yaxis='y'
            ))
            
            # Add line chart for Average
            fig.add_trace(go.Scatter(
                x=matchup_df['Player'],
                y=matchup_df['Average'],
                name='Average',
                mode='lines+markers+text',
                marker=dict(
                    color='#ff6b6b',
                    size=10,
                    line=dict(color='white', width=2)
                ),
                line=dict(color='#ff6b6b', width=3),
                text=matchup_df['Average'].round(1),
                textposition='top center',
                textfont=dict(size=11, color='#ff6b6b', family='Arial Black'),
                hovertemplate='<b>%{{x}}</b><br>' +
                             'Average: %{{y:.1f}}<br>' +
                             '<extra></extra>',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'{team1} Performance vs {bowler_type}',
                    font=dict(size=16, color='#333', family='Arial Black'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text='Player',
                        font=dict(size=14, color='#333')
                    ),
                    tickfont=dict(size=11, color='#333'),
                    tickangle=-45
                ),
                yaxis=dict(
                    title=dict(
                        text='Strike Rate',
                        font=dict(size=14, color=team1_color)
                    ),
                    tickfont=dict(size=11, color=team1_color),
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                yaxis2=dict(
                    title=dict(
                        text='Average',
                        font=dict(size=14, color='#ff6b6b')
                    ),
                    tickfont=dict(size=11, color='#ff6b6b'),
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                plot_bgcolor='rgba(248,249,250,1)',
                paper_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd',
                    borderwidth=1
                ),
                height=400,
                margin=dict(l=60, r=60, t=80, b=100),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced dataframe display
            st.markdown("##### 📊 Detailed Statistics")
            styled_df = matchup_df.style.background_gradient(
                subset=['Strike Rate', 'Average'],
                cmap='RdYlGn',
                vmin=0,
                vmax=matchup_df[['Strike Rate', 'Average']].max().max()
            ).format({
                'Strike Rate': '{:.1f}',
                'Average': '{:.1f}',
                'Runs': '{:.0f}',
                'Balls': '{:.0f}',
                'Dismissals': '{:.0f}'
            })
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        else:
            st.info(f"No data for {team1} vs {bowler_type}")
    else:
        st.warning(f"No batters found for {team1}")

with c2:
    st.markdown(f"### {team2} Top Batters")
    batters2_df = get_top_batters(df, team2, n=5)
    if not batters2_df.empty:
        batters2 = batters2_df['batter'].tolist()
        m2_data = []
        for b in batters2:
            s = calculate_player_matchup(df, b, bowler_type)
            if s: 
                m2_data.append({
                    'Player': str(b), 
                    'Strike Rate': float(s['strike_rate']),
                    'Balls': int(s['balls_faced']),
                    'Runs': int(s['runs_scored']),
                    'Dismissals': int(s['dismissals']),
                    'Average': float(s['average'])
                })
        
        if m2_data:
            matchup_df = pd.DataFrame(m2_data)
            
            # Create enhanced Plotly chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Get team color
            team2_color = TEAM_COLORS.get(team2, '#f093fb')
            
            # Add bar chart for Strike Rate
            fig.add_trace(go.Bar(
                x=matchup_df['Player'],
                y=matchup_df['Strike Rate'],
                name='Strike Rate',
                marker=dict(
                    color=team2_color,
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=matchup_df['Strike Rate'].round(1),
                textposition='outside',
                textfont=dict(size=12, color='black', family='Arial Black'),
                hovertemplate='<b>%{{x}}</b><br>' +
                             'Strike Rate: %{{y:.1f}}<br>' +
                             '<extra></extra>',
                yaxis='y'
            ))
            
            # Add line chart for Average
            fig.add_trace(go.Scatter(
                x=matchup_df['Player'],
                y=matchup_df['Average'],
                name='Average',
                mode='lines+markers+text',
                marker=dict(
                    color='#ff6b6b',
                    size=10,
                    line=dict(color='white', width=2)
                ),
                line=dict(color='#ff6b6b', width=3),
                text=matchup_df['Average'].round(1),
                textposition='top center',
                textfont=dict(size=11, color='#ff6b6b', family='Arial Black'),
                hovertemplate='<b>%{{x}}</b><br>' +
                             'Average: %{{y:.1f}}<br>' +
                             '<extra></extra>',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'{team2} Performance vs {bowler_type}',
                    font=dict(size=16, color='#333', family='Arial Black'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text='Player',
                        font=dict(size=14, color='#333')
                    ),
                    tickfont=dict(size=11, color='#333'),
                    tickangle=-45
                ),
                yaxis=dict(
                    title=dict(
                        text='Strike Rate',
                        font=dict(size=14, color=team2_color)
                    ),
                    tickfont=dict(size=11, color=team2_color),
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                yaxis2=dict(
                    title=dict(
                        text='Average',
                        font=dict(size=14, color='#ff6b6b')
                    ),
                    tickfont=dict(size=11, color='#ff6b6b'),
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                plot_bgcolor='rgba(248,249,250,1)',
                paper_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd',
                    borderwidth=1
                ),
                height=400,
                margin=dict(l=60, r=60, t=80, b=100),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced dataframe display
            st.markdown("##### 📊 Detailed Statistics")
            styled_df = matchup_df.style.background_gradient(
                subset=['Strike Rate', 'Average'],
                cmap='RdYlGn',
                vmin=0,
                vmax=matchup_df[['Strike Rate', 'Average']].max().max()
            ).format({
                'Strike Rate': '{:.1f}',
                'Average': '{:.1f}',
                'Runs': '{:.0f}',
                'Balls': '{:.0f}',
                'Dismissals': '{:.0f}'
            })
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        else:
            st.info(f"No data for {team2} vs {bowler_type}")
    else:
        st.warning(f"No batters found for {team2}")

# 3. Runs Distribution
st.markdown("---")
st.markdown("## 📈 Runs Distribution Analysis")
st.markdown("Comprehensive breakdown of scoring patterns and run composition")

d1, d2 = st.columns(2)

with d1:
    st.markdown(f"### {team1} Scoring Breakdown")
    rd1 = df[df['batting_team'] == team1]['runs_off_bat'].value_counts().sort_index()
    
    # Calculate percentages and create enhanced data
    total_balls_1 = rd1.sum()
    runs_dist_data = []
    for runs, count in rd1.items():
        runs_dist_data.append({
            'Runs': f'{runs} Run{"s" if runs != 1 else ""}',
            'Count': int(count),
            'Percentage': (count / total_balls_1) * 100
        })
    
    dist_df1 = pd.DataFrame(runs_dist_data)
    
    # Create enhanced donut chart with Plotly
    import plotly.graph_objects as go
    
    colors = ['#808080', '#2196f3', '#4caf50', '#ffeb3b', '#ff9800', '#00ff00', '#9c27b0', '#f44336']
    
    fig = go.Figure(data=[go.Pie(
        labels=dist_df1['Runs'],
        values=dist_df1['Count'],
        hole=0.4,
        marker=dict(
            colors=colors[:len(dist_df1)],
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=13, color='white', family='Arial Black'),
        hovertemplate='<b>%{{label}}</b><br>' +
                     'Balls: %{{value}}<br>' +
                     'Percentage: %{{percent}}<br>' +
                     '<extra></extra>'
    )])
    
    team1_color = TEAM_COLORS.get(team1, '#667eea')
    
    fig.update_layout(
        title=dict(
            text=f'{team1} - Runs per Ball',
            font=dict(size=18, color='#333', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[dict(
            text=f'{total_balls_1}<br>Total<br>Balls',
            x=0.5, y=0.5,
            font=dict(size=16, color=team1_color, family='Arial Black'),
            showarrow=False
        )],
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11)
        ),
        height=400,
        margin=dict(l=20, r=120, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.markdown("##### 📊 Scoring Summary")
    col1a, col1b, col1c = st.columns(3)
    
    total_runs_1 = sum([int(k) * v for k, v in rd1.items()])
    boundaries_1 = rd1.get(4, 0) + rd1.get(6, 0)
    dots_1 = rd1.get(0, 0)
    
    with col1a:
        st.metric("Total Runs", f"{total_runs_1:,}", help="Total runs scored")
    with col1b:
        st.metric("Boundaries", f"{boundaries_1:,}", help="Fours + Sixes", delta=f"{(boundaries_1/total_balls_1)*100:.1f}%")
    with col1c:
        st.metric("Dot Balls", f"{dots_1:,}", help="No runs scored", delta=f"{(dots_1/total_balls_1)*100:.1f}%")
    
    # Detailed table
    st.markdown("##### 📋 Detailed Breakdown")
    styled_df1 = dist_df1.style.background_gradient(
        subset=['Percentage'],
        cmap='YlGnBu',
        vmin=0,
        vmax=100
    ).format({
        'Count': '{:,}',
        'Percentage': '{:.2f}%'
    })
    st.dataframe(styled_df1, hide_index=True, use_container_width=True)

with d2:
    st.markdown(f"### {team2} Scoring Breakdown")
    rd2 = df[df['batting_team'] == team2]['runs_off_bat'].value_counts().sort_index()
    
    # Calculate percentages and create enhanced data
    total_balls_2 = rd2.sum()
    runs_dist_data2 = []
    for runs, count in rd2.items():
        runs_dist_data2.append({
            'Runs': f'{runs} Run{"s" if runs != 1 else ""}',
            'Count': int(count),
            'Percentage': (count / total_balls_2) * 100
        })
    
    dist_df2 = pd.DataFrame(runs_dist_data2)
    
    # Create enhanced donut chart with Plotly
    fig2 = go.Figure(data=[go.Pie(
        labels=dist_df2['Runs'],
        values=dist_df2['Count'],
        hole=0.4,
        marker=dict(
            colors=colors[:len(dist_df2)],
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=13, color='white', family='Arial Black'),
        hovertemplate='<b>%{{label}}</b><br>' +
                     'Balls: %{{value}}<br>' +
                     'Percentage: %{{percent}}<br>' +
                     '<extra></extra>'
    )])
    
    team2_color = TEAM_COLORS.get(team2, '#f093fb')
    
    fig2.update_layout(
        title=dict(
            text=f'{team2} - Runs per Ball',
            font=dict(size=18, color='#333', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[dict(
            text=f'{total_balls_2}<br>Total<br>Balls',
            x=0.5, y=0.5,
            font=dict(size=16, color=team2_color, family='Arial Black'),
            showarrow=False
        )],
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11)
        ),
        height=400,
        margin=dict(l=20, r=120, t=60, b=20)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Add summary statistics
    st.markdown("##### 📊 Scoring Summary")
    col2a, col2b, col2c = st.columns(3)
    
    total_runs_2 = sum([int(k) * v for k, v in rd2.items()])
    boundaries_2 = rd2.get(4, 0) + rd2.get(6, 0)
    dots_2 = rd2.get(0, 0)
    
    with col2a:
        st.metric("Total Runs", f"{total_runs_2:,}", help="Total runs scored")
    with col2b:
        st.metric("Boundaries", f"{boundaries_2:,}", help="Fours + Sixes", delta=f"{(boundaries_2/total_balls_2)*100:.1f}%")
    with col2c:
        st.metric("Dot Balls", f"{dots_2:,}", help="No runs scored", delta=f"{(dots_2/total_balls_2)*100:.1f}%")
    
    # Detailed table
    st.markdown("##### 📋 Detailed Breakdown")
    styled_df2 = dist_df2.style.background_gradient(
        subset=['Percentage'],
        cmap='YlGnBu',
        vmin=0,
        vmax=100
    ).format({
        'Count': '{:,}',
        'Percentage': '{:.2f}%'
    })
    st.dataframe(styled_df2, hide_index=True, use_container_width=True)

# 4. Pitch Maps
st.subheader("🎯 Advanced Pitch Maps - Multi-Panel Analysis")
st.markdown("_4-panel view: Wickets, Hitting, Density Heat Map, and Combined_")

# Use the sidebar phase filter
phase_val = None if selected_phase == 'All Phases' else selected_phase

st.markdown(f"### {team1} - Advanced Pitch Analysis")
pitch_data1 = generate_pitch_map_data(df, team=team1, bowler_type=bowler_type, phase=phase_val)
if pitch_data1:
    components.html(render_advanced_pitch_viz(pitch_data1, f"{team1} - {selected_phase}", 1200, 450), height=500)
    st.caption(f"📊 Total deliveries analyzed: {len(pitch_data1)} | Wickets: {sum(1 for d in pitch_data1 if d['wicket'] == 1)}")
else:
    st.info(f"No data available for {team1}")

st.markdown(f"### {team2} - Advanced Pitch Analysis")
pitch_data2 = generate_pitch_map_data(df, team=team2, bowler_type=bowler_type, phase=phase_val)
if pitch_data2:
    components.html(render_advanced_pitch_viz(pitch_data2, f"{team2} - {selected_phase}", 1200, 450), height=500)
    st.caption(f"📊 Total deliveries analyzed: {len(pitch_data2)} | Wickets: {sum(1 for d in pitch_data2 if d['wicket'] == 1)}")
else:
    st.info(f"No data available for {team2}")

# 5. Stumps View
st.subheader("🎯 Stumps View - Line & Length Analysis")
st.markdown("_Behind-the-bowler perspective showing ball position (horizontal & vertical)_")

# Color legend bar for Stumps View
st.markdown("""
<div style="display: flex; gap: 15px; padding: 15px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
            border-radius: 10px; margin: 15px 0; justify-content: center; flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #e74c3c; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Wickets</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #9b59b6; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Sixes</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #2ecc71; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Fours</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #3498db; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">1-3 Runs</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #95a5a6; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Dots</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Add Manim animation button
with st.expander("🎬 View Animated Ball Trajectory Demo (Manim)", expanded=False):
    st.markdown("""
    Click below to generate a **professional Manim animation** showing cricket ball trajectories 
    from the stumps view with different lines and lengths:
    - ✅ **Yorker** (Full length on stumps)
    - ✅ **Off Stump** (Outside off)
    - ✅ **Leg Side** (Inside leg)
    - ✅ **Wide** (Too wide)
    - ✅ **Good Length** (Perfect bowling length)
    """)
    
    if st.button("🎥 Generate Manim Animation", key="manim_btn"):
        with st.spinner("Creating animation... This may take 30-60 seconds..."):
            try:
                video_path = create_manim_animation("cricket_trajectory.mp4")
                if video_path and os.path.exists(video_path):
                    st.success("✅ Animation created successfully!")
                    st.video(video_path)
                    
                    # Offer download
                    with open(video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Animation",
                            data=file,
                            file_name="cricket_ball_trajectory.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("⚠️ Manim library may not be installed. Install with: `pip install manim`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure manim is installed: `pip install manim`")

st.markdown("")
sv1, sv2 = st.columns(2)

with sv1:
    st.markdown(f"**{team1} Stumps View**")
    stumps_data1 = generate_stumps_view_data(df, team=team1, phase=phase_val)
    if stumps_data1:
        components.html(render_stumps_view(stumps_data1, f"{team1} - {selected_phase}", 500, 600), height=650)
    else:
        st.info(f"No data available for {team1}")

with sv2:
    st.markdown(f"**{team2} Stumps View**")
    stumps_data2 = generate_stumps_view_data(df, team=team2, phase=phase_val)
    if stumps_data2:
        components.html(render_stumps_view(stumps_data2, f"{team2} - {selected_phase}", 500, 600), height=650)
    else:
        st.info(f"No data available for {team2}")

# 6. Wagon Wheel (Ground Map)
st.markdown("---")
st.markdown("## ⚾ Wagon Wheel - Shot Directions & Scoring Zones")
st.markdown("Overhead view of the ground showing where runs were scored - **rotate and zoom for better viewing**")

# Color legend bar for Wagon Wheel
st.markdown("""
<div style="display: flex; gap: 15px; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            border-radius: 10px; margin: 15px 0; justify-content: center; flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #ff0000; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">6 Runs</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #ff6600; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">4 Runs</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #ffcc00; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">3 Runs</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #00ff00; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">2 Runs</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 30px; height: 30px; background: #00ccff; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">1 Run</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")
ww1, ww2 = st.columns(2)

with ww1:
    st.markdown(f"**{team1} Wagon Wheel**")
    wagon_data1 = generate_wagon_wheel_data(df, team=team1, phase=phase_val)
    if wagon_data1:
        components.html(render_wagon_wheel(wagon_data1, f"{team1} - {selected_phase}", 600, 600), height=650)
        st.caption(f"📊 Total scoring shots: {len(wagon_data1)}")
    else:
        st.info(f"No data available for {team1}")

with ww2:
    st.markdown(f"**{team2} Wagon Wheel**")
    wagon_data2 = generate_wagon_wheel_data(df, team=team2, phase=phase_val)
    if wagon_data2:
        components.html(render_wagon_wheel(wagon_data2, f"{team2} - {selected_phase}", 600, 600), height=650)
        st.caption(f"📊 Total scoring shots: {len(wagon_data2)}")
    else:
        st.info(f"No data available for {team2}")

# 7. Player Statistics Cards
st.markdown("---")
st.markdown("## 📊 Player Statistics - Top Performers")
st.markdown("Beautiful gradient cards displaying comprehensive batting statistics")

st.markdown("")
st.markdown(f"### {team1} - Top Batters")
stats1 = get_player_statistics(df, team1, phase=phase_val)
if not stats1.empty:
    st.write(f"Found {len(stats1)} players")  # Debug info
    components.html(render_player_stats_cards(stats1, f"{team1}"), height=800, scrolling=True)
    st.caption(f"📈 Showing top {len(stats1)} batters with minimum 30 balls faced")
else:
    st.info(f"No player statistics available for {team1}")

st.markdown(f"### {team2} - Top Batters")
stats2 = get_player_statistics(df, team2, phase=phase_val)
if not stats2.empty:
    st.write(f"Found {len(stats2)} players")  # Debug info
    components.html(render_player_stats_cards(stats2, f"{team2}"), height=800, scrolling=True)
    st.caption(f"📈 Showing top {len(stats2)} batters with minimum 30 balls faced")
else:
    st.info(f"No player statistics available for {team2}")

# 8. Bowling Length Analysis - 3D visualization
st.markdown("---")
st.markdown("## 🎯 Bowling Length Analysis - 3D Pitch Zones")
st.markdown("""Professional bowling analysis with **color-coded zones** and live percentage statistics.
**Use view controls** on the left to change camera angles!""")

# Color legend bar for Bowling Length
st.markdown("""
<div style="display: flex; gap: 15px; padding: 15px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
            border-radius: 10px; margin: 15px 0; justify-content: center; flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 40px; height: 30px; background: #2ecc71; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Full (16-22 yards) - Yorker</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 40px; height: 30px; background: #f39c12; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Good Length (10-16 yards)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 40px; height: 30px; background: #e74c3c; border-radius: 6px; border: 2px solid white;"></div>
        <span style="color: white; font-weight: 600; font-size: 14px;">Short (4-10 yards) - Bouncer</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Understanding Bowling Zones", expanded=False):
    st.markdown("""
    **Length Zones (Distance from batter):**
    - 🟢 **Full (16+ yards)**: Yorker and full toss length
    - 🟡 **Length (10-16 yards)**: Classic good length bowling
    - 🔴 **Short (4-10 yards)**: Bouncer territory, pull shot length
    
    **Interactive Controls:**
    - 📐 Top View: Bird's eye perspective
    - 🎯 Bowler End: View from bowler's position
    - 🏏 Batter End: View from batter's position  
    - 👁️ Side View: Side angle perspective
    - 🔄 Reset: Return to default view
    """)

st.markdown("")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{team1} Bowling Length**")
    bowling_html_1 = render_bowling_length_map(df, team1, phase=phase_val, unique_id="team1_bowling")
    components.html(bowling_html_1, height=800, scrolling=True)

with col2:
    st.markdown(f"**{team2} Bowling Length**")
    bowling_html_2 = render_bowling_length_map(df, team2, phase=phase_val, unique_id="team2_bowling")
    components.html(bowling_html_2, height=800, scrolling=True)

# 9. Altair Statistical Visualizations
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 25px; border-radius: 15px; margin: 20px 0; 
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);">
    <h2 style="color: white; margin: 0; font-size: 28px; font-weight: 700;">
        📈 Statistical Analysis - Interactive Altair Charts
    </h2>
    <p style="color: rgba(255,255,255,0.95); margin: 10px 0 0 0; font-size: 16px;">
        Professional statistical visualizations with <strong>interactive filtering, tooltips, and advanced analytics</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Summary Statistics Cards
st.markdown("")
col1, col2, col3, col4 = st.columns(4)

team1_data = df[df['batting_team'] == team1]
team2_data = df[df['batting_team'] == team2]
if phase_val:
    team1_data = team1_data[team1_data['phase'] == phase_val]
    team2_data = team2_data[team2_data['phase'] == phase_val]

with col1:
    total_runs_1 = int(team1_data['runs_off_bat'].sum())
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 12px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 16px;">🏏 {team1} Runs</h3>
        <h2 style="color: white; margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{total_runs_1}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_runs_2 = int(team2_data['runs_off_bat'].sum())
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 12px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 16px;">🏏 {team2} Runs</h3>
        <h2 style="color: white; margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{total_runs_2}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    boundaries_1 = len(team1_data[(team1_data['runs_off_bat'] == 4) | (team1_data['runs_off_bat'] == 6)])
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 20px; border-radius: 12px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 16px;">🎯 {team1} Boundaries</h3>
        <h2 style="color: white; margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{boundaries_1}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    boundaries_2 = len(team2_data[(team2_data['runs_off_bat'] == 4) | (team2_data['runs_off_bat'] == 6)])
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                padding: 20px; border-radius: 12px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 16px;">🎯 {team2} Boundaries</h3>
        <h2 style="color: white; margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{boundaries_2}</h2>
    </div>
    """, unsafe_allow_html=True)

# Runs Distribution
st.markdown("")
st.markdown("""
<div style="background: linear-gradient(to right, #667eea, #764ba2); 
            padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; font-size: 22px; font-weight: 600;">📊 Runs Distribution Analysis</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">Ball-by-ball scoring patterns and run accumulation breakdown</p>
</div>
""", unsafe_allow_html=True)

# Team 1 Analysis
st.markdown(f"""<div style='background: rgba(102, 126, 234, 0.1); padding: 10px; 
            border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 10px;'>
            <strong style='color: #667eea; font-size: 16px;'>{team1} - Runs Distribution Analysis</strong>
            </div>""", unsafe_allow_html=True)

# Calculate stats for Team 1
team1_filtered = df[df['batting_team'] == team1].copy()
if phase_val:
    team1_filtered = team1_filtered[team1_filtered['phase'] == phase_val]

t1_total_balls = len(team1_filtered)
t1_total_runs = int(team1_filtered['runs_off_bat'].sum())
t1_dots = len(team1_filtered[team1_filtered['runs_off_bat'] == 0])
t1_singles = len(team1_filtered[team1_filtered['runs_off_bat'] == 1])
t1_twos = len(team1_filtered[team1_filtered['runs_off_bat'] == 2])
t1_threes = len(team1_filtered[team1_filtered['runs_off_bat'] == 3])
t1_fours = len(team1_filtered[team1_filtered['runs_off_bat'] == 4])
t1_sixes = len(team1_filtered[team1_filtered['runs_off_bat'] == 6])
t1_strike_rate = round((t1_total_runs / t1_total_balls) * 100, 2) if t1_total_balls > 0 else 0
t1_boundary_runs = (t1_fours * 4) + (t1_sixes * 6)
t1_boundary_contribution = round((t1_boundary_runs / t1_total_runs) * 100, 1) if t1_total_runs > 0 else 0

# Stats cards for Team 1
col1a, col1b, col1c, col1d, col1e = st.columns(5)
with col1a:
    st.metric("Strike Rate", f"{t1_strike_rate}", "")
with col1b:
    st.metric("Dot Ball %", f"{round((t1_dots/t1_total_balls)*100, 1)}%", "")
with col1c:
    st.metric("Boundaries", f"{t1_fours + t1_sixes}", f"4s:{t1_fours} 6s:{t1_sixes}")
with col1d:
    st.metric("Rotation", f"{t1_singles + t1_twos + t1_threes}", f"1s:{t1_singles} 2s:{t1_twos}")
with col1e:
    st.metric("Boundary Runs %", f"{t1_boundary_contribution}%", "")

runs_chart_1 = create_runs_distribution_chart(df, team1, phase=phase_val)
st.altair_chart(runs_chart_1, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Team 2 Analysis
st.markdown(f"""<div style='background: rgba(118, 75, 162, 0.1); padding: 10px; 
            border-radius: 8px; border-left: 4px solid #764ba2; margin-bottom: 10px;'>
            <strong style='color: #764ba2; font-size: 16px;'>{team2} - Runs Distribution Analysis</strong>
            </div>""", unsafe_allow_html=True)

# Calculate stats for Team 2
team2_filtered = df[df['batting_team'] == team2].copy()
if phase_val:
    team2_filtered = team2_filtered[team2_filtered['phase'] == phase_val]

t2_total_balls = len(team2_filtered)
t2_total_runs = int(team2_filtered['runs_off_bat'].sum())
t2_dots = len(team2_filtered[team2_filtered['runs_off_bat'] == 0])
t2_singles = len(team2_filtered[team2_filtered['runs_off_bat'] == 1])
t2_twos = len(team2_filtered[team2_filtered['runs_off_bat'] == 2])
t2_threes = len(team2_filtered[team2_filtered['runs_off_bat'] == 3])
t2_fours = len(team2_filtered[team2_filtered['runs_off_bat'] == 4])
t2_sixes = len(team2_filtered[team2_filtered['runs_off_bat'] == 6])
t2_strike_rate = round((t2_total_runs / t2_total_balls) * 100, 2) if t2_total_balls > 0 else 0
t2_boundary_runs = (t2_fours * 4) + (t2_sixes * 6)
t2_boundary_contribution = round((t2_boundary_runs / t2_total_runs) * 100, 1) if t2_total_runs > 0 else 0

# Stats cards for Team 2
col2a, col2b, col2c, col2d, col2e = st.columns(5)
with col2a:
    st.metric("Strike Rate", f"{t2_strike_rate}", "")
with col2b:
    st.metric("Dot Ball %", f"{round((t2_dots/t2_total_balls)*100, 1)}%", "")
with col2c:
    st.metric("Boundaries", f"{t2_fours + t2_sixes}", f"4s:{t2_fours} 6s:{t2_sixes}")
with col2d:
    st.metric("Rotation", f"{t2_singles + t2_twos + t2_threes}", f"1s:{t2_singles} 2s:{t2_twos}")
with col2e:
    st.metric("Boundary Runs %", f"{t2_boundary_contribution}%", "")

runs_chart_2 = create_runs_distribution_chart(df, team2, phase=phase_val)
st.altair_chart(runs_chart_2, use_container_width=True)

# Strike Rate Comparison
st.markdown("")
st.markdown("""
<div style="background: linear-gradient(to right, #f093fb, #f5576c); 
            padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; font-size: 22px; font-weight: 600;">⚡ Strike Rate Comparison - Top Performers</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">Most aggressive batters ranked by scoring rate (minimum 50 balls faced)</p>
</div>
""", unsafe_allow_html=True)
strike_rate_chart = create_strike_rate_comparison(df, phase=phase_val)
st.altair_chart(strike_rate_chart, use_container_width=True)

# Runs Progression
st.markdown("")
st.markdown("""
<div style="background: linear-gradient(to right, #fa709a, #fee140); 
            padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; font-size: 22px; font-weight: 600;">📈 Runs Progression Over Overs</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">Cumulative run accumulation and scoring rate throughout the innings</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""<div style='background: rgba(250, 112, 154, 0.1); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #fa709a; margin-bottom: 10px;'>
                <strong style='color: #fa709a; font-size: 16px;'>{team1} - Over-by-Over Progression</strong>
                </div>""", unsafe_allow_html=True)
    progression_1 = create_runs_over_progression(df, team1, phase=phase_val)
    st.altair_chart(progression_1, use_container_width=True)

with col2:
    st.markdown(f"""<div style='background: rgba(254, 225, 64, 0.1); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #fee140; margin-bottom: 10px;'>
                <strong style='color: #d4a500; font-size: 16px;'>{team2} - Over-by-Over Progression</strong>
                </div>""", unsafe_allow_html=True)
    progression_2 = create_runs_over_progression(df, team2, phase=phase_val)
    st.altair_chart(progression_2, use_container_width=True)

# Wicket Timeline
st.markdown("")
st.markdown("""
<div style="background: linear-gradient(to right, #30cfd0, #330867); 
            padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; font-size: 22px; font-weight: 600;">🎯 Wicket Fall Timeline</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">When and how wickets fell - dismissal patterns throughout the innings</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""<div style='background: rgba(48, 207, 208, 0.1); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #30cfd0; margin-bottom: 10px;'>
                <strong style='color: #30cfd0; font-size: 16px;'>{team1} Bowling - Wickets Timeline</strong>
                </div>""", unsafe_allow_html=True)
    wicket_chart_1 = create_wicket_timeline(df, team1, phase=phase_val)
    if wicket_chart_1:
        st.altair_chart(wicket_chart_1, use_container_width=True)
    else:
        st.markdown("""
        <div style='background: rgba(48, 207, 208, 0.1); padding: 20px; 
                    border-radius: 8px; border: 2px dashed #30cfd0; text-align: center;'>
            <p style='color: #30cfd0; margin: 0; font-size: 16px;'>📊 No wickets data available</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div style='background: rgba(51, 8, 103, 0.1); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #330867; margin-bottom: 10px;'>
                <strong style='color: #330867; font-size: 16px;'>{team2} Bowling - Wickets Timeline</strong>
                </div>""", unsafe_allow_html=True)
    wicket_chart_2 = create_wicket_timeline(df, team2, phase=phase_val)
    if wicket_chart_2:
        st.altair_chart(wicket_chart_2, use_container_width=True)
    else:
        st.markdown("""
        <div style='background: rgba(51, 8, 103, 0.1); padding: 20px; 
                    border-radius: 8px; border: 2px dashed #330867; text-align: center;'>
            <p style='color: #330867; margin: 0; font-size: 16px;'>📊 No wickets data available</p>
        </div>
        """, unsafe_allow_html=True)

# Bowler Economy
st.markdown("")
st.markdown("""
<div style="background: linear-gradient(to right, #a8edea, #fed6e3); 
            padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3 style="color: #333; margin: 0; font-size: 22px; font-weight: 600;">💰 Bowler Economy Rate Analysis</h3>
    <p style="color: #555; margin: 5px 0 0 0; font-size: 14px;">Most economical bowlers - runs conceded per over analysis</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""<div style='background: rgba(168, 237, 234, 0.2); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #a8edea; margin-bottom: 10px;'>
                <strong style='color: #00a8a8; font-size: 16px;'>{team1} - Bowler Economy Rates</strong>
                </div>""", unsafe_allow_html=True)
    economy_chart_1 = create_bowler_economy_chart(df, team1, phase=phase_val)
    st.altair_chart(economy_chart_1, use_container_width=True)

with col2:
    st.markdown(f"""<div style='background: rgba(254, 214, 227, 0.2); padding: 10px; 
                border-radius: 8px; border-left: 4px solid #fed6e3; margin-bottom: 10px;'>
                <strong style='color: #d45087; font-size: 16px;'>{team2} - Bowler Economy Rates</strong>
                </div>""", unsafe_allow_html=True)
    economy_chart_2 = create_bowler_economy_chart(df, team2, phase=phase_val)
    st.altair_chart(economy_chart_2, use_container_width=True)

