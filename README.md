TRY THE IPL STATS ANALYSIS BY COMAPARING YOUR FAVOURITE TEAMS , HERE IS THE LINK : https://ipl-stats-and-player-analysis-5.onrender.com/


# 🏏 IPL Team Performance Dashboard

A comprehensive data analysis dashboard for comparing IPL team performance across different match phases and analyzing batter vs bowler matchups.

## 📊 Features

- **Phase Analysis**: Compare run rates, strike rates, and boundaries in Powerplay (1-6), Middle (7-15), and Death (16-20) overs
- **Matchup Analysis**: Interactive heatmaps showing batter performance against different bowler types
- **Team Summaries**: Overall team statistics and top performer rankings
- **Detailed Statistics**: Over-by-over analysis and complete matchup data

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations and charts
- **NumPy**: Numerical computations

## 📥 Installation

1. **Clone or navigate to the project directory:**
   ```powershell
   cd "d:\projects\IPL+perfromance analysis"
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

## 📁 Data Setup

### Option 1: Use Sample Data (Quick Start)
The dashboard comes with sample data generation built-in. Just run the app and check "Use Sample Data" in the sidebar.

### Option 2: Use Real IPL Data from Cricsheet

1. **Visit [Cricsheet.org](https://cricsheet.org/downloads/)**

2. **Download IPL ball-by-ball data:**
   - Navigate to the "Indian Premier League" section
   - Download the CSV format files (recommended)
   - You can download season-specific files or the complete dataset

3. **Create a data folder and extract files:**
   ```powershell
   New-Item -ItemType Directory -Path "data" -Force
   # Extract downloaded CSV files to the data folder
   ```

4. **Expected CSV format:**
   The Cricsheet CSV files should have columns like:
   - `match_id`
   - `season`
   - `batting_team`
   - `bowling_team`
   - `ball` or `over`
   - `batter` or `striker`
   - `bowler`
   - `runs_off_bat` or `batsman_runs`
   - `extras`
   - `wicket_type`
   - `player_dismissed`

## 🚀 Running the Dashboard

### Option 1: Modern Vue.js + GSAP Dashboard (Recommended)

1. **Install Flask dependencies:**
   ```powershell
   pip install -r requirements_flask.txt
   ```

2. **Start the Flask server:**
   ```powershell
   python app_flask.py
   ```

3. **Access the dashboard:**
   - Open your browser and navigate to `http://localhost:5000`
   - Features advanced GSAP animations and Vue.js powered statistics
   - No "About" section - pure analytics focus

### Option 2: Classic Streamlit Dashboard

1. **Install Streamlit dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

3. **Access the dashboard:**
   - Navigate to `http://localhost:8501`

### Using the dashboard:
- Select two teams from the selector to compare
- Navigate between Phase Analysis, Matchup, Summary, and Detailed Stats
- Enjoy smooth GSAP animations and interactive charts
- All statistics powered by Vue.js for reactive updates

## 📂 Project Structure

```
IPL+perfromance analysis/
├── app.py                 # Main Streamlit dashboard application
├── data_processor.py      # Data loading and cleaning module
├── analysis.py            # Analysis functions (phase, matchup, metrics)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── data/                 # Place your IPL CSV files here (not tracked)
```

## 📊 Dashboard Sections

### 1. Phase Analysis
- Compare run rates across Powerplay, Middle, and Death overs
- Visualize boundaries and wickets by phase
- Side-by-side team comparison with interactive charts

### 2. Matchup Analysis
- Heatmaps showing batter performance vs different bowler types
- Strike rate and dismissal rate metrics
- Top 10 performing matchups for each team

### 3. Team Summary
- Overall team statistics (runs, matches, run rate)
- Top 5 batters with runs, strike rate, and average
- Quick comparison metrics

### 4. Detailed Stats
- Runs distribution per ball
- Over-by-over scoring analysis
- Complete matchup data table

## 🎯 Example Use Cases

### Compare CSK vs MI
1. Select "Chennai Super Kings" as Team 1
2. Select "Mumbai Indians" as Team 2
3. Navigate to Phase Analysis to see how each team performs in different overs
4. Check Matchup Analysis to see how Rohit Sharma performs against left-arm spin

### Analyze Team Strategy
- Use Phase Analysis to identify if a team is conservative in powerplay or aggressive in death overs
- Use Matchup Analysis to find player weaknesses against specific bowling types

## 🔧 Troubleshooting

### No data found error
- Ensure CSV files are in the `data/` folder
- Check that CSV files have the expected column names
- Try using sample data first to test the dashboard

### Module import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one

### Dashboard not loading
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 📈 Future Enhancements

- Add venue-based analysis
- Include player-specific detailed profiles
- Add season-wise performance trends
- Implement win/loss correlation analysis
- Export reports as PDF

## 📝 Notes

- The sample data includes 50 matches with realistic statistics
- Real Cricsheet data provides comprehensive IPL history
- All calculations use ball-by-ball granularity for accuracy

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Additional analysis features
- UI improvements
- Bug fixes
- Performance optimizations

## 📄 License

This project is for educational and analytical purposes. IPL data is provided by Cricsheet.org under their terms of use.

## 🙏 Acknowledgments

- **Cricsheet.org**: For providing comprehensive cricket data
- **Streamlit**: For the excellent dashboard framework
- **Plotly**: For interactive visualization capabilities

---

**Built with ❤️ for cricket analytics enthusiasts**
