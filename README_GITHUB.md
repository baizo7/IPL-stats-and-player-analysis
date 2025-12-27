# IPL Performance Analysis Dashboard

A comprehensive IPL (Indian Premier League) cricket analytics dashboard built with Streamlit, featuring advanced 3D visualizations, interactive charts, and detailed performance metrics.

## 🏏 Features

### Core Analytics
- **Team Comparison**: Head-to-head analysis between any two IPL teams
- **Phase-wise Analysis**: Powerplay, Middle Overs, and Death Overs breakdown
- **Player Statistics**: Detailed batting performance with franchise-colored cards
- **Strike Rate Comparison**: Top batters comparison with team colors

### Advanced Visualizations

#### 3D Interactive Views
- **Pitch Maps**: 3D ball landing positions with outcome color coding
- **Stumps View**: Ball trajectory visualization from stumps perspective
- **Wagon Wheel**: Shot distribution analysis in 3D
- **Bowling Length Analysis**: 3D pitch zones showing bowling length distribution

#### Statistical Charts
- **Runs Distribution Analysis**: 
  - Main bar chart with run type breakdown
  - Donut chart showing run contribution percentages
  - Horizontal bar graph for visual comparison
- **Runs Progression**: Over-by-over cumulative runs with wicket markers
- **Wicket Timeline**: Dismissal patterns throughout innings
- **Bowler Economy**: Comprehensive bowling statistics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ipl-performance-analysis.git
cd ipl-performance-analysis
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to:
   - Select season/date range
   - Choose teams for comparison
   - Filter by match phase
   - Select bowler type

## 📁 Project Structure

```
ipl-performance-analysis/
├── app.py                          # Main Streamlit application
├── data_processor.py               # Data processing utilities
├── requirements.txt                # Python dependencies
├── dataset/                        # Match JSON files
│   ├── 1082591.json
│   ├── 1082592.json
│   └── ...
├── ipl_data/                       # Additional data files
├── media/                          # Media assets
└── README.md                       # This file
```

## 🎨 Key Features Explained

### Franchise Colors
All visualizations use authentic IPL team colors for consistent branding:
- Royal Challengers Bangalore: Red (#EC1C24)
- Chennai Super Kings: Yellow (#FDB913)
- Mumbai Indians: Blue (#004BA0)
- And more...

### Interactive 3D Visualizations
Built with Three.js, featuring:
- Realistic cricket stadium environment
- Multiple camera angles (Top, Bowler End, Batter End, Side View)
- Real-time statistics overlay
- Smooth camera animations

### Data Sources
The dashboard processes ball-by-ball data from Cricsheet JSON format, providing comprehensive match statistics.

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Visualizations**: 
  - Altair (2D charts)
  - Three.js (3D visualizations)
  - Plotly (Interactive plots)
- **Data Processing**: Pandas, NumPy
- **Styling**: Custom HTML/CSS

## 📈 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Heroku

1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👨‍💻 Author

Your Name
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Data sourced from Cricsheet
- IPL team colors and branding from official IPL sources
- Built with Streamlit framework

## 📞 Support

For support, email your-email@example.com or open an issue in the repository.

---

⭐ Star this repo if you find it helpful!
