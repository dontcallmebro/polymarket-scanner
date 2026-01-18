# 🎯 Polymarket Volatility Scanner

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A real-time trading opportunity scanner for Polymarket prediction markets. This dashboard identifies high-potential trades using a proprietary scoring model based on volatility, liquidity, and momentum analysis.

![Dashboard Preview](docs/preview.png)

## ✨ Features

- **📊 Live Market Data** - Real-time polling from Polymarket's Gamma API
- **🧠 Smart Scoring Model** - Multi-factor scoring (Volatility 35%, Liquidity 25%, Opportunity 25%, Momentum 15%)
- **🎛️ Granular Controls** - Customize minimum age, maturity, volume, risk/reward ratio
- **🔍 Price Filters** - Filter opportunities by probability range (default: 3% - 95%)
- **📈 Target Performance** - Shows expected yield from entry to target price
- **🖥️ Live Console** - Mini-terminal showing real-time processing logs
- **🎨 Polymarket Theme** - Clean blue interface matching Polymarket branding

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/polymarket-scanner.git
   cd polymarket-scanner
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   python main.py
   ```

5. **Open your browser** at `http://127.0.0.1:8050`

### Quick Launch (Windows)

Double-click `run_dashboard.bat` to start the dashboard.

## 📖 How It Works

### Model Pipeline

The scanner follows this pipeline to identify opportunities:

1. **Data Fetch** - Retrieves active markets from Polymarket API
2. **Initial Filter** - Applies age, maturity, and volume criteria
3. **Price Filter** - Keeps only markets within your probability range
4. **Scoring** - Calculates composite score for each market
5. **Ranking** - Returns top N opportunities by score

### Scoring Breakdown

| Factor | Weight | Description |
|--------|--------|-------------|
| **Volatility** | 35% | Recent price movement and ATR |
| **Liquidity** | 25% | Volume and market depth |
| **Opportunity** | 25% | Distance from fair value |
| **Momentum** | 15% | Trend strength and direction |

### Signal Types

- **🟢 LONG** - Buy YES tokens (expect price to rise)
- **🔴 SHORT** - Buy NO tokens (expect price to fall)

## ⚙️ Configuration

### Model Parameters (UI Controls)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Age (days) | 3 | Minimum market age |
| Min Maturity (days) | 7 | Days until market resolution |
| Min Volume ($) | 10,000 | Minimum daily volume |
| Min R/R Ratio | 1.5 | Minimum risk/reward ratio |
| Results | 10 | Number of opportunities to display |

### Price Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Min Price | 0.03 | Minimum probability (3%) |
| Max Price | 0.95 | Maximum probability (95%) |

### Environment Variables (Optional)

Create a `config/.env` file for advanced features:

```env
# OpenAI API for AI analysis (optional)
OPENAI_API_KEY=your_api_key_here
```

## 📁 Project Structure

```
polymarket_project/
├── src/
│   ├── dashboard/
│   │   ├── app.py          # Main Dash application
│   │   └── theme.py        # UI theme configuration
│   ├── api_client.py       # Polymarket API integration
│   ├── trade_recommender.py # Trade recommendation engine
│   └── trade_scorer.py     # Market scoring logic
├── config/
│   └── settings.yaml       # Application settings
├── data/                   # Local database (excluded from git)
├── output/                 # Generated reports
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔧 Development

### Running Tests

```bash
python test_api.py
```

### API Rate Limits

The Polymarket Gamma API has rate limits. The dashboard polls every 30 seconds by default to respect these limits.

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- This is NOT financial advice
- Prediction markets involve significant risk
- Past performance does not guarantee future results
- Always do your own research before trading
- Only trade with money you can afford to lose

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Polymarket](https://polymarket.com) for the prediction market platform
- [Dash by Plotly](https://dash.plotly.com) for the web framework
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) for UI components

---

**Made with ❤️ for the prediction market community**
