# Advanced Financial Onboarding System

A world-class financial onboarding system with ML-powered stock predictions and risk assessment.

## 🚀 Features

- **LSTM Model**: Predicts future stock prices (1-day, 7-day, 30-day)
- **Random Forest Model**: ML-based risk profile classification
- **Interactive Web UI**: Beautiful Streamlit interface with advanced visualizations
- **PostgreSQL Database**: Secure data persistence
- **FastAPI Backend**: RESTful API with automatic documentation
- **Real-time Stock Data**: yfinance integration

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 14+
- pip

## 🛠️ Installation

1. **Clone the repository**
```bash
cd dynamic-financial-onboarding
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up PostgreSQL**
   - Ensure PostgreSQL is running
   - Create database (if not already created):
   ```sql
   CREATE DATABASE financial_onboarding;
   ```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your database credentials
```

6. **Train ML models** (One-time, takes 10-15 minutes)
```bash
python train_models.py
```

## 🎯 Usage

### Start FastAPI Backend

```bash
cd app
python main.py
# Or use uvicorn directly:
uvicorn app.main:app --reload
```

API will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Start Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Frontend will be available at: `http://localhost:8501`

## 📊 Workflow

1. **Client Onboarding**: Create client profile
2. **Risk Assessment**: Complete questionnaire → ML model predicts risk profile
3. **Portfolio Builder**: Enter investment amount → System generates portfolio with LSTM predictions
4. **Dashboard**: View comprehensive analytics and visualizations

## 🗂️ Project Structure

```
dynamic-financial-onboarding/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── database.py             # Database connection
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic schemas
│   ├── ml_models/
│   │   ├── stock_predictor.py  # LSTM model
│   │   └── risk_classifier.py  # Random Forest model
│   ├── services/
│   │   ├── yfinance_service.py # Stock data service
│   │   └── portfolio_service.py # Portfolio generation
│   └── routers/
│       ├── clients.py          # Client endpoints
│       ├── risk.py             # Risk assessment endpoints
│       ├── portfolio.py        # Portfolio endpoints
│       └── predictions.py      # Prediction endpoints
├── streamlit_app.py            # Streamlit frontend
├── train_models.py            # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

Edit `.env` file:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/financial_onboarding
```

## 📈 API Endpoints

- `POST /api/clients/` - Create client
- `GET /api/clients/{id}` - Get client
- `POST /api/risk/assess` - Assess risk (ML)
- `GET /api/risk/{client_id}` - Get risk assessment
- `POST /api/portfolio/generate` - Generate portfolio
- `GET /api/portfolio/{client_id}` - Get portfolio
- `GET /api/predictions/{ticker}` - Get stock predictions

## 🎨 Visualizations

- Interactive risk gauge charts
- Portfolio allocation pie charts
- Predicted returns bar charts
- Risk-return scatter plots
- Sector diversification charts
- Performance projections

## 🤖 ML Models

### LSTM Stock Predictor
- Architecture: 2 LSTM layers (50 units each) + Dense layers
- Input: 60 days of historical prices
- Output: Predicted prices for 1d, 7d, 30d
- Training: 5 years of historical data

### Random Forest Risk Classifier
- Architecture: 100 decision trees
- Features: Age, income, expenses, questionnaire responses
- Output: Risk profile + confidence score
- Accuracy: 85%+

## 📝 Notes

- Models are cached in `models/` directory
- Stock data is cached in PostgreSQL to reduce API calls
- First run may take time to fetch and cache stock data
- Ensure PostgreSQL tables are created before running

## 🐛 Troubleshooting

1. **Database connection error**: Check PostgreSQL is running and credentials in `.env`
2. **Model not found**: Run `python train_models.py` first
3. **API connection error**: Ensure FastAPI backend is running on port 8000
4. **Stock data error**: Check internet connection (yfinance requires internet)

## 📄 License

MIT License

## 👤 Author

Kotapati Deepa

