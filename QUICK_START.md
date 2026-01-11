# 🚀 Quick Start Guide

## Step-by-Step Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database
Edit `.env` file with your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/financial_onboarding
```

### 3. Train ML Models (One-time, 10-15 minutes)
```bash
python train_models.py
```
This will:
- Train Random Forest risk classifier
- Train LSTM models for top 10 stocks

### 4. Start Backend API
```bash
# Terminal 1
cd app
python main.py
# Or: uvicorn app.main:app --reload
```
✅ Backend running at: http://localhost:8000
✅ API Docs at: http://localhost:8000/docs

### 5. Start Frontend
```bash
# Terminal 2
streamlit run streamlit_app.py
```
✅ Frontend running at: http://localhost:8501

## 🎯 Usage Flow

1. **Open**: http://localhost:8501
2. **Create Client**: Fill in personal information
3. **Assess Risk**: Answer questionnaire → ML predicts risk profile
4. **Build Portfolio**: Enter investment amount → Get ML-powered portfolio
5. **View Dashboard**: See advanced visualizations and analytics

## ✅ Verification Checklist

- [ ] PostgreSQL is running
- [ ] Database `financial_onboarding` exists
- [ ] Tables are created (clients, risk_assessments, portfolios, etc.)
- [ ] `.env` file configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models trained (`python train_models.py`)
- [ ] Backend running (port 8000)
- [ ] Frontend running (port 8501)

## 🐛 Common Issues

**Issue**: Database connection error
- **Fix**: Check PostgreSQL is running and `.env` has correct credentials

**Issue**: Models not found
- **Fix**: Run `python train_models.py` first

**Issue**: API connection error in Streamlit
- **Fix**: Ensure FastAPI backend is running on port 8000

**Issue**: Stock data not loading
- **Fix**: Check internet connection (yfinance requires internet)

## 📞 Need Help?

Check the main `README.md` for detailed documentation.

