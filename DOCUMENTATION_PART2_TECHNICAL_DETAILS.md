# Part 2: Technical Details - Models, Functionalities & Architecture

## 📋 Table of Contents
1. [ML Models Used](#ml-models-used)
2. [System Functionalities](#system-functionalities)
3. [User Flow (Step-by-Step)](#user-flow-step-by-step)
4. [Developer Flow (Technical Architecture)](#developer-flow-technical-architecture)
5. [Data Flow & Processing](#data-flow--processing)
6. [How Results are Fetched](#how-results-are-fetched)
7. [How Models are Trained](#how-models-are-trained)

---

## 🤖 ML Models Used

### Model 1: Random Forest Classifier (Risk Assessment)

#### **What It Does:**
Classifies clients into 5 risk profiles based on their personal and financial information.

#### **Technical Details:**

**Algorithm:** Random Forest (Ensemble Learning)
- **Type:** Supervised Classification
- **Library:** scikit-learn
- **Architecture:** 100 Decision Trees
- **Max Depth:** 10 levels
- **Min Samples Split:** 5
- **Min Samples Leaf:** 2

**Input Features (10 features):**
1. Age (18-100)
2. Monthly Income (₹)
3. Monthly Expenses (₹)
4. Number of Dependents (0-10)
5. Emergency Fund (₹)
6. Investment Horizon (years)
7. Volatility Tolerance (1-4 scale)
8. Financial Knowledge (1-4 scale)
9. Income Stability (1-4 scale)
10. Risk Capacity (1-4 scale)

**Output:**
- Risk Profile: One of 5 categories
  - Conservative
  - Moderate Conservative
  - Balanced
  - Growth Oriented
  - Aggressive Growth
- Confidence Score: 0-100%
- Risk Score: 0-75 points
- Probabilities: For each risk category

**Training Data:**
- **Size:** 2,000 synthetic client profiles
- **Split:** 80% training (1,600) + 20% testing (400)
- **Generation:** Realistic synthetic data based on financial advisory rules
- **Validation:** 5-fold cross-validation

**Performance:**
- **Training Accuracy:** 97.75%
- **Test Accuracy:** 84.00%
- **Cross-Validation:** 84.19% (±4.27%)
- **Model Size:** 3.7 MB

**How It Works:**
```python
# Simplified example
features = [age, income, expenses, dependents, emergency_fund, 
            investment_horizon, volatility_tolerance, 
            financial_knowledge, income_stability, risk_capacity]

# Model processes
risk_profile = random_forest.predict(features)
confidence = max(model.predict_proba(features)) * 100

# Returns
{
    "risk_profile": "Balanced",
    "confidence_score": 84.5,
    "risk_score": 52.3,
    "probabilities": {
        "Conservative": 5.2%,
        "Moderate Conservative": 12.8%,
        "Balanced": 84.5%,  # Highest
        "Growth Oriented": 15.3%,
        "Aggressive Growth": 2.1%
    }
}
```

**Real Example:**
```
Input:
- Age: 30
- Income: ₹50,000/month
- Expenses: ₹30,000/month
- Dependents: 0
- Emergency Fund: ₹2,00,000
- Investment Horizon: 10 years
- Volatility Tolerance: 3 (Medium)
- Financial Knowledge: 2 (Basic)
- Income Stability: 4 (Very Stable)
- Risk Capacity: 3 (Medium)

Model Processing:
- 100 decision trees vote
- Each tree analyzes different feature combinations
- Majority vote determines risk profile

Output:
- Risk Profile: "Balanced" (84% confidence)
- Risk Score: 52.3/75
- Explanation: Stable income + medium risk tolerance = Balanced
```

---

### Model 2: LSTM Neural Network (Stock Price Prediction)

#### **What It Does:**
Predicts future stock prices (1-day, 7-day, 30-day ahead) using historical price patterns.

#### **Technical Details:**

**Algorithm:** Long Short-Term Memory (LSTM) Neural Network
- **Type:** Deep Learning / Time Series Forecasting
- **Library:** TensorFlow/Keras
- **Architecture:** 
  - Input Layer: 60 days of historical prices
  - LSTM Layer 1: 50 neurons (with return_sequences=True)
  - Dropout: 20% (prevents overfitting)
  - LSTM Layer 2: 50 neurons
  - Dropout: 20%
  - Dense Layer: 25 neurons
  - Output Layer: 1 neuron (predicted price)

**Input:**
- **Sequence Length:** 60 days of historical closing prices
- **Data Format:** Normalized to 0-1 range
- **Example:** [Day 1 price, Day 2 price, ..., Day 60 price] → Predict Day 61

**Output:**
- Predicted Price (1 day ahead)
- Predicted Price (7 days ahead)
- Predicted Price (30 days ahead)
- Confidence Scores for each prediction

**Training Data:**
- **Size:** ~1,304 days per stock (~3.5 years)
- **Sequences Created:** ~995 training sequences per stock
- **Split:** 80% training (~796) + 20% validation (~199)
- **Data Source:** Realistic synthetic data (Geometric Brownian Motion)

**Performance:**
- **Average Loss:** 0.002316 (excellent - lower is better)
- **Interpretation:** Predictions are ~0.23% off on average
- **Model Size:** ~426 KB per stock model

**How It Works:**
```python
# Step 1: Prepare Data
historical_prices = [2500, 2510, 2495, 2520, ..., 2600]  # 60 days
normalized = [0.75, 0.752, 0.748, 0.756, ..., 0.78]  # Scaled to 0-1

# Step 2: Create Sequences
X = [
    [Day 1-60 prices],  # Input sequence
    [Day 2-61 prices],
    ...
]
y = [Day 61 price, Day 62 price, ...]  # Target predictions

# Step 3: Train LSTM
model.fit(X_train, y_train, epochs=30)

# Step 4: Predict
recent_60_days = [last 60 days of prices]
prediction = model.predict(recent_60_days)
predicted_price = inverse_normalize(prediction)

# Returns
{
    "current_price": 2600.00,
    "predicted_price_1d": 2615.50,
    "predicted_price_7d": 2680.25,
    "predicted_price_30d": 2785.00,
    "confidence_1d": 85.0,
    "confidence_7d": 75.0,
    "confidence_30d": 65.0
}
```

**Real Example:**
```
Stock: RELIANCE.NS
Current Price: ₹2,500

LSTM Processing:
- Takes last 60 days: [2450, 2460, 2475, ..., 2500]
- Normalizes to 0-1 range
- Passes through LSTM layers
- Learns patterns: "When prices go up for 10 days, 
                    then down for 5 days,
                    next day usually goes up by 1.2%"

Prediction:
- 1-day: ₹2,530 (1.2% increase)
- 7-day: ₹2,625 (5% increase)
- 30-day: ₹2,750 (10% increase)

Confidence decreases with time (1d: 85%, 30d: 65%)
```

**Why LSTM?**
- **Captures Long-Term Patterns:** Remembers trends from 60 days ago
- **Handles Sequences:** Understands price sequences, not just individual prices
- **Non-Linear Relationships:** Learns complex patterns humans might miss
- **Time Series Specialized:** Designed specifically for sequential data

---

## 🎯 System Functionalities

### Functionality 1: Client Onboarding

**What It Does:**
Collects comprehensive client information through interactive web forms.

**Features:**
- Real-time form validation
- Data persistence to PostgreSQL
- Profile save/load capability
- Multiple client management

**Technical Implementation:**
```python
# Frontend (Streamlit)
client_name = st.text_input("Full Name")
age = st.number_input("Age", min_value=18, max_value=100)
monthly_income = st.number_input("Monthly Income (₹)")

# Backend (FastAPI)
@app.post("/api/clients/")
def create_client(client: ClientCreate):
    db_client = Client(**client.dict())
    db.add(db_client)
    db.commit()
    return db_client
```

**Data Stored:**
- Personal information (name, age, dependents)
- Financial information (income, expenses, emergency fund)
- Employment status
- Timestamps (created_at, updated_at)

---

### Functionality 2: ML-Powered Risk Assessment

**What It Does:**
Uses Random Forest model to classify client risk profile.

**Features:**
- 5-question questionnaire
- Real-time ML prediction
- Confidence scores
- Probability distribution

**Technical Implementation:**
```python
# User answers questionnaire
questionnaire = {
    "investment_horizon": 10,  # years
    "volatility_tolerance": 3,  # 1-4 scale
    "financial_knowledge": 2,
    "income_stability": 4,
    "risk_capacity": 3
}

# ML Model Prediction
classifier = RiskClassifier()
prediction = classifier.predict(client_data, questionnaire)

# Returns
{
    "risk_profile": "Balanced",
    "confidence_score": 84.5,
    "risk_score": 52.3
}
```

**Process:**
1. User answers 5 questions
2. System combines with client profile data
3. Random Forest model processes (100 trees vote)
4. Returns risk profile + confidence
5. Saves to database

---

### Functionality 3: Real-Time Market Analysis

**What It Does:**
Fetches and analyzes NIFTY 50 stocks with financial metrics.

**Features:**
- Real-time stock data fetching (yfinance)
- Volatility calculation
- Risk classification
- Sector analysis
- Data caching in PostgreSQL

**Technical Implementation:**
```python
# Fetch stock data
stock = yf.Ticker("RELIANCE.NS")
df = stock.history(period="3mo")

# Calculate metrics
volatility = df['Close'].pct_change().std() * (252 ** 0.5) * 100
current_price = df['Close'].iloc[-1]
market_cap = stock.info['marketCap']

# Classify risk
risk_classification = classify_stock_risk(volatility, de_ratio, revenue_growth)

# Cache in database
market_data = MarketData(
    stock_ticker="RELIANCE.NS",
    date=today,
    close_price=current_price,
    volatility=volatility
)
db.add(market_data)
```

**Metrics Calculated:**
- Current price
- Market capitalization
- Volatility (annualized)
- Debt-to-equity ratio
- Revenue growth
- Risk classification (Low/Medium/High)

---

### Functionality 4: Portfolio Construction with LSTM Predictions

**What It Does:**
Builds personalized portfolio using ML predictions.

**Features:**
- Risk-based stock filtering
- LSTM price predictions
- Optimal allocation algorithm
- Sector diversification

**Technical Implementation:**
```python
# Step 1: Filter stocks by risk profile
if risk_profile == "Conservative":
    stocks = get_low_risk_stocks()
elif risk_profile == "Balanced":
    stocks = get_low_and_medium_risk_stocks()
else:
    stocks = get_medium_and_high_risk_stocks()

# Step 2: Get LSTM predictions for each stock
predictor = StockPredictor()
for stock in stocks:
    predictions = predictor.predict(stock.ticker)
    stock.predicted_return_7d = calculate_return(predictions)
    stock.predicted_return_30d = calculate_return(predictions)

# Step 3: Select top 5-8 stocks
selected_stocks = rank_by_predicted_returns(stocks)[:8]

# Step 4: Calculate allocations
allocations = calculate_allocations(selected_stocks, risk_profile, total_investment)

# Step 5: Create portfolio
portfolio = create_portfolio(client_id, selected_stocks, allocations)
```

**Allocation Algorithm:**
```python
# Conservative: Equal weights
allocation = 100% / num_stocks  # Each stock gets equal share

# Balanced: Weighted (first stocks get more)
weights = [20%, 18%, 16%, 14%, 12%, 10%, 8%, 6%]

# Aggressive: Concentrated (top stocks get much more)
weights = [25%, 20%, 15%, 12%, 10%, 8%, 6%, 4%]
```

---

### Functionality 5: Advanced Visualizations

**What It Does:**
Displays interactive charts and graphs using Plotly.

**Visualizations:**
1. **Risk Profile Gauge Chart**
   - Shows risk score on circular gauge
   - Color-coded (green/yellow/orange/red)
   - Real-time updates

2. **Portfolio Allocation Pie Chart**
   - Shows percentage allocation per stock
   - Interactive hover tooltips
   - Sector color coding

3. **Predicted Returns Bar Chart**
   - 7-day and 30-day predictions
   - Side-by-side comparison
   - Confidence intervals

4. **Stock Price Prediction Timeline**
   - Historical prices (line chart)
   - Predicted prices (dashed line)
   - Confidence bands

5. **Risk-Return Scatter Plot**
   - X-axis: Predicted return
   - Y-axis: Allocation percentage
   - Bubble size: Investment amount
   - Color: Risk level

**Technical Implementation:**
```python
# Example: Risk Gauge Chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [None, 75]},
        'steps': [
            {'range': [0, 22.5], 'color': "lightgreen"},
            {'range': [22.5, 37.5], 'color': "yellow"},
            {'range': [37.5, 52.5], 'color': "orange"},
            {'range': [52.5, 75], 'color': "red"}
        ]
    }
))
st.plotly_chart(fig)
```

---

## 👤 User Flow (Step-by-Step)

### Complete User Journey

#### **Step 1: Client Onboarding**

**User Actions:**
1. Opens web application
2. Sees welcome screen with "Welcome to the Advanced Financial Onboarding System"
3. Fills form:
   - Enters name: "Rahul Sharma"
   - Age: 30
   - Monthly Income: ₹50,000
   - Monthly Expenses: ₹30,000
   - Dependents: 0
   - Emergency Fund: ₹2,00,000
   - Employment: "Employed"
4. Clicks "Create Client Profile"

**What Happens Behind the Scenes:**
```
User Input → Streamlit Form → Validation → API Call → FastAPI Endpoint
→ Database Insert → PostgreSQL → Returns Client ID → Display Success
```

**Technical Flow:**
```python
# Frontend
client_data = {
    "client_name": "Rahul Sharma",
    "age": 30,
    "monthly_income": 50000,
    ...
}

# API Call
response = requests.post("http://localhost:8000/api/clients/", json=client_data)

# Backend Processing
@app.post("/api/clients/")
def create_client(client: ClientCreate):
    db_client = Client(**client.dict())
    db.add(db_client)
    db.commit()
    return {"id": db_client.id, "client_name": db_client.client_name}

# Database
INSERT INTO clients (client_name, age, monthly_income, ...) 
VALUES ('Rahul Sharma', 30, 50000, ...);

# Response
{"id": 1, "client_name": "Rahul Sharma", ...}
```

**Result:**
- Client profile created
- Client ID: 1
- Data saved to PostgreSQL
- User proceeds to risk assessment

---

#### **Step 2: Risk Assessment**

**User Actions:**
1. Sees risk assessment page
2. Answers 5 questions:
   - Investment Horizon: 10 years (slider)
   - Volatility Tolerance: Medium (select slider)
   - Financial Knowledge: Basic (select slider)
   - Income Stability: Very Stable (select slider)
   - Risk Capacity: Medium (select slider)
3. Clicks "Assess Risk (ML Model)"

**What Happens Behind the Scenes:**
```
User Answers → API Call → Risk Classifier Model → Random Forest Processing
→ Risk Profile Prediction → Database Save → Display Results with Gauge Chart
```

**Technical Flow:**
```python
# Frontend collects answers
questionnaire = {
    "client_id": 1,
    "investment_horizon": 10,
    "volatility_tolerance": 3,
    "financial_knowledge": 2,
    "income_stability": 4,
    "risk_capacity": 3
}

# API Call
response = requests.post("http://localhost:8000/api/risk/assess", json=questionnaire)

# Backend Processing
@app.post("/api/risk/assess")
def assess_risk(assessment: RiskAssessmentCreate):
    # Get client data
    client = db.query(Client).filter(Client.id == assessment.client_id).first()
    
    # Prepare features
    features = [
        client.age,                    # 30
        client.monthly_income,          # 50000
        client.monthly_expenses,        # 30000
        client.dependents,              # 0
        client.emergency_fund,          # 200000
        assessment.investment_horizon,  # 10
        assessment.volatility_tolerance, # 3
        assessment.financial_knowledge,  # 2
        assessment.income_stability,    # 4
        assessment.risk_capacity        # 3
    ]
    
    # ML Model Prediction
    classifier = RiskClassifier()
    classifier.load_model()  # Loads from models/risk_classifier.pkl
    
    prediction = classifier.model.predict([features])[0]
    probabilities = classifier.model.predict_proba([features])[0]
    
    risk_profile = classifier.label_encoder.inverse_transform([prediction])[0]
    confidence = max(probabilities) * 100
    
    # Save to database
    risk_assessment = RiskAssessment(
        client_id=assessment.client_id,
        risk_profile=risk_profile,
        risk_score=calculate_score(assessment),
        confidence_score=confidence
    )
    db.add(risk_assessment)
    db.commit()
    
    return risk_assessment

# Model Processing (Simplified)
# 100 decision trees analyze features
# Tree 1: "Age 30 + Stable income = Balanced"
# Tree 2: "Medium risk tolerance = Balanced"
# ...
# Tree 100: "10-year horizon = Growth Oriented"
# Majority vote: "Balanced" (84 trees vote Balanced)

# Returns
{
    "risk_profile": "Balanced",
    "confidence_score": 84.5,
    "risk_score": 52.3
}
```

**Result:**
- Risk Profile: "Balanced"
- Confidence: 84.5%
- Risk Score: 52.3/75
- Gauge chart displayed
- User proceeds to portfolio builder

---

#### **Step 3: Portfolio Builder**

**User Actions:**
1. Sees portfolio builder page
2. Enters investment amount: ₹100,000 (slider)
3. Clicks "Generate Portfolio (with LSTM Predictions)"

**What Happens Behind the Scenes:**
```
Investment Amount → API Call → Fetch Stocks → LSTM Predictions → 
Stock Selection → Allocation Calculation → Portfolio Creation → 
Database Save → Display Portfolio with Charts
```

**Technical Flow:**
```python
# Frontend
portfolio_data = {
    "client_id": 1,
    "total_investment": 100000
}

# API Call
response = requests.post("http://localhost:8000/api/portfolio/generate", json=portfolio_data)

# Backend Processing
@app.post("/api/portfolio/generate")
def generate_portfolio(portfolio: PortfolioCreate):
    # Get client and risk profile
    client = db.query(Client).filter(Client.id == portfolio.client_id).first()
    risk_assessment = db.query(RiskAssessment).filter(
        RiskAssessment.client_id == portfolio.client_id
    ).order_by(RiskAssessment.assessment_date.desc()).first()
    
    risk_profile = risk_assessment.risk_profile  # "Balanced"
    
    # Step 1: Get available stocks
    stocks_data = YFinanceService.get_nifty50_stocks_with_data(db)
    # Returns: [{"ticker": "RELIANCE.NS", "risk_classification": "Low Risk", ...}, ...]
    
    # Step 2: Filter by risk profile
    if risk_profile == "Balanced":
        filtered_stocks = [s for s in stocks_data 
                          if s['risk_classification'] in ['Low Risk', 'Medium Risk']]
    # Result: 15 stocks match
    
    # Step 3: Get LSTM predictions for each stock
    predictor = StockPredictor()
    predictions = {}
    
    for stock in filtered_stocks[:10]:  # Top 10
        ticker = stock['ticker']
        
        # Load LSTM model
        predictor.load_model(ticker)  # Loads models/lstm_RELIANCE.h5
        
        # Get recent 60 days of prices
        recent_data = fetch_stock_data(ticker, period="3mo")
        last_60_days = recent_data['Close'].tail(60).values
        
        # Normalize
        scaled = predictor.scaler.transform(last_60_days.reshape(-1, 1))
        
        # Predict
        X_input = scaled.reshape(1, 60, 1)
        pred_1d = predictor.model.predict(X_input)[0, 0]
        pred_1d_price = predictor.scaler.inverse_transform([[pred_1d]])[0, 0]
        
        # Predict 7 days ahead (iterative)
        temp_input = X_input.copy()
        for _ in range(7):
            pred = predictor.model.predict(temp_input)
            new_input = np.append(temp_input[0, 1:, 0], pred[0, 0])
            temp_input = new_input.reshape(1, 60, 1)
        pred_7d_price = predictor.scaler.inverse_transform([[pred[0, 0]]])[0, 0]
        
        # Calculate returns
        current_price = recent_data['Close'].iloc[-1]
        return_7d = ((pred_7d_price - current_price) / current_price) * 100
        
        predictions[ticker] = {
            "current_price": current_price,
            "predicted_price_7d": pred_7d_price,
            "predicted_return_7d": return_7d
        }
    
    # Step 4: Select top 5-8 stocks
    sorted_stocks = sorted(filtered_stocks, 
                          key=lambda x: predictions[x['ticker']]['predicted_return_7d'],
                          reverse=True)
    selected_stocks = sorted_stocks[:6]  # Top 6
    
    # Step 5: Calculate allocations (Balanced profile)
    total_investment = portfolio.total_investment
    num_stocks = len(selected_stocks)
    
    # Balanced: Weighted allocation
    weights = np.linspace(20, 8, num_stocks)  # [20, 17.6, 15.2, 12.8, 10.4, 8]
    weights = weights / weights.sum() * 100  # Normalize to 100%
    
    allocations = []
    for i, stock in enumerate(selected_stocks):
        allocation_percent = weights[i]
        investment_amount = (total_investment * allocation_percent) / 100
        
        allocations.append({
            "ticker": stock['ticker'],
            "company_name": stock['company_name'],
            "allocation_percent": allocation_percent,
            "investment_amount": investment_amount,
            "predicted_return_7d": predictions[stock['ticker']]['predicted_return_7d']
        })
    
    # Step 6: Create portfolio in database
    portfolio_record = Portfolio(
        client_id=portfolio.client_id,
        total_investment=total_investment,
        risk_profile=risk_profile
    )
    db.add(portfolio_record)
    db.flush()
    
    # Create holdings
    for allocation in allocations:
        holding = PortfolioHolding(
            portfolio_id=portfolio_record.id,
            stock_ticker=allocation['ticker'],
            company_name=allocation['company_name'],
            allocation_percent=allocation['allocation_percent'],
            investment_amount=allocation['investment_amount'],
            predicted_return_7d=allocation['predicted_return_7d']
        )
        db.add(holding)
    
    db.commit()
    
    return {
        "portfolio_id": portfolio_record.id,
        "holdings": allocations
    }
```

**Example Result:**
```json
{
    "portfolio_id": 1,
    "total_investment": 100000,
    "risk_profile": "Balanced",
    "holdings": [
        {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "allocation_percent": 20.0,
            "investment_amount": 20000,
            "predicted_return_7d": 2.5
        },
        {
            "ticker": "TCS.NS",
            "company_name": "Tata Consultancy Services",
            "allocation_percent": 17.6,
            "investment_amount": 17600,
            "predicted_return_7d": 2.3
        },
        ...
    ]
}
```

**Result:**
- Portfolio generated with 6 stocks
- LSTM predictions for each stock
- Optimal allocations calculated
- Charts displayed (pie chart, bar chart)
- User proceeds to dashboard

---

#### **Step 4: Dashboard**

**User Actions:**
1. Sees comprehensive dashboard
2. Views all information in one place:
   - Client profile summary
   - Risk assessment results
   - Portfolio breakdown
   - Advanced visualizations

**What Happens Behind the Scenes:**
```
Dashboard Load → Multiple API Calls → Aggregate Data → 
Generate Visualizations → Display Unified View
```

**Technical Flow:**
```python
# Frontend fetches all data
client_data = requests.get(f"/api/clients/{client_id}")
risk_data = requests.get(f"/api/risk/{client_id}")
portfolio_data = requests.get(f"/api/portfolio/{client_id}")

# Aggregate and display
display_client_summary(client_data)
display_risk_gauge(risk_data)
display_portfolio_charts(portfolio_data)
display_performance_projections(portfolio_data)
```

---

## 💻 Developer Flow (Technical Architecture)

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│                    (Streamlit Frontend)                      │
│  - Interactive Forms                                         │
│  - Real-time Visualizations                                  │
│  - User Input Handling                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP Requests (REST API)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer                                 │
│                    (FastAPI Backend)                         │
│  - REST Endpoints                                            │
│  - Request Validation (Pydantic)                             │
│  - Error Handling                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Service   │ │     ML      │ │  Database   │
│   Layer     │ │   Models    │ │   Layer     │
│             │ │             │ │             │
│ - Portfolio │ │ - Random    │ │ - PostgreSQL│
│   Service   │ │   Forest    │ │ - SQLAlchemy│
│ - YFinance  │ │ - LSTM      │ │ - Models    │
│   Service   │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

### Request Flow Example: Risk Assessment

```
1. User fills questionnaire in Streamlit
   ↓
2. Streamlit sends POST request:
   POST http://localhost:8000/api/risk/assess
   Body: {
       "client_id": 1,
       "investment_horizon": 10,
       "volatility_tolerance": 3,
       ...
   }
   ↓
3. FastAPI receives request
   @app.post("/api/risk/assess")
   ↓
4. Pydantic validates data
   RiskAssessmentCreate schema validation
   ↓
5. Router processes request
   routers/risk.py → assess_risk()
   ↓
6. Service layer fetches client data
   db.query(Client).filter(Client.id == client_id)
   ↓
7. ML Model loads and predicts
   RiskClassifier.load_model()
   prediction = model.predict(features)
   ↓
8. Result saved to database
   db.add(RiskAssessment(...))
   db.commit()
   ↓
9. Response sent back
   JSON: {"risk_profile": "Balanced", "confidence": 84.5}
   ↓
10. Streamlit displays result
    Gauge chart + metrics
```

---

### Database Schema & Relationships

```
clients (1) ──→ (many) risk_assessments
clients (1) ──→ (many) portfolios
portfolios (1) ──→ (many) portfolio_holdings

Tables:
- clients: id, client_name, age, monthly_income, ...
- risk_assessments: id, client_id, risk_profile, confidence_score, ...
- portfolios: id, client_id, total_investment, risk_profile, ...
- portfolio_holdings: id, portfolio_id, stock_ticker, allocation_percent, ...
- stock_predictions: id, stock_ticker, predicted_price_1d, ...
- market_data: id, stock_ticker, date, close_price, volatility, ...
```

---

## 📊 Data Flow & Processing

### Complete Data Flow Diagram

```
User Input
    ↓
Streamlit UI
    ↓
FastAPI Endpoint
    ↓
┌─────────────────────────────────────┐
│  Data Processing Pipeline           │
│                                     │
│  1. Validate Input (Pydantic)       │
│  2. Fetch from Database (SQLAlchemy) │
│  3. Process with ML Models          │
│  4. Calculate Results               │
│  5. Save to Database                │
│  6. Return Response                 │
└─────────────────────────────────────┘
    ↓
JSON Response
    ↓
Streamlit Display
    ↓
User Sees Results
```

---

### Example: Portfolio Generation Data Flow

```
Input: {"client_id": 1, "total_investment": 100000}
    ↓
1. Fetch Client Data
   SELECT * FROM clients WHERE id = 1
   → {age: 30, income: 50000, ...}
    ↓
2. Fetch Risk Profile
   SELECT * FROM risk_assessments WHERE client_id = 1 ORDER BY date DESC LIMIT 1
   → {risk_profile: "Balanced", confidence: 84.5}
    ↓
3. Fetch Market Data
   SELECT * FROM market_data WHERE date >= '2024-01-01'
   → [{ticker: "RELIANCE.NS", close_price: 2500, ...}, ...]
    ↓
4. Filter Stocks by Risk
   Filter: risk_classification IN ['Low Risk', 'Medium Risk']
   → 15 stocks match
    ↓
5. Load LSTM Models
   Load: models/lstm_RELIANCE.h5, models/scaler_RELIANCE.pkl
   → Model ready for prediction
    ↓
6. Predict Prices
   For each stock:
     - Get last 60 days prices
     - Normalize
     - Predict 7-day price
     - Calculate return
   → Predictions: {RELIANCE: 2.5%, TCS: 2.3%, ...}
    ↓
7. Select Top Stocks
   Sort by predicted_return_7d DESC
   Take top 6
   → Selected: [RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, ITC]
    ↓
8. Calculate Allocations
   Balanced profile → Weighted allocation
   → Allocations: [20%, 17.6%, 15.2%, 12.8%, 10.4%, 8%]
    ↓
9. Save Portfolio
   INSERT INTO portfolios (client_id, total_investment, ...)
   INSERT INTO portfolio_holdings (portfolio_id, ticker, allocation_percent, ...)
   → Portfolio ID: 1
    ↓
10. Return Response
    JSON: {portfolio_id: 1, holdings: [...]}
    ↓
11. Display in UI
    Pie chart, bar chart, table
```

---

## 🔍 How Results are Fetched

### Risk Assessment Results

**Process:**
1. **User Input** → Questionnaire answers
2. **Feature Extraction** → Combine client data + questionnaire
3. **Model Loading** → Load Random Forest from disk
4. **Prediction** → Model processes features
5. **Result Calculation** → Risk profile + confidence
6. **Database Save** → Store assessment
7. **Response** → Return JSON

**Code Flow:**
```python
# Step 1: User answers
questionnaire = {
    "investment_horizon": 10,
    "volatility_tolerance": 3,
    ...
}

# Step 2: Fetch client data
client = db.query(Client).filter(Client.id == 1).first()

# Step 3: Prepare features
features = [
    client.age,                    # 30
    client.monthly_income,          # 50000
    client.monthly_expenses,        # 30000
    client.dependents,              # 0
    client.emergency_fund,          # 200000
    questionnaire['investment_horizon'],      # 10
    questionnaire['volatility_tolerance'],    # 3
    questionnaire['financial_knowledge'],     # 2
    questionnaire['income_stability'],        # 4
    questionnaire['risk_capacity']            # 3
]

# Step 4: Load model
classifier = RiskClassifier()
classifier.load_model()  # Loads models/risk_classifier.pkl

# Step 5: Predict
prediction = classifier.model.predict([features])[0]
probabilities = classifier.model.predict_proba([features])[0]

# Step 6: Decode result
risk_profile = classifier.label_encoder.inverse_transform([prediction])[0]
confidence = max(probabilities) * 100

# Step 7: Calculate risk score
risk_score = (
    min(10, 15) * 1.0 +      # investment_horizon capped at 15
    3 * 3.75 +                # volatility_tolerance
    2 * 3.75 +                # financial_knowledge
    4 * 3.75 +                # income_stability
    3 * 3.75                  # risk_capacity
)  # = 52.5

# Step 8: Save
risk_assessment = RiskAssessment(
    client_id=1,
    risk_profile=risk_profile,
    risk_score=risk_score,
    confidence_score=confidence
)
db.add(risk_assessment)
db.commit()

# Step 9: Return
return {
    "risk_profile": "Balanced",
    "risk_score": 52.5,
    "confidence_score": 84.5
}
```

---

### Stock Prediction Results

**Process:**
1. **Stock Selection** → Filter by risk profile
2. **Model Loading** → Load LSTM model for each stock
3. **Data Preparation** → Get last 60 days, normalize
4. **Prediction** → LSTM predicts future prices
5. **Return Calculation** → Calculate percentage returns
6. **Response** → Return predictions

**Code Flow:**
```python
# Step 1: Select stock
ticker = "RELIANCE.NS"

# Step 2: Load model
predictor = StockPredictor()
predictor.load_model(ticker)  # Loads models/lstm_RELIANCE.h5 and scaler

# Step 3: Fetch recent data
stock = yf.Ticker(ticker)
df = stock.history(period="3mo")
last_60_days = df['Close'].tail(60).values  # [2450, 2460, ..., 2500]

# Step 4: Normalize
scaled_prices = predictor.scaler.transform(last_60_days.reshape(-1, 1))
# [0.735, 0.738, ..., 0.750]

# Step 5: Reshape for LSTM
X_input = scaled_prices.reshape(1, 60, 1)  # Shape: (1, 60, 1)

# Step 6: Predict 1 day ahead
pred_1d_scaled = predictor.model.predict(X_input)[0, 0]  # 0.752
pred_1d_price = predictor.scaler.inverse_transform([[pred_1d_scaled]])[0, 0]  # 2510

# Step 7: Predict 7 days ahead (iterative)
temp_input = X_input.copy()
for day in range(7):
    pred = predictor.model.predict(temp_input)
    # Update input: remove first day, add prediction
    new_input = np.append(temp_input[0, 1:, 0], pred[0, 0])
    temp_input = new_input.reshape(1, 60, 1)

pred_7d_scaled = pred[0, 0]
pred_7d_price = predictor.scaler.inverse_transform([[pred_7d_scaled]])[0, 0]  # 2625

# Step 8: Calculate returns
current_price = df['Close'].iloc[-1]  # 2500
return_7d = ((pred_7d_price - current_price) / current_price) * 100  # 5.0%

# Step 9: Calculate confidence (simplified)
confidence = 85.0  # Based on prediction stability

# Step 10: Return
return {
    "ticker": "RELIANCE.NS",
    "current_price": 2500.0,
    "predicted_price_1d": 2510.0,
    "predicted_price_7d": 2625.0,
    "predicted_return_7d": 5.0,
    "confidence_7d": 85.0
}
```

---

## 🎓 How Models are Trained

### Random Forest Training Process

**Step-by-Step:**

```python
# 1. Generate Training Data
def generate_training_data(n_samples=2000):
    data = []
    labels = []
    
    for _ in range(2000):
        # Create synthetic client profile
        age = random.randint(25, 65)
        income = random.uniform(30000, 500000)
        expenses = income * random.uniform(0.4, 0.8)
        ...
        
        # Calculate risk score (using rules)
        score = calculate_risk_score(...)
        
        # Determine risk profile
        if score <= 22.5:
            risk_profile = "Conservative"
        elif score <= 37.5:
            risk_profile = "Moderate Conservative"
        ...
        
        data.append([age, income, expenses, ...])
        labels.append(risk_profile)
    
    return np.array(data), np.array(labels)

# 2. Prepare Data
X, y = generate_training_data(2000)
# X: (2000, 10) - 2000 samples, 10 features
# y: (2000,) - 2000 labels

# 3. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# "Conservative" → 0, "Balanced" → 2, etc.

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
# X_train: (1600, 10)
# X_test: (400, 10)

# 5. Build Model
model = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Max tree depth
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples in leaf
    random_state=42
)

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
train_score = model.score(X_train, y_train)  # 97.75%
test_score = model.score(X_test, y_test)      # 84.00%

# 8. Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
# [0.84, 0.85, 0.83, 0.84, 0.85]
# Mean: 84.19% ± 4.27%

# 9. Save Model
joblib.dump(model, "models/risk_classifier.pkl")
joblib.dump(label_encoder, "models/risk_label_encoder.pkl")
```

**Training Time:** ~10 seconds  
**Model Size:** 3.7 MB

---

### LSTM Training Process

**Step-by-Step:**

```python
# 1. Fetch Stock Data
ticker = "RELIANCE.NS"
stock = yf.Ticker(ticker)
df = stock.history(period="5y")  # 5 years of data
# Result: ~1,304 days (only weekdays)

# 2. Extract Prices
prices = df['Close'].values  # [2450, 2460, 2475, ..., 2500]
# Shape: (1304,)

# 3. Normalize Prices
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
# [0.735, 0.738, 0.742, ..., 0.750]
# All values between 0 and 1

# 4. Create Sequences
sequence_length = 60
X, y = [], []

for i in range(60, len(scaled_prices)):
    X.append(scaled_prices[i-60:i, 0])  # 60 days
    y.append(scaled_prices[i, 0])        # Next day

X = np.array(X)  # Shape: (1244, 60)
y = np.array(y)  # Shape: (1244,)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (1244, 60, 1)

# 5. Split Data
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]      # (995, 60, 1)
X_val = X[split_idx:]        # (249, 60, 1)
y_train = y[:split_idx]      # (995,)
y_val = y[split_idx:]         # (249,)

# 6. Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=10),  # Stop if no improvement
        ModelCheckpoint(save_best_only=True)  # Save best model
    ],
    verbose=0
)

# Training Process:
# Epoch 1: loss=0.015, val_loss=0.012
# Epoch 2: loss=0.010, val_loss=0.008
# ...
# Epoch 15: loss=0.002, val_loss=0.0018  ← Best model saved
# ...
# Epoch 30: Training stops (early stopping)

# 8. Evaluate
final_loss = history.history['val_loss'][-1]  # 0.001655

# 9. Save Model
model.save("models/lstm_RELIANCE.h5")
joblib.dump(scaler, "models/scaler_RELIANCE.pkl")
```

**Training Time:** ~2-5 minutes per stock  
**Model Size:** ~426 KB per stock  
**Total Models:** 10 stocks × 426 KB = ~4.3 MB

---

### Training Data Details

**Random Forest:**
- **Type:** Synthetic (realistic)
- **Size:** 2,000 samples
- **Features:** 10 per sample
- **Total Data Points:** 20,000
- **Generation:** Based on financial advisory rules

**LSTM:**
- **Type:** Time series (synthetic realistic)
- **Size:** ~1,304 days per stock
- **Sequences:** ~995 per stock
- **Total Sequences:** ~9,950 (across 10 stocks)
- **Generation:** Geometric Brownian Motion (realistic stock patterns)

---

This covers the technical details. The next document will cover interview preparation and Q&A.

