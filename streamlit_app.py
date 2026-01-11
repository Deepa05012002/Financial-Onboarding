"""
Streamlit Frontend - Advanced Financial Onboarding System
World-class UI with top-notch visualizations
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Financial Onboarding System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'client_id' not in st.session_state:
    st.session_state.client_id = None
if 'risk_profile' not in st.session_state:
    st.session_state.risk_profile = None
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None


def make_api_request(endpoint, method="GET", data=None):
    """Make API request"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


# Welcome Header (Always visible)
st.markdown('<h1 class="main-header">Welcome to the Advanced Financial Onboarding System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white; margin-bottom: 30px;'>
    <h3>Investment Planning Platform</h3>
    <p>Get personalized investment recommendations powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("💰 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["👤 Client Onboarding", "📊 Risk Assessment", "📈 Portfolio Builder", "🎯 Dashboard"]
)

# Client Onboarding Page (Default)
if page == "👤 Client Onboarding":
    st.markdown("### 👤 Step 1: Client Onboarding")
    st.markdown("---")
    
    with st.form("client_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            client_name = st.text_input("Full Name *", placeholder="Enter your name")
            age = st.number_input("Age *", min_value=18, max_value=100, value=30)
            monthly_income = st.number_input("Monthly Income (₹) *", min_value=0, value=50000, step=1000)
        
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses (₹) *", min_value=0, value=30000, step=1000)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=0, step=10000)
        
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-Employed", "Retired", "Student", "Unemployed"]
        )
        
        submitted = st.form_submit_button("💾 Create Client Profile", use_container_width=True)
        
        if submitted:
            if not client_name or monthly_income == 0:
                st.error("Please fill in all required fields (*)")
            else:
                client_data = {
                    "client_name": client_name,
                    "age": age,
                    "monthly_income": float(monthly_income),
                    "monthly_expenses": float(monthly_expenses),
                    "dependents": dependents,
                    "employment_status": employment_status,
                    "emergency_fund": float(emergency_fund)
                }
                
                result = make_api_request("/api/clients/", method="POST", data=client_data)
                
                if result:
                    st.session_state.client_id = result['id']
                    st.success(f"✅ Client profile created! Client ID: {result['id']}")
                    st.balloons()
                    
                    # Display summary
                    st.subheader("Profile Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Monthly Surplus", f"₹{monthly_income - monthly_expenses:,.0f}")
                    with col2:
                        st.metric("Savings Rate", f"{(monthly_income - monthly_expenses) / monthly_income * 100:.1f}%")
                    with col3:
                        st.metric("Emergency Fund", f"₹{emergency_fund:,.0f}")

# Risk Assessment Page
elif page == "📊 Risk Assessment":
    st.markdown("### 📊 Step 2: Risk Assessment")
    st.markdown("---")
    
    if not st.session_state.client_id:
        st.warning("⚠️ Please create a client profile first!")
        st.info("Go to 'Client Onboarding' page to create your profile.")
    else:
        st.info(f"📋 Assessing risk for Client ID: {st.session_state.client_id}")
        
        with st.form("risk_form"):
            st.subheader("Investment Questionnaire")
            
            investment_horizon = st.slider(
                "Investment Horizon (years)",
                min_value=1,
                max_value=30,
                value=5,
                help="How long do you plan to invest?"
            )
            
            volatility_tolerance = st.select_slider(
                "Volatility Tolerance",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    1: "Very Low (Prefer stability)",
                    2: "Low (Some fluctuations OK)",
                    3: "Medium (Comfortable with ups/downs)",
                    4: "High (Seek high returns)"
                }[x]
            )
            
            financial_knowledge = st.select_slider(
                "Financial Knowledge",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    1: "Beginner",
                    2: "Basic",
                    3: "Intermediate",
                    4: "Advanced"
                }[x]
            )
            
            income_stability = st.select_slider(
                "Income Stability",
                options=[1, 2, 3, 4],
                value=3,
                format_func=lambda x: {
                    1: "Very Unstable",
                    2: "Somewhat Unstable",
                    3: "Stable",
                    4: "Very Stable"
                }[x]
            )
            
            risk_capacity = st.select_slider(
                "Risk Capacity",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    1: "Very Low",
                    2: "Low",
                    3: "Medium",
                    4: "High"
                }[x]
            )
            
            submitted = st.form_submit_button("🤖 Assess Risk (ML Model)", use_container_width=True)
            
            if submitted:
                with st.spinner("🤖 ML Model is analyzing your risk profile..."):
                    assessment_data = {
                        "client_id": st.session_state.client_id,
                        "investment_horizon": investment_horizon,
                        "volatility_tolerance": volatility_tolerance,
                        "financial_knowledge": financial_knowledge,
                        "income_stability": income_stability,
                        "risk_capacity": risk_capacity
                    }
                    
                    result = make_api_request("/api/risk/assess", method="POST", data=assessment_data)
                    
                    if result:
                        st.session_state.risk_profile = result['risk_profile']
                        
                        st.success("✅ Risk Assessment Complete!")
                        
                        # Display results with visualizations
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Risk Profile Gauge Chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=float(result['risk_score']),
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Risk Score"},
                                delta={'reference': 37.5},
                                gauge={
                                    'axis': {'range': [None, 75]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 22.5], 'color': "lightgreen"},
                                        {'range': [22.5, 37.5], 'color': "yellow"},
                                        {'range': [37.5, 52.5], 'color': "orange"},
                                        {'range': [52.5, 75], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 75
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Risk Profile", result['risk_profile'])
                            st.metric("Risk Score", f"{result['risk_score']:.1f}/75")
                            st.metric("ML Confidence", f"{result['confidence_score']:.1f}%")
                        
                        # Risk Profile Explanation
                        risk_explanations = {
                            "Conservative": "🛡️ You prefer safety and stability. Your portfolio will focus on low-risk investments.",
                            "Moderate Conservative": "⚖️ You prefer a balanced approach with slight preference for safety.",
                            "Balanced": "🎯 You're comfortable with moderate risk for moderate returns.",
                            "Growth Oriented": "📈 You seek higher returns and are comfortable with higher risk.",
                            "Aggressive Growth": "🚀 You seek maximum returns and are comfortable with high volatility."
                        }
                        
                        st.info(risk_explanations.get(result['risk_profile'], ""))

# Portfolio Builder Page
elif page == "📈 Portfolio Builder":
    st.markdown("### 📈 Step 3: Portfolio Builder")
    st.markdown("---")
    
    if not st.session_state.client_id:
        st.warning("⚠️ Please create a client profile first!")
    elif not st.session_state.risk_profile:
        st.warning("⚠️ Please complete risk assessment first!")
    else:
        st.info(f"💰 Building portfolio for {st.session_state.risk_profile} investor")
        
        investment_amount = st.number_input(
            "Investment Amount (₹)",
            min_value=10000,
            value=100000,
            step=10000,
            help="Enter the amount you want to invest"
        )
        
        if st.button("🎯 Generate Portfolio (with LSTM Predictions)", use_container_width=True):
            with st.spinner("🤖 Generating personalized portfolio with ML predictions..."):
                portfolio_data = {
                    "client_id": st.session_state.client_id,
                    "total_investment": float(investment_amount)
                }
                
                result = make_api_request("/api/portfolio/generate", method="POST", data=portfolio_data)
                
                if result:
                    st.session_state.portfolio_data = result
                    st.success("✅ Portfolio Generated Successfully!")
                    st.balloons()
                    
                    # Portfolio Holdings Table
                    st.subheader("📊 Portfolio Holdings")
                    holdings_df = pd.DataFrame(result['holdings'])
                    st.dataframe(holdings_df, use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Allocation Pie Chart
                        fig = px.pie(
                            holdings_df,
                            values='allocation_percent',
                            names='company_name',
                            title="Portfolio Allocation",
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Expected Returns Bar Chart
                        fig = px.bar(
                            holdings_df,
                            x='company_name',
                            y=['predicted_return_7d', 'predicted_return_30d'],
                            title="Predicted Returns (LSTM)",
                            barmode='group',
                            labels={'value': 'Return %', 'company_name': 'Company'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sector Diversification
                    if 'sector' in holdings_df.columns:
                        sector_counts = holdings_df['sector'].value_counts()
                        fig = px.bar(
                            x=sector_counts.index,
                            y=sector_counts.values,
                            title="Sector Diversification",
                            labels={'x': 'Sector', 'y': 'Number of Holdings'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

# Dashboard Page
elif page == "🎯 Dashboard":
    st.markdown("### 🎯 Step 4: Comprehensive Dashboard")
    st.markdown("---")
    
    if not st.session_state.client_id:
        st.warning("⚠️ Please complete the onboarding process first!")
    else:
        # Get client data
        client_data = make_api_request(f"/api/clients/{st.session_state.client_id}")
        
        if client_data:
            # Client Overview
            st.subheader("👤 Client Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Name", client_data['client_name'])
            with col2:
                st.metric("Age", client_data['age'])
            with col3:
                st.metric("Monthly Income", f"₹{client_data['monthly_income']:,.0f}")
            with col4:
                st.metric("Monthly Expenses", f"₹{client_data['monthly_expenses']:,.0f}")
            
            # Risk Profile
            if st.session_state.risk_profile:
                risk_data = make_api_request(f"/api/risk/{st.session_state.client_id}")
                if risk_data:
                    st.subheader("📊 Risk Profile")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Risk Profile", risk_data['risk_profile'])
                        st.metric("Risk Score", f"{risk_data['risk_score']:.1f}/75")
                    with col2:
                        st.metric("ML Confidence", f"{risk_data['confidence_score']:.1f}%")
            
            # Portfolio Summary
            portfolio_data = make_api_request(f"/api/portfolio/{st.session_state.client_id}")
            if portfolio_data:
                st.subheader("💼 Portfolio Summary")
                st.metric("Total Investment", f"₹{portfolio_data['total_investment']:,.0f}")
                
                holdings_df = pd.DataFrame(portfolio_data['holdings'])
                
                # Performance Projection
                total_return_7d = (holdings_df['predicted_return_7d'] * holdings_df['investment_amount'] / 100).sum()
                total_return_30d = (holdings_df['predicted_return_30d'] * holdings_df['investment_amount'] / 100).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("7-Day Projected Return", f"₹{total_return_7d:,.0f}")
                with col2:
                    st.metric("30-Day Projected Return", f"₹{total_return_30d:,.0f}")
                
                # Advanced Visualizations
                st.subheader("📈 Advanced Analytics")
                
                # Risk-Return Scatter
                fig = px.scatter(
                    holdings_df,
                    x='predicted_return_30d',
                    y='allocation_percent',
                    size='investment_amount',
                    color='risk_classification',
                    hover_name='company_name',
                    title="Risk-Return Analysis",
                    labels={
                        'predicted_return_30d': 'Predicted Return (%)',
                        'allocation_percent': 'Allocation %',
                        'risk_classification': 'Risk Level'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Advanced Financial Onboarding System | Powered by ML Models | Built with Streamlit & FastAPI"
    "</div>",
    unsafe_allow_html=True
)

