"""
Script to train ML models
Run this once to train and save the models
"""
import os
from app.ml_models.risk_classifier import RiskClassifier
from app.ml_models.stock_predictor import StockPredictor

def train_risk_classifier():
    """Train Random Forest risk classifier"""
    print("=" * 60)
    print("Training Risk Classifier (Random Forest)")
    print("=" * 60)
    
    classifier = RiskClassifier()
    classifier.train(n_samples=2000, n_estimators=100)
    
    print("\n✅ Risk Classifier trained successfully!")
    print(f"Model saved to: {classifier.model_dir}/risk_classifier.pkl")


def train_stock_predictors():
    """Train LSTM models for top stocks"""
    print("=" * 60)
    print("Training Stock Predictors (LSTM)")
    print("=" * 60)
    print("Using robust data fetching with multiple retry strategies...")
    print("This may take 20-40 minutes depending on API response times.\n")
    
    # Top 10 NIFTY 50 stocks for training
    top_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'BHARTIARTL.NS', 'MARUTI.NS', 'ASIANPAINT.NS'
    ]
    
    predictor = StockPredictor()
    successful = 0
    failed = []
    
    for i, ticker in enumerate(top_stocks, 1):
        print(f"\n[{i}/{len(top_stocks)}] Training model for {ticker}...")
        try:
            predictor.train(ticker, epochs=30, batch_size=32)
            successful += 1
            print(f"  ✅ {ticker} trained successfully!")
        except Exception as e:
            error_msg = str(e)[:100]  # Truncate long errors
            print(f"  ❌ Failed: {error_msg}")
            failed.append(ticker)
            # Wait before next attempt to avoid rate limiting
            import time
            time.sleep(3)
            continue
    
    print(f"\n" + "=" * 60)
    print(f"✅ Stock predictors training completed!")
    print(f"   Successfully trained: {successful}/{len(top_stocks)}")
    print(f"   Failed: {len(failed)}/{len(top_stocks)}")
    
    if failed:
        print(f"\n⚠️ Failed tickers: {', '.join(failed)}")
        print("\n💡 Solutions:")
        print("   1. Wait 5-10 minutes and retry (API rate limiting)")
        print("   2. Check internet connection")
        print("   3. Verify ticker symbols are correct")
        print("   4. Try training one stock at a time for debugging")
        print("\n   Retry command: python3 train_models.py")
    else:
        print("\n🎉 All models trained successfully!")


if __name__ == "__main__":
    print("\n🚀 Starting ML Model Training...")
    print("This may take 10-15 minutes depending on your system.\n")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train models
    try:
        train_risk_classifier()
        print("\n")
        train_stock_predictors()
        
        print("\n" + "=" * 60)
        print("✅ All models trained successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

