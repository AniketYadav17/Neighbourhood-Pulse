import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def run():
    print("🧠 Training Model...")
    
    # Load Unified Data
    df = pd.read_csv("data/processed/unified_training_data.csv")
    
    # Define Features (X) and Target (y)
    # Drop non-numeric and target columns
    X = df.drop(columns=['avg_house_price', 'h3_index']) 
    y = df['avg_house_price']
    
    print(f"   Features: {list(X.columns)}")
    
    # Train/Test Split (Tiny data, so this is just symbolic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost
    model = XGBRegressor(objective='reg:squarederror', n_estimators=10)
    model.fit(X_train, y_train)
    
    # Mock Evaluation
    score = model.score(X_train, y_train)
    print(f"   Model R2 Score (on training): {score:.2f}")
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    with open("models/gentrification_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Model saved to models/gentrification_model.pkl")

if __name__ == "__main__":
    run()