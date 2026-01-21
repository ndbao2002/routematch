import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# ==========================================
# 1. PREPROCESSING
# ==========================================

def get_preprocessor(X):
    categorical_features = ['requested_vehicle_type', 'service_type', 'is_raining']
    numeric_features = [c for c in X.columns if c not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

# ==========================================
# 2. BASELINE MODEL
# ==========================================

def train_baseline_pipeline(X_train, y_train, X_test, y_test):
    print("\n--- 📉 Training Logistic Regression Baseline ---")
    preprocessor = get_preprocessor(X_train)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"   ✅ Baseline AUC: {roc_auc_score(y_test, y_prob):.4f}")
    return model

# ==========================================
# 3. XGBOOST (Tuned)
# ==========================================

def tune_xgboost_pipeline(X_train, y_train, X_test, y_test):
    print("\n--- 🧪 Tuning XGBoost Hyperparameters ---")
    preprocessor = get_preprocessor(X_train)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss'
        ))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    # TimeSeriesSplit prevents Future Data Leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=5, # Kept low for speed in demo
        scoring='roc_auc',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print(f"   🚀 Tuned XGBoost AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    return best_model

# ==========================================
# 4. CALIBRATION ANALYSIS
# ==========================================

def analyze_calibration(model, X_test, y_test, model_name="Model"):
    print(f"\n--- 📊 Analyzing Calibration for {model_name} ---")
    
    # 1. Get Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 2. Brier Score (MSE of probabilities)
    # Lower is better. 0.0 = Perfect, 0.25 = Random Guessing (if balanced)
    brier = brier_score_loss(y_test, y_prob)
    print(f"   🎯 Brier Score: {brier:.4f} (Lower is better)")
    
    # 3. Calibration Curve Data
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # 4. Plotting, we should save an image to data/plots/
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title(f'Calibration Curve - {model_name}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    plt.grid()
    os.makedirs("data/plots", exist_ok=True)
    plt.savefig(f"data/plots/calibration_{model_name.replace(' ', '_').lower()}.png")
    plt.close()
        
    return brier

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Check data existence
    if not os.path.exists("data/processed/feature_data.csv"):
        print("❌ Error: Feature data not found.")
        exit()
        
    df = pd.read_csv("data/processed/feature_data.csv")

    # Time-based Split
    split_idx = int(len(df) * 0.8)
    X = df.drop(columns=['is_accepted', 'order_id']) # Let's forget order_id here
    y = df['is_accepted']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 1. Train Baseline
    baseline_model = train_baseline_pipeline(X_train, y_train, X_test, y_test)
    
    # 2. Train Tuned XGBoost
    xgb_model = tune_xgboost_pipeline(X_train, y_train, X_test, y_test)
    
    # 3. Compare Calibration
    # If XGBoost has higher AUC but worse Brier score, it's "Overconfident"
    analyze_calibration(baseline_model, X_test, y_test, "Logistic Baseline")
    analyze_calibration(xgb_model, X_test, y_test, "Tuned XGBoost")