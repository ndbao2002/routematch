import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib 
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
    # We must exclude 'order_id' if it accidentally slipped in here
    categorical_features = ['requested_vehicle_type', 'service_type', 'is_raining']
    
    # Identify numeric features dynamically
    numeric_features = [c for c in X.columns 
                       if c not in categorical_features 
                       and c != 'order_id']

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

# ==========================================
# 2. RECALL @ K METRIC (New Addition)
# ==========================================

def calculate_recall_at_k(df_test: pd.DataFrame, y_prob: np.ndarray, k=5):
    """
    Calculates the 'Hit Rate' @ K.
    Did the driver who ACTUALLY accepted the order appear in the Top K predictions?
    """
    # 1. Attach predictions to the test dataframe
    df_eval = df_test.copy()
    df_eval['score'] = y_prob
    
    # 2. Group by Order ID to rank drivers per order
    # We only care about orders that had at least one acceptance
    accepted_orders = df_eval[df_eval['is_accepted'] == 1]['order_id'].unique()
    df_eval = df_eval[df_eval['order_id'].isin(accepted_orders)]
    
    hits = 0
    total_orders = len(df_eval['order_id'].unique())
    
    if total_orders == 0:
        return 0.0

    # 3. Iterate through each order to check rank
    # (Vectorized implementation is possible but this is clearer for understanding)
    for oid, group in df_eval.groupby('order_id'):
        # Sort drivers by model score (Highest first)
        sorted_group = group.sort_values('score', ascending=False)
        
        # Get the top K drivers
        top_k = sorted_group.head(k)
        
        # Did any of the Top K drivers accept?
        if (top_k['is_accepted'] == 1).any():
            hits += 1
            
    recall = hits / total_orders
    print(f"   🎯 Recall@{k}: {recall:.4f} (Hit Rate)")
    return recall

# ==========================================
# 3. XGBOOST (Tuned)
# ==========================================

def tune_xgboost_pipeline(X_train, y_train, X_test, y_test, df_test_full):
    print("\n--- 🧪 Tuning XGBoost Hyperparameters ---")
    
    # Drop order_id for Training! (Metadata only)
    X_train_clean = X_train.drop(columns=['order_id'], errors='ignore')
    X_test_clean = X_test.drop(columns=['order_id'], errors='ignore')
    
    preprocessor = get_preprocessor(X_train_clean)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss'
        ))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.9, 1.0]
    }
    
    # TimeSeriesSplit prevents Future Data Leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20, 
        scoring='roc_auc',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train_clean, y_train)
    best_model = search.best_estimator_
    
    # AUC
    y_prob = best_model.predict_proba(X_test_clean)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"   🚀 Tuned XGBoost AUC: {auc:.4f}")
    
    # RECALL @ K (Using df_test_full which still has order_id)
    calculate_recall_at_k(df_test_full, y_prob, k=1)
    calculate_recall_at_k(df_test_full, y_prob, k=3)
    calculate_recall_at_k(df_test_full, y_prob, k=5)
    
    return best_model

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    if not os.path.exists("data/processed/feature_data.csv"):
        print("❌ Error: Feature data not found.")
        exit()
        
    df = pd.read_csv("data/processed/feature_data.csv")

    # Time-based Split
    split_idx = int(len(df) * 0.8)
    
    # X contains Features + Metadata (order_id)
    # y is Target
    X = df.drop(columns=['is_accepted'])
    y = df['is_accepted']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Pass X_test (with order_id) for metric calc, but handle dropping inside function
    xgb_model = tune_xgboost_pipeline(X_train, y_train, X_test, y_test, df.iloc[split_idx:])
    
    # Save Model for Dispatcher
    print("\n--- 💾 Saving Model ---")
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(xgb_model, "models/xgb_scoring_model.pkl")
    print("   ✅ Model saved to models/xgb_scoring_model.pkl")