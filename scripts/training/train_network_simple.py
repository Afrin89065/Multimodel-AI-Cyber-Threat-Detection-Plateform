import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pickle
from pathlib import Path

print("🔌 TRAINING NETWORK MODEL (XGBoost)")
print("=" * 80)

try:
    # Load REAL data
    print("\n1️⃣ Loading Network data...")
    df_train = pd.read_csv('datasets/processed/network/train.csv')
    df_test = pd.read_csv('datasets/processed/network/test.csv')
    
    print(f"  ✅ Train: {len(df_train)} samples")
    print(f"  ✅ Test: {len(df_test)} samples")
    
    # Extract
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values
    
    # Scale
    print("\n2️⃣ Preprocessing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\n3️⃣ Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train, verbose=0)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\n  ✅ Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  ✅ Testing Accuracy:   {test_acc*100:.2f}%")
    print(f"  ✅ Overfitting Gap:    {(train_acc-test_acc)*100:.2f}%")
    
    # Cross-validation
    print("\n4️⃣ Cross-validation:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save
    Path('backend/models').mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open('backend/models/network_model.pkl', 'wb'))
    pickle.dump(scaler, open('backend/models/network_scaler.pkl', 'wb'))
    
    print("\n✅ NETWORK MODEL SAVED!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()