import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
from loguru import logger

print("🧠 TRAINING NLP MODEL (Simple Logistic Regression)")
print("=" * 80)

try:
    # Load data
    print("\n1️⃣ Loading NLP data...")
    df_train = pd.read_csv('datasets/processed/nlp/train.csv')
    df_test = pd.read_csv('datasets/processed/nlp/test.csv')
    
    print(f"  ✅ Train: {len(df_train)} samples")
    print(f"  ✅ Test: {len(df_test)} samples")
    
    # Use simple features we created
    X_train = df_train[['len', 'at_count', 'slash_count']]
    y_train = df_train['label']
    
    X_test = df_test[['len', 'at_count', 'slash_count']]
    y_test = df_test['label']
    
    # Train simple logistic regression
    print("\n2️⃣ Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n  ✅ Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  ✅ Testing Accuracy:   {test_acc*100:.2f}%")
    print(f"  ✅ Overfitting Gap:    {(train_acc-test_acc)*100:.2f}%")
    
    # Cross-validation
    print("\n3️⃣ Cross-validation:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save
    Path('backend/models').mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open('backend/models/nlp_model.pkl', 'wb'))
    
    print("\n✅ NLP MODEL SAVED!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()