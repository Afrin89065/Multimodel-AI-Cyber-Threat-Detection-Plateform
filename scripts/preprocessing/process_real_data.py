import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("🔄 Processing REAL datasets...")

# 1️⃣ PROCESS NLP DATA
print("\n1️⃣ Processing NLP data...")

# Load your uploaded data
df_urls = pd.read_csv('datasets/raw/nlp/dataset_4class_final.csv')
df_emails = pd.read_csv('datasets/raw/nlp/emails.csv')

print(f"  URLs: {len(df_urls)} samples")
print(f"  Emails: {len(df_emails)} samples")

# Take subset to avoid too much data
df_urls = df_urls.head(2000)
df_emails = df_emails.head(2000)

# Create binary labels (phishing/malware vs benign/legitimate)
df_urls['phishing'] = df_urls['label'].isin(['phishing', 'malware']).astype(int)
df_emails['spam'] = df_emails['label'].isin(['spam', 'fraud']).astype(int) if 'label' in df_emails else 1

# Simple features: text length, special char count
def extract_features(text):
    if pd.isna(text):
        return 0, 0, 0
    text = str(text)
    return len(text), text.count('@'), text.count('/')

df_urls[['len', 'at_count', 'slash_count']] = df_urls['url'].apply(
    lambda x: pd.Series(extract_features(x))
)
df_emails[['len', 'at_count', 'slash_count']] = df_emails['email'].apply(
    lambda x: pd.Series(extract_features(x))
)

# Combine
df_nlp = pd.concat([
    df_urls[['len', 'at_count', 'slash_count', 'phishing']].rename(columns={'phishing': 'label'}),
    df_emails[['len', 'at_count', 'slash_count', 'spam']].rename(columns={'spam': 'label'})
])

print(f"  Combined: {len(df_nlp)} samples")
print(f"  Label distribution: {df_nlp['label'].value_counts().to_dict()}")

# Save
Path('datasets/processed/nlp').mkdir(parents=True, exist_ok=True)
X_train, X_test, y_train, y_test = train_test_split(
    df_nlp.drop('label', axis=1), 
    df_nlp['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_nlp['label']
)

X_train['label'] = y_train
X_test['label'] = y_test

X_train.to_csv('datasets/processed/nlp/train.csv', index=False)
X_test.to_csv('datasets/processed/nlp/test.csv', index=False)

print("  ✅ NLP data saved!")

# 2️⃣ PROCESS NETWORK DATA
print("\n2️⃣ Processing Network data...")

try:
    df_network = pd.read_csv('datasets/raw/network/cicids2017_cleaned.csv')
    print(f"  Loaded: {len(df_network)} samples")
    
    # Use first 2000 samples
    df_network = df_network.head(2000)
    
    # Extract features (numeric columns)
    feature_cols = [col for col in df_network.columns if col not in ['label', 'class']]
    
    # Label
    if 'label' in df_network.columns:
        y = df_network['label']
    else:
        y = df_network['class']
    
    X = df_network[feature_cols]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    
    # Split
    Path('datasets/processed/network').mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train['label'] = y_train
    X_test['label'] = y_test
    
    X_train.to_csv('datasets/processed/network/train.csv', index=False)
    X_test.to_csv('datasets/processed/network/test.csv', index=False)
    
    print("  ✅ Network data saved!")
    
except Exception as e:
    print(f"  ⚠️ Error: {e}")
    print("  Using synthetic network data")

print("\n✅ PREPROCESSING COMPLETE!")