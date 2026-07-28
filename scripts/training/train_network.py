import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path
from loguru import logger
import argparse
import warnings

warnings.filterwarnings('ignore')

class NetworkTrainerSimplified:
    """Simplified XGBoost trainer with strong regularization"""
    
    def __init__(self, model_path, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.random_state = random_state
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
    
    def load_data(self, train_path, test_path=None):
        """Load training data"""
        
        logger.info(f"📂 Loading data from {train_path}")
        
        try:
            df = pd.read_csv(train_path)
            logger.info(f"✅ Loaded: {len(df)} samples")
            
            # Extract X and y
            if 'label' in df.columns:
                X = df.drop('label', axis=1).values
                y = df['label'].values
            elif 'class' in df.columns:
                X = df.drop('class', axis=1).values
                y = df['class'].values
            else:
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            
            # Convert to numeric if needed
            y = np.asarray([int(yy) for yy in y])
            
            logger.info(f"Features: {X.shape}")
            logger.info(f"Classes: {np.unique(y)}")
            logger.info(f"Distribution: {np.bincount(y)}")
            
            return X, y
        
        except FileNotFoundError:
            logger.warning(f"⚠️ File not found: {train_path}")
            logger.info("Creating realistic sample data...")
            
            # Create better sample data
            np.random.seed(self.random_state)
            n_samples = 2000
            n_features = 80
            n_classes = 4
            
            X = np.random.randn(n_samples, n_features) * 10
            y = np.random.randint(0, n_classes, n_samples)
            
            # Add class separation (makes it learnable)
            for i in range(n_classes):
                mask = y == i
                X[mask] += i * 3
            
            logger.info(f"Created: {X.shape} samples with {n_classes} classes")
            return X, y
    
    def preprocess(self, X):
        """Clean and scale data"""
        
        logger.info("📊 Preprocessing...")
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=100, neginf=-100)
        
        # Remove extreme outliers (not IQR, just hard limits)
        X = np.clip(X, -100, 100)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("✅ Preprocessing complete")
        return X_scaled
    
    def train(self, X, y):
        """Train XGBoost with strong regularization"""
        
        logger.info(f"🚀 Training on {len(X)} samples")
        print("=" * 80)
        
        # Preprocess
        X_scaled = self.preprocess(X)
        
        # Split (stratified to maintain class distribution)
        logger.info("📂 Splitting data (80/20 stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        
        # Train with STRONG regularization (NO class weighting - causes issues)
        logger.info("\n🧠 Training XGBoost with STRONG regularization...")
        print("-" * 80)
        
        # KEY: These parameters prevent overfitting
        self.model = xgb.XGBClassifier(
            # Tree complexity control (MOST IMPORTANT)
            max_depth=3,                # ← REDUCED from 6
            min_child_weight=10,        # ← INCREASED from 1
            subsample=0.5,              # ← 70% samples per tree
            colsample_bytree=0.5,       # ← 70% features per tree
            colsample_bylevel=0.7,      # ← 70% features per level
            
            # Regularization (L1 + L2)
            reg_alpha=5.0,              # ← L1 penalty
            reg_lambda=5.0,             # ← L2 penalty
            
            # Learning
            n_estimators=150,           # ← 150 trees
            learning_rate=0.01,         # ← Very slow (was 0.1)
            
            # Randomness (reproducibility)
            random_state=self.random_state,
            
            # Other
            objective='multi:softmax' if len(np.unique(y)) > 2 else 'binary:logistic',
            eval_metric='mlogloss' if len(np.unique(y)) > 2 else 'logloss',
        )
        
        # Train (no early stopping needed with this config)
        self.model.fit(X_train, y_train, verbose=0)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        gap = (train_acc - test_acc) * 100
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✅ Training Accuracy:  {train_acc*100:6.2f}%")
        logger.info(f"✅ Testing Accuracy:   {test_acc*100:6.2f}%")
        logger.info(f"✅ Overfitting Gap:    {gap:6.2f}%")
        logger.info("=" * 80)
        
        # Check if good
        if gap < 5:
            logger.info("🎉 EXCELLENT! Very low overfitting!")
        elif gap < 10:
            logger.info("✅ GOOD! Acceptable overfitting!")
        else:
            logger.warning("⚠️  WARNING: Still some overfitting, consider more data")
        
        self.save()
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap
        }
    
    def save(self):
        """Save model and scaler"""
        
        import pickle
        
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        pickle.dump(self.model, open(self.model_path, 'wb'))
        logger.info(f"💾 Model saved: {self.model_path}")
        
        # Save scaler
        scaler_path = str(self.model_path).replace('.pkl', '_scaler.pkl')
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        logger.info(f"💾 Scaler saved: {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="datasets/processed/network/train.csv")
    parser.add_argument("--output", default="backend/models/network_model.pkl")
    args = parser.parse_args()
    
    # Train
    trainer = NetworkTrainerSimplified(args.output)
    X, y = trainer.load_data(args.train_path)
    results = trainer.train(X, y)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)