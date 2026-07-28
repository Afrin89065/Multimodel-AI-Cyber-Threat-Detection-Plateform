import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from loguru import logger
import argparse

class NetworkPreprocessor:
    def __init__(self, output_dir="data/network"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess(self, X, y):
        """Preprocess network traffic features"""
        logger.info(f"Preprocessing network data: {X.shape}")
        
        # Handle missing values
        X = np.nan_to_num(X)
        
        # Remove outliers (IQR method)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        mask = ~np.any((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR)), axis=1)
        X, y = X[mask], y[mask]
        
        logger.info(f"After outlier removal: {X.shape}")
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info("✅ Preprocessing complete")
        return X_scaled, y, scaler

if __name__ == "__main__":
    processor = NetworkPreprocessor()
    
    # Sample data
    X = np.random.randn(1000, 80)
    y = np.random.randint(0, 4, 1000)
    
    X_processed, y_processed, scaler = processor.preprocess(X, y)
    logger.info(f"Processed shape: {X_processed.shape}")