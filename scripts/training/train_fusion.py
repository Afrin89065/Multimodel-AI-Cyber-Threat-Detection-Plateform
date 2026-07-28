import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from loguru import logger
import argparse

# v3: AttentionFusionEngine (NOT FusionMLP)
class AttentionFusionEngine(nn.Module):
    def __init__(self, n_modules=4, feature_dim=3, hidden_dim=64, n_classes=5):
        super().__init__()
        self.module_embedding = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * n_modules, n_classes)
    
    def forward(self, x):
        emb = self.module_embedding(x)
        attended, attn_weights = self.attention(emb, emb, emb)
        return self.classifier(attended.reshape(attended.size(0), -1)), attn_weights

class FusionTrainer:
    def __init__(self, model_path, epochs=10, lr=1e-3):
        self.model = AttentionFusionEngine()
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.model_path = model_path
    
    def train(self, nlp_scores, vision_scores, network_scores, malware_scores, labels):
        """Train fusion model"""
        logger.info("Training AttentionFusionEngine...")
        
        # Stack module scores [B, 4, 3]
        X = np.stack([
            np.column_stack([nlp_scores, np.ones_like(nlp_scores), nlp_scores > 0.5]),
            np.column_stack([vision_scores, np.ones_like(vision_scores), vision_scores > 0.5]),
            np.column_stack([network_scores, np.ones_like(network_scores), network_scores > 0.5]),
            np.column_stack([malware_scores, np.ones_like(malware_scores), malware_scores > 0.5])
        ], axis=1)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits, attn = self.model(X_batch)
                loss = criterion(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        self.save()
    
    def save(self):
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Fusion model saved to {self.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="backend/models/fusion_model.pt")
    args = parser.parse_args()
    
    trainer = FusionTrainer(args.output, args.epochs, args.lr)
    
    # Sample data
    nlp_scores = np.random.rand(100)
    vision_scores = np.random.rand(100)
    network_scores = np.random.rand(100)
    malware_scores = np.random.rand(100)
    labels = np.random.randint(0, 5, 100)
    
    trainer.train(nlp_scores, vision_scores, network_scores, malware_scores, labels)