import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from loguru import logger
import argparse

# v3: Use SecureBERT (NOT DistilBERT)
SECUREBERT_MODEL = "ehsanaghaei/SecureBERT"

class NLPTrainer:
    def __init__(self, model_path, batch_size=32, epochs=3, lr=2e-5):
        self.tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
        self.model = AutoModel.from_pretrained(SECUREBERT_MODEL)
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model_path = model_path
        
    def train(self, emails, labels):
        """Train on phishing emails"""
        logger.info(f"Training NLP on {len(emails)} samples")
        
        # Tokenize
        encodings = self.tokenizer(
            emails, 
            max_length=256,
            padding=True, 
            truncation=True,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels)
        )
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = nn.functional.mse_loss(
                    outputs.pooler_output.squeeze(),
                    batch_labels.float()
                )
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        self.save()
    
    def save(self):
        """Save model"""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", default="backend/models/nlp_model.pt")
    args = parser.parse_args()
    
    trainer = NLPTrainer(args.output, args.batch_size, args.epochs, args.lr)
    
    # Load sample data
    emails = ["Click here to verify PayPal", "Welcome to our service"]
    labels = [1.0, 0.0]  # 1=phishing, 0=legitimate
    
    trainer.train(emails, labels)