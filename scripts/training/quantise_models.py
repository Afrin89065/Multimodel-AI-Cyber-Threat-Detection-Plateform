import torch
import pickle
from pathlib import Path
from loguru import logger

class ModelQuantizer:
    def __init__(self, output_dir="backend/models"):
        self.output_dir = Path(output_dir)
    
    def quantize_nlp(self, model_path):
        """Quantize NLP model"""
        logger.info("Quantizing NLP model...")
        model = torch.load(model_path, map_location="cpu")
        
        # Convert to int8
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        output_path = self.output_dir / "nlp_model_quantized.pt"
        torch.save(quantized.state_dict(), output_path)
        logger.info(f"Quantized model saved: {output_path}")
    
    def quantize_all(self):
        """Quantize all models"""
        logger.info("Quantizing all models...")
        
        models = [
            "nlp_model.pt",
            "network_model.pt",
            "fusion_model.pt"
        ]
        
        for model_name in models:
            model_path = self.output_dir / model_name
            if model_path.exists():
                self.quantize_nlp(model_path)
            else:
                logger.warning(f"Model not found: {model_path}")
        
        logger.info("✅ Quantization complete")

if __name__ == "__main__":
    quantizer = ModelQuantizer()
    quantizer.quantize_all()