import numpy as np
from pathlib import Path
from loguru import logger

class NLPDataAugmenter:
    def __init__(self, output_dir="data/nlp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def augment(self, texts, labels, multiplier=2):
        """Augment NLP training data"""
        logger.info(f"Augmenting {len(texts)} samples x{multiplier}")
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        augmentations = [
            self._lowercase,
            self._remove_punctuation,
            self._character_swap,
            self._synonym_replace
        ]
        
        for _ in range(multiplier - 1):
            for text, label in zip(texts, labels):
                augmented_text = np.random.choice(augmentations)(text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
        
        logger.info(f"✅ Augmented to {len(augmented_texts)} samples")
        return augmented_texts, augmented_labels
    
    @staticmethod
    def _lowercase(text):
        return text.lower()
    
    @staticmethod
    def _remove_punctuation(text):
        return text.replace(".", "").replace(",", "").replace("!", "")
    
    @staticmethod
    def _character_swap(text):
        chars = list(text)
        i, j = np.random.choice(len(chars), 2, replace=False)
        chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)
    
    @staticmethod
    def _synonym_replace(text):
        replacements = {"paypal": "pay-pal", "verify": "validate", "account": "profile"}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

if __name__ == "__main__":
    augmenter = NLPDataAugmenter()
    
    texts = ["Click to verify PayPal", "Validate your account"]
    labels = [1, 0]
    
    augmented_texts, augmented_labels = augmenter.augment(texts, labels, multiplier=3)
    logger.info(f"Original: {len(texts)}, Augmented: {len(augmented_texts)}")