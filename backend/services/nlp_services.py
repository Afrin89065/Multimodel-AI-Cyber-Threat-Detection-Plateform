import torch
import torch.nn as nn
import numpy as np
import tldextract
import math
import re
import hashlib
import pickle
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from loguru import logger
from typing import Optional
from utils.metrics import INFERENCE_LATENCY, MODEL_CONFIDENCE, timer

LABELS = ["CLEAN", "SPAM", "PHISHING", "BEC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
SECUREBERT_MODEL = "ehsanaghaei/SecureBERT"  # v3 NEW


class NLPClassifier(nn.Module):
    def __init__(self, num_url_features: int = 25, dropout: float = 0.3):
        super().__init__()
        # v3 CHANGE: SecureBERT instead of DistilBERT
        self.bert = AutoModel.from_pretrained(SECUREBERT_MODEL)
        # Freeze first 4 transformer layers to speed up CPU inference
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False

        self.url_encoder = nn.Sequential(
            nn.Linear(num_url_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, len(LABELS))
        )

    def forward(self, input_ids, attention_mask, url_features):
        cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        url_enc = self.url_encoder(url_features)
        combined = torch.cat([cls, url_enc], dim=1)
        return self.classifier(combined)

    def forward_mc(self, input_ids, attention_mask, url_features, n_samples: int = 10):
        """MC-Dropout: run forward pass n times with dropout active."""
        self.train()  # enable dropout
        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(input_ids, attention_mask, url_features)
                outputs.append(torch.softmax(out, dim=1))
        self.eval()
        stacked = torch.stack(outputs)
        mean_probs = stacked.mean(dim=0)
        std_probs = stacked.std(dim=0)
        return mean_probs, std_probs


URL_FEATURE_NAMES = [
    "url_length", "domain_length", "dot_count", "dash_count",
    "digit_count_domain", "domain_entropy", "risky_tld", "ip_in_url",
    "at_symbol", "double_slash", "multi_subdomain", "login_kw",
    "verify_kw", "account_kw", "secure_kw", "update_kw",
    "shortener", "percent_count", "equals_count", "question_count",
    "is_https", "subdomain_depth", "underscore_count",
    "non_ascii_domain", "url_length_norm"
]

RISKY_TLDS = {".xyz", ".top", ".tk", ".cc", ".pw", ".ml", ".ga", ".cf", ".gq", ".su"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly", "buff.ly"}


def extract_url_features(url: str) -> np.ndarray:
    if not url:
        return np.zeros(25, dtype=np.float32)
    ext = tldextract.extract(url)
    domain = ext.domain or ""

    def entropy(s):
        if not s:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    return np.array([
        min(len(url) / 200, 1.0),
        min(len(domain) / 50, 1.0),
        min(url.count(".") / 10, 1.0),
        min(url.count("-") / 10, 1.0),
        min(sum(c.isdigit() for c in domain) / 10, 1.0),
        entropy(domain) / 5.0,
        float("." + ext.suffix in RISKY_TLDS),
        float(bool(re.search(r"\d{1,3}(\.\d{1,3}){3}", url))),
        float("@" in url),
        float("//" in url[8:]),
        float(ext.subdomain.count(".") > 1) if ext.subdomain else 0.0,
        min(url.lower().count("login") / 3, 1.0),
        min(url.lower().count("verify") / 3, 1.0),
        min(url.lower().count("account") / 3, 1.0),
        min(url.lower().count("secure") / 3, 1.0),
        min(url.lower().count("update") / 3, 1.0),
        float(any(s in url.lower() for s in SHORTENERS)),
        min(url.count("%") / 10, 1.0),
        min(url.count("=") / 5, 1.0),
        min(url.count("?") / 3, 1.0),
        float(url.lower().startswith("https")),
        min(len(ext.subdomain.split(".")) / 5, 1.0) if ext.subdomain else 0.0,
        min(url.count("_") / 5, 1.0),
        float(domain != domain.encode("ascii", "ignore").decode()),
        min(len(url) / 200, 1.0),
    ], dtype=np.float32)


class NLPService:
    """
    v3 FIX: two operating modes.

    "full" mode: the original SecureBERT-based NLPClassifier, loaded from a
    .pt state_dict at `model_path`. This is what should run in production.

    "lite" mode: falls back to a plain sklearn classifier operating on the
    same 25-dim extract_url_features() vector this module already computes,
    used automatically when `model_path` doesn't exist and `lite_model_path`
    does. It has no text-semantics understanding at all (URL structure
    signals only) and is meant to keep the pipeline functional — not as a
    long-term substitute for the transformer model. Every response makes
    the active mode explicit via "model_mode" so this is never silently
    mistaken for the real thing downstream (dashboard, fusion, audit log).

    Neither of the two training scripts in scripts/training/ (train_nlp.py,
    train_nlp_simple.py) previously produced a checkpoint that actually
    matched either of these code paths — see CHANGELOG for the fixes to
    both.
    """

    def __init__(self, model_path: str, lite_model_path: Optional[str] = None):
        self.mode = "uninitialised"
        self.lite_model = None
        self.tokenizer = None
        self.model = None

        if Path(model_path).exists():
            # v3 CHANGE: AutoTokenizer with SecureBERT
            self.tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
            self.model = NLPClassifier()
            state = torch.load(model_path, map_location="cpu")
            # Handle both fp16 and fp32 checkpoints
            self.model.load_state_dict(
                {k: v.float() if v.dtype == torch.float16 else v
                 for k, v in state.items()}
            )
            self.model.eval()
            self.mode = "full"
            logger.info(f"NLP model loaded (full/SecureBERT): {model_path}")
        elif lite_model_path and Path(lite_model_path).exists():
            with open(lite_model_path, "rb") as f:
                self.lite_model = pickle.load(f)
            self.mode = "lite"
            logger.warning(
                f"NLP full model not found at {model_path}. Running in LITE "
                f"mode from {lite_model_path} — URL-structure heuristics "
                f"only, no email/text semantic understanding. Train and "
                f"deploy the SecureBERT checkpoint (scripts/training/"
                f"train_nlp.py) for production use."
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
            self.model = NLPClassifier()
            self.model.eval()
            self.mode = "full_untrained"
            logger.warning(
                f"NLP model not found at {model_path} and no lite_model_path "
                f"given — using randomly-initialised weights. Predictions "
                f"are meaningless until a real checkpoint is trained."
            )

    def _encode(self, text: str, url: str):
        enc = self.tokenizer(
            text[:1000], max_length=256,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        url_feat = torch.tensor(
            extract_url_features(url), dtype=torch.float32
        ).unsqueeze(0)
        return enc["input_ids"], enc["attention_mask"], url_feat

    def _analyse_lite(self, url: str) -> dict:
        feats = extract_url_features(url).reshape(1, -1)
        probs_bin = self.lite_model.predict_proba(feats)[0]  # [P(benign), P(phishing)]
        phishing_p = float(probs_bin[-1])
        idx = 1 if phishing_p >= 0.5 else 0
        # Map the lite model's binary output onto the full 4-class label
        # space so downstream consumers (fusion, dashboard) see a consistent
        # schema regardless of which mode produced it.
        class_probs = {l: 0.0 for l in LABELS}
        class_probs["CLEAN"] = round(1.0 - phishing_p, 4)
        class_probs["PHISHING"] = round(phishing_p, 4)
        return {
            "nlp_score": round(phishing_p, 4),
            "threat_type": "PHISHING" if idx == 1 else "CLEAN",
            "confidence": round(max(phishing_p, 1 - phishing_p), 4),
            "uncertainty": None,
            "is_uncertain": None,
            "class_probs": class_probs,
            "url_features": {
                name: round(float(v), 3)
                for name, v in zip(URL_FEATURE_NAMES, extract_url_features(url))
            },
            "model_mode": "lite",
        }

    @torch.no_grad()
    def analyse(self, text: str, url: str = "", use_uncertainty: bool = True) -> dict:
        with timer("nlp"):
            if self.mode == "lite":
                result = self._analyse_lite(url)
                MODEL_CONFIDENCE.labels(module="nlp").set(result["confidence"])
                return result

            ids, mask, url_feat = self._encode(text, url)

            if use_uncertainty:
                mean_probs, std_probs = self.model.forward_mc(
                    ids, mask, url_feat, n_samples=15
                )
                probs = mean_probs.squeeze().numpy()
                uncertainty = float(std_probs.squeeze().numpy().max())
            else:
                logits = self.model(ids, mask, url_feat)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                uncertainty = 0.0

            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            MODEL_CONFIDENCE.labels(module="nlp").set(confidence)

            return {
                "nlp_score": round(float(1.0 - probs[0]), 4),
                "threat_type": LABELS[idx],
                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty, 4),
                "is_uncertain": uncertainty > 0.15,
                "class_probs": {
                    LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "url_features": {
                    name: round(float(v), 3)
                    for name, v in zip(
                        URL_FEATURE_NAMES,
                        extract_url_features(url)
                    )
                },
                "model_mode": self.mode,
            }