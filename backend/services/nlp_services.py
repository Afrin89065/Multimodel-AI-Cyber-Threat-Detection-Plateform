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
from utils.metrics import (
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    timer,
)

LABELS = ["CLEAN", "SPAM", "PHISHING", "BEC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
SECUREBERT_MODEL = "ehsanaghaei/SecureBERT"  # v3 NEW


class NLPService:
    """
    NLP Service supporting both:
      1. Full SecureBERT (.pt)
      2. Lite sklearn (.pkl)
    """

    def __init__(self, model_path: str = "", lite_model_path: Optional[str] = None):

        self.mode = "uninitialised"
        self.lite_model = None
        self.tokenizer = None
        self.model = None

        # ---------------------------------------------------------
        # FULL MODEL (.pt)
        # ---------------------------------------------------------
        if model_path and Path(model_path).is_file():

            self.tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)

            self.model = NLPClassifier()

            state = torch.load(model_path, map_location="cpu")

            if isinstance(state, dict):
                self.model.load_state_dict(
                    {
                        k: (
                            v.float()
                            if isinstance(v, torch.Tensor)
                            and v.dtype == torch.float16
                            else v
                        )
                        for k, v in state.items()
                    },
                    strict=False,
                )

            self.model.eval()

            self.mode = "full"

            logger.info(f"✅ NLP full model loaded: {model_path}")

        # ---------------------------------------------------------
        # LITE MODEL (.pkl)
        # ---------------------------------------------------------
        elif lite_model_path and Path(lite_model_path).is_file():

            with open(lite_model_path, "rb") as f:
                self.lite_model = pickle.load(f)

            self.mode = "lite"

            logger.warning(
                f"⚠ Full NLP model not found.\n"
                f"Running with LITE model:\n{lite_model_path}"
            )

        # ---------------------------------------------------------
        # FALLBACK
        # ---------------------------------------------------------
        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                SECUREBERT_MODEL
            )

            self.model = NLPClassifier()

            self.model.eval()

            self.mode = "full_untrained"

            logger.warning(
                "⚠ No NLP checkpoint found. "
                "Using randomly initialized SecureBERT."
            )

    # ---------------------------------------------------------
    # Encode
    # ---------------------------------------------------------
    def _encode(self, text: str, url: str):

        enc = self.tokenizer(
            text[:1000],
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        url_feat = torch.tensor(
            extract_url_features(url),
            dtype=torch.float32,
        ).unsqueeze(0)

        return (
            enc["input_ids"],
            enc["attention_mask"],
            url_feat,
        )

    # ---------------------------------------------------------
    # Lite prediction
    # ---------------------------------------------------------
    def _analyse_lite(self, url: str):

        feats = extract_url_features(url).reshape(1, -1)

        probs = self.lite_model.predict_proba(feats)[0]

        phishing = float(probs[-1])

        class_probs = {l: 0.0 for l in LABELS}
        class_probs["CLEAN"] = round(1 - phishing, 4)
        class_probs["PHISHING"] = round(phishing, 4)

        return {
            "nlp_score": round(phishing, 4),
            "threat_type": "PHISHING" if phishing >= 0.5 else "CLEAN",
            "confidence": round(max(phishing, 1 - phishing), 4),
            "uncertainty": None,
            "is_uncertain": None,
            "class_probs": class_probs,
            "model_mode": "lite",
            "url_features": {
                k: round(float(v), 3)
                for k, v in zip(
                    URL_FEATURE_NAMES,
                    extract_url_features(url),
                )
            },
        }

    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------
    @torch.no_grad()
    def analyse(
        self,
        text: str,
        url: str = "",
        use_uncertainty: bool = True,
    ):

        with timer("nlp"):

            if self.mode == "lite":

                result = self._analyse_lite(url)

                MODEL_CONFIDENCE.labels(
                    module="nlp"
                ).set(result["confidence"])

                return result

            ids, mask, url_feat = self._encode(text, url)

            if use_uncertainty:

                probs, std = self.model.forward_mc(
                    ids,
                    mask,
                    url_feat,
                    n_samples=15,
                )

                probs = probs.squeeze().numpy()

                uncertainty = float(std.squeeze().numpy().max())

            else:

                logits = self.model(ids, mask, url_feat)

                probs = torch.softmax(
                    logits,
                    dim=1,
                ).squeeze().numpy()

                uncertainty = 0.0

            idx = int(np.argmax(probs))

            confidence = float(probs[idx])

            MODEL_CONFIDENCE.labels(
                module="nlp"
            ).set(confidence)

            return {
                "nlp_score": round(float(1 - probs[0]), 4),
                "threat_type": LABELS[idx],
                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty, 4),
                "is_uncertain": uncertainty > 0.15,
                "class_probs": {
                    LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "model_mode": self.mode,
                "url_features": {
                    k: round(float(v), 3)
                    for k, v in zip(
                        URL_FEATURE_NAMES,
                        extract_url_features(url),
                    )
                },
            }