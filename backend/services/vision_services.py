import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
import imagehash
import pickle
import io
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from loguru import logger
from utils.metrics import timer

# v3 NEW: CLIP imports
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("open-clip-torch not installed")

SPOOF_CLASSES = ["LEGITIMATE", "PHISHING_PAGE"]
BRAND_CLASSES = ["amazon", "facebook", "google", "microsoft", "paypal"]


# v3 NEW: CLIPVisionService class
class CLIPVisionService:
    BRANDS = {
        "paypal": "PayPal blue P logo",
        "microsoft": "Microsoft four colored squares logo",
        "google": "Google colorful G logo",
        "amazon": "Amazon smile arrow logo",
        "facebook": "Facebook lowercase blue f logo",
        "apple": "Apple bitten apple logo",
    }
    PHISHING_PROMPTS = [
        "a legitimate company login page",
        "a phishing website impersonating a bank to steal credentials",
        "a fake PayPal login page designed for fraud",
        "a real corporate website homepage",
        "a spoofed Microsoft login page",
    ]

    def __init__(self):
        if not CLIP_AVAILABLE:
            self.available = False
            return
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model.eval()
            self.available = True
            logger.info("CLIP ViT-B/32 ready")
        except Exception as e:
            self.available = False
            logger.warning(f"CLIP init failed: {e}")

    @torch.no_grad()
    def classify_screenshot(self, image_bytes: bytes) -> dict:
        if not self.available:
            return {"clip_cv_score": 0.0, "clip_pred": "UNKNOWN"}
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        texts = self.tokenizer(self.PHISHING_PROMPTS)
        img_f = self.model.encode_image(tensor)
        txt_f = self.model.encode_text(texts)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt_f.T).squeeze().softmax(dim=-1)
        phishing_score = float(sims[1] + sims[2] + sims[4])
        return {
            "clip_cv_score": round(phishing_score, 4),
            "clip_pred": "PHISHING_PAGE" if phishing_score > 0.45 else "LEGITIMATE",
        }

    @torch.no_grad()
    def detect_brand(self, image_bytes: bytes) -> dict:
        if not self.available:
            return {"detected_brand": None, "brand_confidence": 0.0}
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        texts = self.tokenizer(list(self.BRANDS.values()))
        img_f = self.model.encode_image(tensor)
        txt_f = self.model.encode_text(texts)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt_f.T).squeeze()
        brands = list(self.BRANDS.keys())
        top_idx = int(sims.argmax())
        top_score = float(sims[top_idx])
        return {
            "detected_brand": brands[top_idx] if top_score > 0.25 else None,
            "brand_confidence": round(top_score, 4),
        }

    @torch.no_grad()
    def check_consistency(self, email_text: str, image_bytes: bytes) -> dict:
        """v3: cross-modal consistency check."""
        if not self.available:
            return {"consistency_score": 0.5, "is_inconsistent": False}
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        text_token = self.tokenizer([email_text[:77]])
        img_f = self.model.encode_image(tensor)
        txt_f = self.model.encode_text(text_token)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        sim = float((img_f @ txt_f.T).item())
        consistency = (sim + 1) / 2
        return {
            "consistency_score": round(consistency, 4),
            "inconsistency_score": round(1 - consistency, 4),
            "is_inconsistent": consistency < 0.35,
        }


class SpoofDetector(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

    def forward_mc(self, x, n_samples: int = 10):
        self.train()
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                p = torch.softmax(self.model(x), dim=1)
                probs.append(p)
        self.eval()
        stacked = torch.stack(probs)
        return stacked.mean(0), stacked.std(0)


class VisionService:
    TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, clf_path: str, yolo_path: str, phash_db_path: str = ""):
        self.clf = SpoofDetector()
        if Path(clf_path).exists():
            state = torch.load(clf_path, map_location="cpu")
            self.clf.load_state_dict(
                {k: v.float() if v.dtype == torch.float16 else v
                 for k, v in state.items()}
            )
        self.clf.eval()

        self.yolo = (
            YOLO(yolo_path) if Path(yolo_path).exists()
            else YOLO("yolov8n.pt")
        )

        self.phash_db: dict = {}
        if phash_db_path and Path(phash_db_path).exists():
            self.phash_db = pickle.load(open(phash_db_path, "rb"))
            logger.info(f"pHash DB loaded: {len(self.phash_db)} brands")

        # v3 NEW: Initialize CLIP service
        self.clip = CLIPVisionService()

        logger.info("Vision Service ready")

    def load_phash_db(self, logo_dir: str):
        for brand in BRAND_CLASSES:
            d = Path(logo_dir) / brand
            if d.exists():
                hashes = []
                for p in d.glob("*.png"):
                    try:
                        hashes.append(
                            imagehash.phash(Image.open(p).convert("RGB"))
                        )
                    except Exception:
                        continue
                if hashes:
                    self.phash_db[brand] = hashes
        logger.info(f"pHash DB built: {len(self.phash_db)} brands")

    def analyse(self, image_bytes: bytes) -> dict:
        with timer("vision"):
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.TRANSFORM(img).unsqueeze(0)

            # MobileNetV3 with MC-Dropout
            mean_p, std_p = self.clf.forward_mc(tensor, n_samples=15)
            probs = mean_p.squeeze().numpy()
            uncertainty = float(std_p.squeeze().numpy().max())
            vit_score = float(probs[1])

            # YOLOv8n brand logo detection
            detected_brands = []
            try:
                results = self.yolo(np.array(img), verbose=False, conf=0.4)
                detected_brands = [
                    {
                        "brand": BRAND_CLASSES[int(b.cls[0])],
                        "confidence": round(float(b.conf[0]), 3)
                    }
                    for b in results[0].boxes
                    if int(b.cls[0]) < len(BRAND_CLASSES)
                ]
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")

            # pHash brand spoofing check
            ph_score, ph_brand = self._phash_check(img)
            brand_mismatch = (
                ph_brand is not None
                and all(b["brand"] != ph_brand for b in detected_brands)
            )

            # v3 NEW: CLIP results
            clip_result = self.clip.classify_screenshot(image_bytes)
            clip_brand = self.clip.detect_brand(image_bytes)

            # Composite score
            cv_score = min(
                (0.45 * vit_score)
                + (0.30 * float(brand_mismatch))
                + (0.25 * ph_score),
                1.0
            )

            return {
                "cv_score": round(cv_score, 4),
                "vit_score": round(vit_score, 4),
                "uncertainty": round(uncertainty, 4),
                "pred_class": SPOOF_CLASSES[int(np.argmax(probs))],
                "class_probs": {
                    SPOOF_CLASSES[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "detected_brands": detected_brands,
                "brand_mismatch": brand_mismatch,
                "phash_brand": ph_brand,
                "phash_score": round(ph_score, 4),
                # v3 NEW: CLIP outputs
                "clip_cv_score": clip_result.get("clip_cv_score", 0.0),
                "clip_pred": clip_result.get("clip_pred", "UNKNOWN"),
                "clip_brand": clip_brand,
            }

    def _phash_check(self, img: Image.Image):
        if not self.phash_db:
            return 0.0, None
        h = imagehash.phash(img)
        best_dist, best_brand = 999, None
        for brand, hashes in self.phash_db.items():
            for ref_hash in hashes:
                d = h - ref_hash
                if d < best_dist:
                    best_dist, best_brand = d, brand
        score = max(0.0, (30 - best_dist) / 30)
        return score, (best_brand if best_dist < 10 else None)