import torch
import numpy as np
import cv2
import io
import base64
from PIL import Image
import torchvision.transforms as T
from loguru import logger


class GradCAMService:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        try:
            target_layer = self.model.model.features[-1]
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            )
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            )
            logger.info("GradCAM hooks registered on MobileNetV3 features[-1]")
        except Exception as e:
            logger.warning(f"GradCAM hook registration failed: {e}")

    def generate(self, image_bytes: bytes) -> dict:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0).requires_grad_(True)

        self.model.eval()
        output = self.model(tensor)
        pred_class = int(output.argmax(1).item())
        confidence = float(torch.softmax(output, dim=1)[0, pred_class].item())

        self.model.zero_grad()
        output[0, pred_class].backward()

        if self.gradients is None or self.activations is None:
            return {
                "error": "GradCAM hooks not available",
                "pred_class": pred_class,
                "confidence": confidence,
            }

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1).squeeze()).numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        original = np.array(img.resize((224, 224)))
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = (
            0.6 * original + 0.4 * cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        ).astype(np.uint8)

        _, buf_orig = cv2.imencode(".jpg", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        _, buf_overlay = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        regions = []
        if cam.max() > 0:
            ys, xs = np.where(cam > 0.7)
            if len(ys) > 0:
                regions.append({
                    "x_center": int(xs.mean()),
                    "y_center": int(ys.mean()),
                    "area_pct": round(float((cam > 0.7).mean()) * 100, 1),
                })

        return {
            "explanation_type": "GradCAM",
            "module": "vision",
            "pred_class": ["LEGITIMATE", "PHISHING_PAGE"][pred_class],
            "confidence": round(confidence, 4),
            "original_image_b64": base64.b64encode(buf_orig).decode(),
            "overlay_image_b64": base64.b64encode(buf_overlay).decode(),
            "high_attention_regions": regions,
            "interpretation": (
                "Suspicious visual patterns detected in highlighted regions."
                if pred_class == 1
                else "No suspicious visual patterns detected."
            ),
        }