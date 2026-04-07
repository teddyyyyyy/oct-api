import torch
import torch.nn as nn
import numpy as np
import timm
from torchvision import models as tv_models

from app.inference.preprocess import build_transform, bytes_to_tensor

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _unwrap_checkpoint(obj):
    """
    Accepts:
      - OrderedDict (pure state_dict)
      - dict with keys like: state_dict/model/model_state_dict/net
    Returns a state_dict (dict[str, Tensor])
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v
    return obj


def _strip_prefix(state_dict, prefix: str):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _is_torchvision_convnext(state_dict: dict) -> bool:
    # Your checkpoint keys look like: features.* and classifier.*
    for k in state_dict.keys():
        if k.startswith("features.") or k.startswith("classifier."):
            return True
    return False


def _build_torchvision_convnext(arch: str, num_classes: int):
    arch = arch.lower()
    if arch in ("convnext_tiny", "torchvision_convnext_tiny", "tv_convnext_tiny"):
        m = tv_models.convnext_tiny(weights=None)
    elif arch in ("convnext_small", "torchvision_convnext_small", "tv_convnext_small"):
        m = tv_models.convnext_small(weights=None)
    elif arch in ("convnext_base", "torchvision_convnext_base", "tv_convnext_base"):
        m = tv_models.convnext_base(weights=None)
    elif arch in ("convnext_large", "torchvision_convnext_large", "tv_convnext_large"):
        m = tv_models.convnext_large(weights=None)
    else:
        raise ValueError(
            f"Unsupported torchvision ConvNeXt arch='{arch}'. "
            "Use one of convnext_tiny/small/base/large (torchvision)."
        )

    # torchvision convnext classifier: Sequential(LayerNorm2d, Flatten, Linear)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


class OCTClassifier:
    def __init__(self, weights_path: str, arch: str, device: str = "cpu", img_size: int = 224):
        self.device = device
        self.tfm = build_transform(img_size)
        self.softmax = nn.Softmax(dim=1)

        raw = torch.load(weights_path, map_location="cpu")
        state = _unwrap_checkpoint(raw)
        state = _strip_prefix(state, "module.")  # in case trained with DDP

        # Auto-detect whether this is torchvision ConvNeXt style
        if _is_torchvision_convnext(state):
            # Your current weights are this type
            self.model = _build_torchvision_convnext(arch, num_classes=len(CLASS_NAMES))
        else:
            # timm-style
            self.model = timm.create_model(arch, pretrained=False, num_classes=len(CLASS_NAMES))

        # Load weights strictly (better to fail loudly during startup)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)

    @torch.inference_mode()
    def predict(self, img_bytes: bytes):
        try:
            x = bytes_to_tensor(img_bytes, self.tfm, self.device)
            if x is None:
                return {"error": "Invalid image format"}

            logits = self.model(x)
            probs = self.softmax(logits).detach().cpu().numpy()[0]

            idx = int(np.argmax(probs))
            label = CLASS_NAMES[idx]

            return {
                "label": label,
                "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            }
        except Exception as e:
            return {"error": str(e)}