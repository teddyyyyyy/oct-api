import numpy as np
from PIL import Image
import io

def img_stats(img: Image.Image):
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    # 簡單銳利度 proxy：Laplacian variance
    lap = np.abs(
        -4*arr
        + np.roll(arr, 1, 0) + np.roll(arr, -1, 0)
        + np.roll(arr, 1, 1) + np.roll(arr, -1, 1)
    )
    sharp = float(lap.var())
    return np.array([mean, std, sharp], dtype=np.float32)

class DriftMonitor:
    def __init__(self, ref_stats_path: str):
        # ref: 你用 training/val 的統計先算好存起來
        try:
            ref = np.load(ref_stats_path)
            self.mu = ref["mu"]
            self.sigma = ref["sigma"]
        except Exception:
            # 沒ref就先用預設（你之後再補）
            self.mu = np.array([0.5, 0.2, 0.01], dtype=np.float32)
            self.sigma = np.array([0.1, 0.1, 0.01], dtype=np.float32)

    def check_and_update(self, img_bytes: bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = img_stats(img)
        z = np.abs((x - self.mu) / (self.sigma + 1e-6))
        score = float(z.mean())

        alert = None
        if score > 5.0:
            alert = "severe_shift"
        elif score > 3.0:
            alert = "moderate_shift"

        return {"score": score, "z": z.tolist(), "alert": alert}