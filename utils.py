# utils.py
import torch
import numpy as np
from skimage import color

# =========================================================
# 1. RGB <-> LAB
# =========================================================

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (H, W, 3), uint8 [0,255]
    return: (3, H, W), float32
    """
    rgb = rgb.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    return lab.transpose(2, 0, 1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    lab: (3, H, W)
    return: (H, W, 3), uint8
    """
    lab = lab.transpose(1, 2, 0)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def split_lab(lab: np.ndarray):
    """
    lab: (3, H, W)
    return:
      L  (1, H, W)
      ab (2, H, W)
    """
    return lab[0:1], lab[1:3]


# =========================================================
# 2. User hint generation (논문: sparse user clicks)
# =========================================================

def generate_hints(ab: np.ndarray, num_hints: int):
    """
    ab: (2, H, W)
    return:
      ab_hint (2, H, W)
      mask    (1, H, W)
    """
    _, H, W = ab.shape

    ab_hint = np.zeros_like(ab)
    mask = np.zeros((1, H, W), dtype=np.float32)

    if num_hints <= 0:
        return ab_hint, mask

    ys = np.random.randint(0, H, size=num_hints)
    xs = np.random.randint(0, W, size=num_hints)

    for y, x in zip(ys, xs):
        ab_hint[:, y, x] = ab[:, y, x]
        mask[0, y, x] = 1.0

    return ab_hint, mask


# =========================================================
# 3. In-gamut ab bins (논문: 313 bins)
# =========================================================

def build_in_gamut_bins(step: int = 10):
    """
    Build in-gamut ab bins exactly as SIGGRAPH 2016 paper
    return: (313, 2)
    """
    bins = []
    for a in range(-110, 111, step):
        for b in range(-110, 111, step):
            lab = np.array([50, a, b], dtype=np.float32)[None, None, :]
            rgb = color.lab2rgb(lab)
            if np.all(rgb >= 0) and np.all(rgb <= 1):
                bins.append([a, b])
    return np.array(bins, dtype=np.float32)


# 논문 기준 313 bins
AB_BINS = build_in_gamut_bins(step=10)[:313]



# =========================================================
# 4. ab → bin index (classification GT)
# =========================================================

def ab_to_bin(ab: np.ndarray) -> np.ndarray:
    """
    ab: (B, 2, H, W) or (2, H, W)
    return: (B, H, W) or (H, W)
    """
    if ab.ndim == 3:
        ab = ab[None]  # (1,2,H,W)

    B, _, H, W = ab.shape
    pts = ab.transpose(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)

    dists = ((pts[:, None, :] - AB_BINS[None]) ** 2).sum(axis=2)
    idx = np.argmin(dists, axis=1)

    return idx.reshape(B, H, W)


# =========================================================
# 5. Annealed Mean (논문 inference 핵심)
# =========================================================

def annealed_mean(logits: torch.Tensor, T: float = 0.38):
    """
    logits: (B, 313, H, W)
    return: (B, 2, H, W)
    """
    # 1. softmax
    prob = torch.softmax(logits, dim=1)

    # 2. annealing (논문 정의)
    prob = prob.pow(1.0 / T)
    prob = prob / prob.sum(dim=1, keepdim=True)

    # 3. expected value
    ab_bins = torch.tensor(AB_BINS, device=logits.device)
    ab = torch.einsum("bchw,cd->bdhw", prob, ab_bins)

    return ab


# =========================================================
# 6. Class rebalancing weights (논문 Eq. 4)
# =========================================================

def compute_rebalance_weights(prior: np.ndarray, lam: float = 0.5):
    """
    prior: (313,) empirical distribution
    return: (313,) weights
    """
    prior = prior / np.sum(prior)
    uniform = np.ones_like(prior) / len(prior)

    weights = 1.0 / ((1 - lam) * prior + lam * uniform)
    return weights.astype(np.float32)


# =========================================================
# 7. Torch helpers
# =========================================================

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
