# demo.py
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import UNetColorizationNet
from dataset import ColorizationDataset
from utils import lab_to_rgb, annealed_mean


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------
    # 1. Dataset (고정 hint 비율)
    # -------------------------------------------------
    dataset = ColorizationDataset(
        train=False,
        hint_ratios=(0.03,)   # demo에서는 고정
    )

    # dataset은 이제 5개를 반환함
    L, ab_hint, mask, ab_bin, ab_gt = dataset[0]

    L = L.unsqueeze(0).to(device)
    ab_hint = ab_hint.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    # -------------------------------------------------
    # 2. Model load
    # -------------------------------------------------
    model = UNetColorizationNet(num_bins=313).to(device)
    model.load_state_dict(
        torch.load("colorization_unet_final.pth", map_location=device)
    )
    model.eval()

    # -------------------------------------------------
    # 3. Inference (논문 방식)
    # -------------------------------------------------
    with torch.no_grad():
        logits, _ = model(L, ab_hint, mask)   # (1, 313, H, W)

        # Annealed Mean (SIGGRAPH 2016)
        pred_ab = annealed_mean(logits, T=0.38)  # (1, 2, H, W)

    # -------------------------------------------------
    # 4. RGB 변환
    # -------------------------------------------------
    L_np = L[0].cpu().numpy()          # (1,H,W)
    ab_gt_np = ab_gt.numpy()           # (2,H,W)
    pred_ab_np = pred_ab[0].cpu().numpy()

    gray_lab = np.concatenate(
        [L_np, np.zeros_like(ab_gt_np)], axis=0
    )
    gt_lab = np.concatenate(
        [L_np, ab_gt_np], axis=0
    )
    pred_lab = np.concatenate(
        [L_np, pred_ab_np], axis=0
    )

    gray_rgb = lab_to_rgb(gray_lab)
    gt_rgb = lab_to_rgb(gt_lab)
    pred_rgb = lab_to_rgb(pred_lab)

    # -------------------------------------------------
    # 5. Visualization
    # -------------------------------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_rgb)
    plt.title("Input (Grayscale)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_rgb)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_rgb)
    plt.title("Predicted Color (Annealed Mean)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("demo_result.png")
    print("Saved demo_result.png")


if __name__ == "__main__":
    main()
