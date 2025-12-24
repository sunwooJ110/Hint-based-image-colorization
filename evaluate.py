import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000

from model import UNetColorizationNet
from dataset import ColorizationDataset
from utils import (
    lab_to_rgb,
    annealed_mean,
    generate_hints
)

# --------------------------------------------------
# 설정
# --------------------------------------------------
HINT_RATIOS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05]
NUM_SAMPLES = 50   # 평가에 사용할 이미지 수
MODEL_PATH = "colorization_unet_final.pth"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 평가 함수
# --------------------------------------------------
def compute_metrics(pred_lab, gt_lab, mask, ab_hint):
    # RGB 변환
    pred_rgb = lab_to_rgb(pred_lab)
    gt_rgb = lab_to_rgb(gt_lab)

    # PSNR / SSIM
    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)
    ssim = structural_similarity(
        gt_rgb, pred_rgb, channel_axis=2, data_range=255
    )

    # Delta E (Lab)
    delta_e = deltaE_ciede2000(
        gt_lab.transpose(1, 2, 0),
        pred_lab.transpose(1, 2, 0)
    ).mean()

    # Hint Consistency Error (HCE)
    if mask.sum() > 0:
        ab_pred = pred_lab[1:3]
        hce = np.linalg.norm(
            ab_pred[:, mask[0] == 1] - ab_hint[:, mask[0] == 1],
            axis=0
        ).mean()
    else:
        hce = np.nan

    return psnr, ssim, delta_e, hce


# --------------------------------------------------
# 메인 평가 루프
# --------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 모델 로드
    model = UNetColorizationNet(num_bins=313).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dataset = ColorizationDataset(train=False)

    results = []

    for ratio in HINT_RATIOS:
        print(f"Evaluating hint ratio: {ratio * 100:.1f}%")

        psnr_list, ssim_list, de_list, hce_list = [], [], [], []

        for i in range(NUM_SAMPLES):
            L, _, _, _, ab_gt = dataset[i]
            H, W = ab_gt.shape[1:]

            if ratio == 0.0:
                ab_hint = torch.zeros_like(ab_gt)
                mask = torch.zeros(1, H, W)
            else:
                num_hints = int(ratio * H * W)
                ab_hint_np, mask_np = generate_hints(
                    ab_gt.numpy(), num_hints
                )
                ab_hint = torch.from_numpy(ab_hint_np).float()
                mask = torch.from_numpy(mask_np).float()

            with torch.no_grad():
                logits, _ = model(
                    L.unsqueeze(0).to(device),
                    ab_hint.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device)
                )
                pred_ab = annealed_mean(logits)[0].cpu().numpy()

            pred_lab = np.concatenate([L.numpy(), pred_ab], axis=0)
            gt_lab = np.concatenate([L.numpy(), ab_gt.numpy()], axis=0)

            psnr, ssim, delta_e, hce = compute_metrics(
                pred_lab, gt_lab, mask.numpy(), ab_hint.numpy()
            )

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            de_list.append(delta_e)
            if not np.isnan(hce):
                hce_list.append(hce)

        results.append([
            ratio,
            np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(de_list),
            np.mean(hce_list) if hce_list else np.nan
        ])

    # --------------------------------------------------
    # CSV 저장
    # --------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["HintRatio", "PSNR", "SSIM", "DeltaE", "HCE"])
        writer.writerows(results)

    print(f"Saved CSV to {csv_path}")

    # --------------------------------------------------
    # 그래프 생성
    # --------------------------------------------------
    ratios = [r[0] * 100 for r in results]
    delta_es = [r[3] for r in results]
    hces = [r[4] for r in results]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ratios, delta_es, marker="o")
    plt.xlabel("Hint Ratio (%)")
    plt.ylabel("ΔE (Lab)")
    plt.title("Color Accuracy vs Hint Ratio")

    plt.subplot(1, 2, 2)
    plt.plot(ratios[1:], hces[1:], marker="o")
    plt.xlabel("Hint Ratio (%)")
    plt.ylabel("HCE")
    plt.title("Hint Consistency vs Hint Ratio")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "hint_ratio_curves.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
