import torch
import matplotlib.pyplot as plt
import numpy as np

from model import UNetColorizationNet
from dataset import ColorizationDataset
from utils import lab_to_rgb, annealed_mean, generate_hints


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)


    # 1. 이미지 하나 고정
    base_dataset = ColorizationDataset(
        train=False,
        hint_ratios=(0.01,) 
    )

    # 같은 이미지 하나 고정
    L, _, _, _, ab_gt = base_dataset[0]

    H, W = ab_gt.shape[1:]

    # 2. 모델 로드 
    model = UNetColorizationNet(num_bins=313).to(device)
    model.load_state_dict(
        torch.load("colorization_unet_final.pth", map_location=device)
    )
    model.eval()


    # 3. 힌트 비율 설정
    hint_ratios = [0.001, 0.01, 0.03]
    titles = ["0.1% Hint", "1% Hint", "3% Hint"]

    results = []

    for ratio in hint_ratios:
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

        # batch 차원 추가
        L_in = L.unsqueeze(0).to(device)
        ab_hint_in = ab_hint.unsqueeze(0).to(device)
        mask_in = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(L_in, ab_hint_in, mask_in)
            pred_ab = annealed_mean(logits, T=0.38)

        pred_lab = np.concatenate(
            [L.numpy(), pred_ab[0].cpu().numpy()],
            axis=0
        )
        pred_rgb = lab_to_rgb(pred_lab)
        results.append(pred_rgb)

    # 4. 시각화
    plt.figure(figsize=(12, 4))

    for i, (img, title) in enumerate(zip(results, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("hint_ratio_comparison_fixed.png")
    print("Saved hint_ratio_comparison_fixed.png")


if __name__ == "__main__":
    main()
