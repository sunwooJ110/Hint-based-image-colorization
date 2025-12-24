# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import UNetColorizationNet
from dataset import ColorizationDataset
from utils import compute_rebalance_weights


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    # 1. Dataset / DataLoader
    dataset = ColorizationDataset(
        train=True,
        hint_ratios=(0.01, 0.03, 0.05),
        max_samples=5000
    )

    if device == "cuda":
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    # 2. Model / Optimizer
    model = UNetColorizationNet(num_bins=313).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3. Class rebalancing weights
    prior = np.ones(313, dtype=np.float32)
    class_weights = compute_rebalance_weights(prior)
    class_weights = torch.tensor(class_weights, device=device)

    # 4. Training loop
    num_epochs = 10 
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for L, ab_hint, mask, ab_bin, ab_gt in loader:
            L = L.to(device)
            ab_hint = ab_hint.to(device)
            mask = mask.to(device)
            ab_bin = ab_bin.to(device)
            ab_gt = ab_gt.to(device)

            # Forward
            logits, ab_reg = model(L, ab_hint, mask)

            # Classification loss
            loss_class = F.cross_entropy(
                logits,
                ab_bin,
                weight=class_weights,
                reduction="none"
            )

            hint_weight_cls = 1.0 + 5.0 * mask.squeeze(1)
            loss_class = (loss_class * hint_weight_cls).sum() / hint_weight_cls.sum()

            # Regression loss
            ab_gt_norm = ab_gt / 110.0
            loss_reg = F.smooth_l1_loss(
                ab_reg,
                ab_gt_norm,
                reduction="none"
            )

            hint_weight_reg = 1.0 + 5.0 * mask
            loss_reg = (loss_reg * hint_weight_reg).sum() / hint_weight_reg.sum()

            # Final loss
            loss = loss_class + 0.1 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 5. Checkpoint saving 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                "colorization_unet_best.pth"
            )
            print("Saved best model")

    # 6. Final model save
    torch.save(model.state_dict(), "colorization_unet_final.pth")
    print("Training finished. Final model saved.")


if __name__ == "__main__":
    main()
