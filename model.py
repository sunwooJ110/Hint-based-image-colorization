# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetColorizationNet(nn.Module):
    def __init__(self, num_bins=313):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------- Bottleneck --------
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------- Decoder --------
        self.up2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------- Heads --------
        self.class_head = nn.Conv2d(128, num_bins, 1)
        self.reg_head   = nn.Conv2d(128, 2, 1)

    def forward(self, L, ab_hint, mask):
        x = torch.cat([L, ab_hint, mask], dim=1)

        e1 = self.enc1(x)   # (B,64,H,W)
        e2 = self.enc2(e1)  # (B,128,H/2,W/2)
        e3 = self.enc3(e2)  # (B,256,H/4,W/4)

        m = self.mid(e3)    # (B,512,H/4,W/4)

        d2 = self.up2(m)                    # (B,256,H/2,W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                   # (B,128,H,W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.class_head(d1)        # (B,313,H,W)
        ab_reg = torch.tanh(self.reg_head(d1))

        return logits, ab_reg
