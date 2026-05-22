#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewPooling(nn.Module):
    def __init__(self):
        super().__init__()
        # reduz H,W por fator 2
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        """
        x: (B, N, C, H, W)

        returns:
            out: (B, N_out, C, H/2, W/2)

        N_out = (4 * N) + 1
        (1 extra view only for n=0 pooled branch)
        """

        B, N, C, H, W = x.shape

        outputs = []

        for n in range(N):

            # x_n: (B, C, H, W)
            x_n = x[:, n]

            # --------------------------------------------------
            # global pooled branch 
            # output: (B, C, H/2, W/2)
            # --------------------------------------------------
            pooled = self.pool(x_n)
            outputs.append(pooled)


        # (B, N_out, C, H/2, W/2)
        out = torch.stack(outputs, dim=1)

        return out


# =========================
# Quadrant Pooling
# =========================
class QuadrantPooling(nn.Module):
    def __init__(self):
        super().__init__()
        # reduz H,W por fator 2
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def split_quadrants(self, x):
        """
        x: (B, C, H, W)
        returns:
            list of 4 tensors:
            each -> (B, C, H/2, W/2)
        """

        B, C, H, W = x.shape

        h_mid = H // 2
        w_mid = W // 2

        # quadrantes (dependem APENAS do shape de entrada)
        return [
            x[:, :, :h_mid, :w_mid],   # Q1 -> (B, C, H/2, W/2)
            x[:, :, :h_mid, w_mid:],   # Q2 -> (B, C, H/2, W/2)
            x[:, :, h_mid:, :w_mid],   # Q3 -> (B, C, H/2, W/2)
            x[:, :, h_mid:, w_mid:]    # Q4 -> (B, C, H/2, W/2)
        ]

    def forward(self, x):
        """
        x: (B, N, C, H, W)

        returns:
            out: (B, N_out, C, H/2, W/2)

        N_out = (4 * N) + 1
        (1 extra view only for n=0 pooled branch)
        """

        B, N, C, H, W = x.shape

        outputs = []

        for n in range(N):

            # x_n: (B, C, H, W)
            x_n = x[:, n]

            # --------------------------------------------------
            # global pooled branch 
            # output: (B, C, H/2, W/2)
            # --------------------------------------------------
            pooled = self.pool(x_n)
            outputs.append(pooled)

        for n in range(N):

            # x_n: (B, C, H, W)
            x_n = x[:, n]

            # --------------------------------------------------
            # quadrant decomposition ALWAYS
            # each: (B, C, H/2, W/2)
            # --------------------------------------------------
            quadrants = self.split_quadrants(x_n)
            outputs.extend(quadrants)

        # (B, N_out, C, H/2, W/2)
        out = torch.stack(outputs, dim=1)

        return out


# =========================
# Reshape utilities
# =========================
class ViewFlatten(nn.Module):
    def forward(self, x):
        """
        x: (B, N, C, H, W)

        returns:
            (B*N, C, H, W)
        """

        B, N, C, H, W = x.shape
        return x.reshape(B * N, C, H, W)


class ViewUnflatten(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.N = num_views

    def forward(self, x):
        """
        x: (B*N, C, H, W)

        returns:
            (B, N, C, H, W)
        """

        BN, C, H, W = x.shape
        B = BN // self.N

        return x.reshape(B, self.N, C, H, W)


# =========================
# CNN block
# =========================
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3):
        super().__init__()

        self.k_size = k_size
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, k_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            #nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# CNN block
# =========================
class MultiViewCNNBlock(nn.Module):
    def __init__(self, N, in_ch, out_ch, k_size=3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.N = N
        self.net = nn.Sequential(
            # x: (B, N, in_ch, H, W)
            ViewFlatten(),  
            # x: (B*N, in_ch, H, W)
            CNNBlock(in_ch, out_ch, k_size),
            # x: (B*N, out_ch, H, W)
            ViewUnflatten(N)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Full Model
# =========================
class QuadrantNet(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Core operator
        # -------------------------
        self.qp = QuadrantPooling()
        self.mvp = MultiViewPooling()

        # -------------------------
        # CNN stages
        # -------------------------
        self.C = [3, 32, 64, 128, 64]
        
        self.N = [1, 5, 25, 125, 125]

        self.MVCnnBlock1 = MultiViewCNNBlock(self.N[0], self.C[0], self.C[1])
        self.MVCnnBlock2 = MultiViewCNNBlock(self.N[1], self.C[1], self.C[2])
        self.MVCnnBlock3 = MultiViewCNNBlock(self.N[2], self.C[2], self.C[3])
        self.MVCnnBlock4 = MultiViewCNNBlock(self.N[3], self.C[3], self.C[4])
        
        # -------------------------
        # classifier head
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.N[-1]*self.C[-1], 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        """
        x: (B, 3, H, W)
        """

        B = x.shape[0]

        x = x.unsqueeze(1)
        # x: (B, self.N[0]=1, self.C[0]=3, H, W)
        
        # ======================================================
        # STAGE 1
        # ======================================================
        x = self.MVCnnBlock1(x)

        x = self.qp(x)


        # ======================================================
        # STAGE 2 QUADRANT POOLING
        # ======================================================
        x = self.MVCnnBlock2(x)

        x = self.qp(x)


        # ======================================================
        # STAGE 3 QUADRANT POOLING
        # ======================================================
        x = self.MVCnnBlock3(x)

        x = self.qp(x)

       
        # ======================================================
        # STAGE 4 multiview POOLING
        # ======================================================
        x = self.MVCnnBlock4(x)

        x = self.mvp(x)


        # ======================================================
        # SPATIAL AGGREGATION PER VIEW
        # ======================================================
        x = x.mean(dim=[3, 4])

        x = x.flatten(1)


        # ======================================================
        # CLASSIFIER
        # ======================================================
        x = self.classifier(x)
        # x: (B, 1)

        return x
