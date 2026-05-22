#!/usr/bin/python3

import torch
from torchinfo import summary

# assume que QuadrantNet está no mesmo ficheiro ou importado
from model import QuadrantNet


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_report(model):
    print("\n================ MODEL REPORT ================\n")

    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_params(model)

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")

    print("\n=============================================\n")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H=W=512

    # ----------------------------
    # Model
    # ----------------------------
    model = QuadrantNet().to(device)
    model.eval()

    print_model_report(model)

    # ----------------------------
    # Fake input
    # ----------------------------
    B = 1
    x = torch.randn(B, 3, H, W).to(device)

    print("\nInput shape:", x.shape)

    # ----------------------------
    # Forward pass
    # ----------------------------
    with torch.no_grad():
        y = model(x)

    print("Output shape:", y.shape)

    # ----------------------------
    # Torchinfo summary (optional but VERY useful)
    # ----------------------------
    print("\n================ TORCHINFO SUMMARY ================\n")

    summary(
        model,
        input_size=(B, 3, H, W),
        depth=3,
        col_names=["input_size", "output_size", "num_params"],
        verbose=1
    )


if __name__ == "__main__":
    main()
