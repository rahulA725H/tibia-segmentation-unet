
import os
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from model import UNet
from dataset import KneeDataset, read_xray


def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)

    return dice, iou, precision


def predict(model, image, device, threshold):
    tensor = torch.from_numpy(
        image.copy()
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= threshold).float()

    return mask[0, 0].cpu().numpy()


def save_comparison(image, actual, predicted, path):
    overlay = cv2.cvtColor(
        (image[0] * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    overlay[predicted > 0] = (
        0.5 * overlay[predicted > 0]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    panels = [
        (image[0], "Original X-ray", "gray"),
        (actual, "Ground Truth", "gray"),
        (predicted, "Prediction", "gray"),
        (overlay, "Tibia Overlay", None)
    ]

    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None,
                  vmax=1 if cmap else None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_history(history_path, output_dir):
    if not os.path.exists(history_path):
        print("Training history not found.")
        return

    history = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        history["epoch"], history["train_loss"],
        label="Training"
    )
    axes[0].plot(
        history["epoch"], history["val_loss"],
        label="Validation"
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history["epoch"], history["val_dice"],
        label="Validation Dice"
    )
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "training_curves.png"),
        dpi=200
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="sessions/final/best_tibia_unet.pth"
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", default="demo_images/final")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cpu")
    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    print("Best model loaded successfully!")

    val_df = pd.read_csv("data/CSVs/val.csv")
    val_dataset = KneeDataset(val_df, augment=False)

    results = []

    for i in range(len(val_dataset)):
        sample = val_dataset[i]

        image = sample["xray"]
        actual = sample["mask"][0]

        predicted = predict(
            model, image, device, args.threshold
        )

        dice, iou, precision = calculate_metrics(
            predicted, actual
        )

        results.append({
            "image": val_df.iloc[i]["xrays"],
            "dice": dice,
            "iou": iou,
            "precision": precision
        })

        if i < 5:
            save_comparison(
                image, actual, predicted,
                os.path.join(
                    args.output,
                    f"validation_{i+1}.png"
                )
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(args.output, "validation_metrics.csv"),
        index=False
    )

    print("\nVALIDATION RESULTS")
    print("----------------------------")
    print(f"Images: {len(results_df)}")
    print(f"Mean Dice: {results_df['dice'].mean():.4f}")
    print(f"Mean IoU: {results_df['iou'].mean():.4f}")
    print(f"Mean Precision: {results_df['precision'].mean():.4f}")

    test_df = pd.read_csv("data/CSVs/test.csv")

    for i, row in test_df.iterrows():
        image = read_xray(row["xrays"])

        predicted = predict(
            model, image, device, args.threshold
        )

        mask_uint8 = (predicted * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(args.output, f"test_mask_{i+1:03d}.png"),
            mask_uint8
        )

        if i < 5:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            axes[0].imshow(image[0], cmap="gray")
            axes[0].set_title("Test X-ray")
            axes[1].imshow(predicted, cmap="gray")
            axes[1].set_title("Predicted Tibia")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(args.output, f"test_example_{i+1}.png"),
                dpi=200
            )
            plt.close()

    plot_history(
        "sessions/final/training_history.csv",
        args.output
    )

    print(f"\nTest predictions generated: {len(test_df)}")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()
