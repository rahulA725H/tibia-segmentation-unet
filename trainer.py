
import os
import csv
import torch
import torch.nn as nn

from utils import dice_loss_from_logits, dice_score_from_logits


def validate_model(model, val_loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = bce(logits, masks) + dice_loss_from_logits(
                logits, masks
            )

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            dice = dice_score_from_logits(logits, masks)
            total_dice += dice * batch_size
            total_images += batch_size

    return total_loss / total_images, total_dice / total_images


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=60,
    learning_rate=0.0003,
    output_dir="sessions",
    patience=15
):
    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    bce = nn.BCEWithLogitsLoss()

    best_dice = -1.0
    epochs_without_improvement = 0

    checkpoint_path = os.path.join(
        output_dir, "best_tibia_unet.pth"
    )

    history_path = os.path.join(
        output_dir, "training_history.csv"
    )

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_dice", "learning_rate"
        ])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = (
                bce(logits, masks)
                + dice_loss_from_logits(logits, masks)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples

        val_loss, val_dice = validate_model(
            model, val_loader, device
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        with open(history_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                val_dice,
                current_lr
            ])

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print("  Best model saved!")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Best Dice: {best_dice:.4f}"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
