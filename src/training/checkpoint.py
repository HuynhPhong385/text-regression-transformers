"""Checkpoint manager: best.pt and last.pt per experiment."""

from pathlib import Path

import torch


class CheckpointManager:
    """Lưu checkpoint trực tiếp trong thư mục experiment."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.best_val_loss = float("inf")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        val_loss,
        filename="last.pt",
    ):
        path = self.directory / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                    if optimizer is not None
                    else None
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
                ),
                "val_loss": float(val_loss),
            },
            path,
        )

        return path

    @staticmethod
    def load(
        path,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):
        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and checkpoint.get("optimizer_state_dict")
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint.get("scheduler_state_dict")
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        return checkpoint
