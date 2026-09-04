"""Generic PyTorch training loop for E03-E06."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import calculate_metrics, save_json
from .checkpoint import CheckpointManager
from .losses import get_loss


class Trainer:
    """Trainer dùng chung cho BERT/RoBERTa/DistilBERT."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        output_dir,
        experiment_id,
        epochs=3,
        learning_rate=2e-5,
        patience=3,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Mỗi experiment có thư mục riêng: outputs/E03/
        self.output_dir = (
            Path(output_dir) / experiment_id
        )
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.experiment_id = experiment_id
        self.epochs = epochs
        self.patience = patience

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        trainable_parameters = [
            p
            for p in self.model.parameters()
            if p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=0.01,
        )

        total_steps = max(
            len(train_loader) * epochs,
            1,
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(
                total_steps // 10,
                1,
            ),
            num_training_steps=total_steps,
        )

        self.loss_fn = get_loss("mse")

        self.checkpoints = CheckpointManager(
            self.output_dir
        )

    @staticmethod
    def move_batch_to_device(batch, device):
        result = {}

        for key, value in batch.items():
            if torch.is_tensor(value):
                result[key] = value.to(device)
            else:
                result[key] = torch.as_tensor(
                    value,
                    dtype=torch.float32,
                    device=device,
                )

        return result

    def _forward_loss(self, batch):
        inputs = {
            key: value
            for key, value in batch.items()
            if key != "labels"
        }

        predictions = self.model(**inputs)

        loss = self.loss_fn(
            predictions,
            batch["labels"].float(),
        )

        return predictions, loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []

        for batch in tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        ):
            batch = self.move_batch_to_device(
                batch,
                self.device,
            )

            self.optimizer.zero_grad()

            batch_predictions, loss = self._forward_loss(
                batch
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            predictions.extend(
                batch_predictions.detach()
                .cpu()
                .numpy()
                .tolist()
            )
            targets.extend(
                batch["labels"]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

        metrics = calculate_metrics(
            targets,
            predictions,
        )

        return (
            total_loss / max(len(self.train_loader), 1),
            metrics,
        )

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss = 0.0
        predictions = []
        targets = []

        for batch in loader:
            batch = self.move_batch_to_device(
                batch,
                self.device,
            )

            batch_predictions, loss = self._forward_loss(
                batch
            )

            total_loss += loss.item()

            predictions.extend(
                batch_predictions.cpu()
                .numpy()
                .tolist()
            )
            targets.extend(
                batch["labels"]
                .cpu()
                .numpy()
                .tolist()
            )

        metrics = calculate_metrics(
            targets,
            predictions,
        )

        return (
            total_loss / max(len(loader), 1),
            metrics,
            predictions,
            targets,
        )

    def fit(self):
        history = []
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            print(
                f"\n{self.experiment_id} | "
                f"Epoch {epoch}/{self.epochs}"
            )

            train_loss, train_metrics = (
                self.train_epoch()
            )

            val_loss, val_metrics, _, _ = self.evaluate(
                self.val_loader
            )

            # last.pt: lưu mỗi epoch.
            self.checkpoints.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                val_loss,
                "last.pt",
            )

            # best.pt: chỉ lưu khi validation loss tốt hơn.
            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                self.checkpoints.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    "best.pt",
                )
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "validation_loss": val_loss,
                "validation_MAE": val_metrics["MAE"],
                "validation_RMSE": val_metrics["RMSE"],
                "validation_R2": val_metrics["R2"],
            }

            history.append(row)

            print(
                f"Train Loss={train_loss:.4f} | "
                f"Train MAE={train_metrics['MAE']:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Val MAE={val_metrics['MAE']:.4f} | "
                f"Val RMSE={val_metrics['RMSE']:.4f} | "
                f"Val R2={val_metrics['R2']:.4f}"
            )

            if (
                epochs_without_improvement
                >= self.patience
            ):
                print(
                    f"Early stopping: validation loss "
                    f"không cải thiện sau "
                    f"{self.patience} epoch."
                )
                break

        best_path = self.output_dir / "best.pt"

        # Luôn đánh giá test bằng best.pt, không phải epoch cuối.
        if best_path.exists():
            checkpoint = torch.load(
                best_path,
                map_location=self.device,
            )
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        test_loss, test_metrics, predictions, targets = (
            self.evaluate(self.test_loader)
        )

        pd.DataFrame(history).to_csv(
            self.output_dir / "training_log.csv",
            index=False,
        )

        prediction_file = (
            self.output_dir / "predictions.csv"
        )

        pd.DataFrame(
            {
                "target": targets,
                "prediction": predictions,
                "error": (
                    np.asarray(predictions)
                    - np.asarray(targets)
                ),
            }
        ).to_csv(
            prediction_file,
            index=False,
        )

        result = {
            "experiment": self.experiment_id,
            "epochs_requested": self.epochs,
            "epochs_completed": len(history),
            "patience": self.patience,
            "test_loss": float(test_loss),
            "test": test_metrics,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(
                self.output_dir / "last.pt"
            ),
        }

        save_json(
            result,
            self.output_dir / "result.json",
        )

        return result
