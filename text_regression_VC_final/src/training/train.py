"""Training engine — AdamW, MSELoss, warmup scheduler, checkpointing."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import IMDbDataset, collate_fn
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import create_model
from src.training.config import load_config, set_seed


def train_one_config(config_path: str | Path, device: str | None = None) -> dict:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model_name"]
    strategy = cfg.get("strategy", "finetune")
    epochs = int(cfg.get("epochs", 3))
    batch_size = int(cfg.get("batch_size", 16))
    lr = float(cfg.get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    max_length = int(cfg.get("max_length", 256))

    # Data
    csv_path = Path("data/processed/imdb_processed.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Run data preparation first: {csv_path} not found")
    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    # Auto-detect model or use configured one
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name, strategy=strategy)
    model.to(device)

    train_ds = IMDbDataset(df_train, tokenizer, max_length=max_length)
    val_ds = IMDbDataset(df_val, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.MSELoss()

    exp_name = cfg.get("experiment_name", f"{model_name}_{strategy}")
    artifact_dir = Path(f"artifacts/{exp_name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_loss = float("inf")
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None  # type: ignore
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):  # type: ignore
                    preds = model(input_ids, attention_mask)
                    loss = criterion(preds, target)
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(optimizer)  # type: ignore
                scaler.update()  # type: ignore
            else:
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * len(target)

        train_loss = train_loss_sum / len(train_ds)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["target"].to(device)
                preds = model(input_ids, attention_mask)
                val_loss_sum += criterion(preds, target).item() * len(target)
        val_loss = val_loss_sum / len(val_ds)
        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            # Save tokenizer alongside
            model.encoder.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))

    train_time = time.time() - t0

    # Save training history
    with open(artifact_dir / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(history)

    # Save config copy
    import shutil
    shutil.copy(str(config_path), str(artifact_dir / "config.yaml"))

    # Metrics on test — generate predictions.csv
    model.eval()
    test_ds = IMDbDataset(df_test, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer, max_length))
    all_preds: list[float] = []
    all_targets: list[float] = []
    all_texts: list[str] = []
    all_ids: list[str] = []
    infer_t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch["target"].tolist())
            # Retrieve text/ids from underlying df slice
            start = i * batch_size
            end = start + len(batch["target"])
            all_texts.extend(df_test.iloc[start:end]["text"].tolist())
            all_ids.extend(df_test.iloc[start:end]["id"].tolist())
    infer_time = time.time() - infer_t0

    metrics = compute_all_metrics(all_targets, all_preds)
    metrics["train_time_s"] = train_time
    metrics["infer_time_s"] = infer_time

    # Param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["total_params"] = total_params
    metrics["trainable_params"] = trainable_params

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "id": all_ids, "text": all_texts,
        "target": all_targets, "prediction": all_preds,
    })
    pred_df["absolute_error"] = (pred_df["target"] - pred_df["prediction"]).abs()
    pred_df.to_csv(artifact_dir / "predictions.csv", index=False)

    meta = {
        "experiment": exp_name, "model": model_name, "strategy": strategy,
        "seed": seed, "device": device, "train_time_s": train_time,
        "infer_time_s": infer_time, "total_params": total_params,
        "trainable_params": trainable_params, "n_test": len(df_test),
    }
    with open(artifact_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done {exp_name}: {metrics}")
    return metrics
