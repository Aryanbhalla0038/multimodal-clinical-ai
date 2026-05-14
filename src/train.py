"""Training entrypoint.

Usage:
    python -m src.train --config configs/base.yaml --modalities img,txt,vit \
                         --fusion cross_attention --labels_csv data/raw/train.csv \
                         --image_root data/raw/CheXpert-v1.0-small
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils.config import load_yaml
from src.data.dataset import MultimodalDataset
from src.models.multimodal_model import MultimodalClinicalModel
from src.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--modalities", type=str, default="img,txt,vit")
    p.add_argument("--fusion", type=str, default="cross_attention",
                   choices=["cross_attention", "early"])
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--notes_col", type=str, default=None)
    p.add_argument("--vitals_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no_pretrained_image", action="store_true",
                   help="Disable downloading ViT weights (offline mode).")
    p.add_argument("--limit_train_batches", type=int, default=None,
                   help="Stop each epoch after N batches (debugging).")
    return p.parse_args()


def build_loader(csv_path, image_root, notes_col, vitals_dir, split, cfg, model_cfg, batch_size, num_workers):
    ds = MultimodalDataset(
        labels_csv=csv_path,
        image_root=image_root,
        notes_col=notes_col,
        vitals_dir=vitals_dir,
        split=split,
        image_size=cfg["data"]["image_size"],
        text_max_len=cfg["data"]["text_max_len"],
        vitals_seq_len=cfg["data"]["vitals_seq_len"],
        uncertainty_policy=cfg["data"]["uncertainty_policy"],
        u_ones_labels=cfg["data"]["uncertainty_u_ones"],
        text_model_name=model_cfg["text"]["backbone"],
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    modalities = set(args.modalities.split(","))

    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA unavailable, falling back to CPU.")

    epochs = args.epochs or cfg["training"]["epochs"]
    bs = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    nw = cfg["training"]["num_workers"]

    train_loader = build_loader(args.labels_csv, args.image_root, args.notes_col,
                                args.vitals_dir, "train", cfg, cfg["model"], bs, nw)
    val_loader = build_loader(args.val_csv or args.labels_csv, args.image_root,
                              args.notes_col, args.vitals_dir, "val", cfg, cfg["model"], bs, nw)

    model = MultimodalClinicalModel(
        num_labels=cfg["model"]["num_labels"],
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        fusion=args.fusion,
        image_backbone=cfg["model"]["image"]["backbone"],
        image_pretrained=cfg["model"]["image"]["pretrained"] and not args.no_pretrained_image,
        text_backbone=cfg["model"]["text"]["backbone"],
        vitals_encoder_type=cfg["model"]["vitals"]["encoder_type"],
        vitals_input_dim=cfg["model"]["vitals"]["input_dim"],
        vitals_hidden_dim=cfg["model"]["vitals"]["hidden_dim"],
        vitals_num_layers=cfg["model"]["vitals"]["num_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg["training"]["mixed_precision"]))

    run_name = args.run_name or f"{args.modalities.replace(',', '_')}__{args.fusion}"
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg["wandb"]["enabled"]
    if use_wandb:
        import wandb  # local import — optional dep at runtime
        wandb.init(project=cfg["wandb"]["project"], name=run_name,
                   config={**cfg, "modalities": args.modalities, "fusion": args.fusion})

    best_auc = -1.0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            if args.limit_train_batches and step >= args.limit_train_batches:
                break

            kwargs = {}
            if "img" in modalities:
                kwargs["image"] = batch["image"].to(device, non_blocking=True)
            if "txt" in modalities:
                kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
                kwargs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)
            if "vit" in modalities:
                kwargs["vitals"] = batch["vitals"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg["training"]["mixed_precision"])):
                logits = model(**kwargs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if step % cfg["training"]["log_interval"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        metrics = evaluate(model, val_loader, device, modalities=modalities)
        train_loss = running_loss / max(1, len(train_loader))
        log = {"epoch": epoch + 1, "train_loss": round(train_loss, 4), **metrics,
               "lr": optimizer.param_groups[0]["lr"]}
        print({k: log[k] for k in ("epoch", "train_loss", "macro_AUC", "macro_AP", "macro_F1@0.5")})
        if use_wandb:
            import wandb
            wandb.log(log)

        # Save best
        if metrics["macro_AUC"] > best_auc:
            best_auc = metrics["macro_AUC"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_auc": metrics["macro_AUC"],
                "config": dict(cfg),
                "modalities": args.modalities,
                "fusion": args.fusion,
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ Saved best to {ckpt_dir/'best_model.pt'} (macro AUC={best_auc})")

    print(f"Done. Best macro AUC = {best_auc}")


if __name__ == "__main__":
    main()
