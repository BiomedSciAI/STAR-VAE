"""
exp19_train_ablation.py
=======================
Generic Python-direct launcher for Exp19 ablation fine-tuning runs.
Bypasses LightningCLI (which crashes with a jsonargparse/ClearML
'morgan_fp' error in this environment) by instantiating
CVAELightningModule and Trainer directly from the YAML config.

Usage
-----
python bmfm_sm_gen/scripts/exp19_train_ablation.py \
    --config paper_results/exp19_ablation_studies/configs/model_vae_ablation_ref.yaml

python bmfm_sm_gen/scripts/exp19_train_ablation.py \
    --config paper_results/exp19_ablation_studies/configs/model_vae_ablation_a2_det_ae.yaml \
    --smoke_test
"""

import argparse
import importlib
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve(path: str) -> str:
    """Make a relative path absolute relative to REPO_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def load_yaml(path: str) -> dict:
    """Load YAML and resolve OmegaConf node and environment interpolations."""
    cfg = OmegaConf.load(path)
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected a mapping at the root of {path}")
    return resolved


def build_model(cfg: dict, smoke_test: bool = False):
    """Instantiate CVAELightningModule from the model section of the YAML."""
    model_cfg = cfg["model"]["init_args"]
    data_cfg  = cfg["data"]["init_args"]

    # Resolve tokenizer and vocab_file from data section (model uses ${data...} refs)
    model_params = dict(model_cfg.get("model_params", {}))

    ModelClass    = import_class(cfg["model"]["class_path"])
    TokenizerCls  = import_class(data_cfg["tokenizer"])
    vocab_file    = resolve(data_cfg["vocab_file"])

    # Build the Lightning module with all hyperparameters from YAML
    init_args = {k: v for k, v in model_cfg.items()
                 if k not in ("model_class", "model_params", "tokenizer", "vocab_file")}

    # Resolve paths
    if "pretrained_path" in init_args and init_args["pretrained_path"]:
        init_args["pretrained_path"] = resolve(init_args["pretrained_path"])
    if "val_csv" in init_args and init_args["val_csv"]:
        init_args["val_csv"] = resolve(init_args["val_csv"])

    lm = ModelClass(
        model_class=cfg["model"]["init_args"]["model_class"],
        model_params=model_params,
        tokenizer=data_cfg["tokenizer"],
        vocab_file=vocab_file,
        **init_args,
    )
    return lm


def build_datamodule(cfg: dict, smoke_test: bool = False):
    """Instantiate MolecularDataModule from the data section of the YAML."""
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    d = cfg["data"]["init_args"]
    train_csv = resolve(d["train_csv"]) if d.get("train_csv") else None
    val_csv   = resolve(d["val_csv"])   if d.get("val_csv")   else None
    if train_csv and val_csv and Path(train_csv).resolve() == Path(val_csv).resolve():
        raise ValueError(
            "Training and validation CSVs resolve to the same file. "
            "Provide a documented, non-overlapping validation split."
        )

    dm = MolecularDataModule(
        train_mode=d.get("train_mode", "mem"),
        data_dir=d.get("data_dir", ""),
        train_csv=train_csv,
        val_csv=val_csv,
        tokenizer=d["tokenizer"],
        vocab_file=resolve(d["vocab_file"]),
        max_len=d.get("max_len", 71),
        batch_size=d.get("batch_size", 64) if not smoke_test else 8,
        dataset_args=d.get("dataset_args", {}),
    )
    return dm


def main():
    parser = argparse.ArgumentParser(description="Exp19 ablation training (direct Python launcher)")
    parser.add_argument("--config",     required=True, help="Path to ablation YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 steps only")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Training seed. The published Exp19 runs used 42. Pass a different "
             "value to obtain an independent fine-tuning replicate.",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Appended to the trainer output directory, so replicate seeds do "
             "not overwrite each other (e.g. '_seed7' -> <root>_seed7).",
    )
    args = parser.parse_args()

    cfg_path = args.config if Path(args.config).is_absolute() else str(REPO_ROOT / args.config)
    log.info(f"Config: {cfg_path}")

    cfg = load_yaml(cfg_path)
    # Seeds Python, NumPy, and torch (CPU + CUDA). Note that deterministic cuDNN
    # kernels are NOT enabled, so runs are statistically independent replicates
    # rather than bitwise-reproducible.
    pl.seed_everything(args.seed)
    log.info(f"Training seed: {args.seed}")

    # Enable Tensor Core utilization if available (A30/A100 have bf16 Tensor Cores)
    import torch
    torch.set_float32_matmul_precision("high")

    t = cfg["trainer"]
    if args.output_suffix:
        t["default_root_dir"] = t["default_root_dir"].rstrip("/") + args.output_suffix
    out_dir  = resolve(t["default_root_dir"])
    ckpt_dir = resolve(t["default_root_dir"]) + "/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    log.info(f"Output dir: {out_dir}")

    # ── Trainer topology from YAML (supports multi-GPU / multi-node DDP) ─────
    devices   = t.get("devices", 1)
    num_nodes = t.get("num_nodes", 1)
    strategy  = t.get("strategy", "auto")

    # Explicitly set "ddp" when multi-node or multi-device
    if (isinstance(devices, int) and devices > 1 or num_nodes > 1) and strategy == "auto":
        strategy = "ddp"

    # ── Model + data ──────────────────────────────────────────────────────────
    log.info("Building model…")
    lm = build_model(cfg, args.smoke_test)

    log.info("Building datamodule…")
    dm = build_datamodule(cfg, args.smoke_test)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = []

    ckpt_cb = ModelCheckpoint(
        dirpath   = ckpt_dir,
        filename  = "vae-epoch={epoch:02d}-step={step}-val_loss={val_loss:.2f}",
        monitor   = "val_loss",
        mode      = "min",
        save_top_k= 3,
        save_last = True,
        save_on_train_epoch_end=False,
    )
    callbacks.append(ckpt_cb)

    # EarlyStopping (from YAML callbacks if present, else default)
    patience = 8
    for cb in t.get("callbacks", []):
        if "EarlyStopping" in cb.get("class_path", ""):
            patience = cb.get("init_args", {}).get("patience", 8)
    callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # ── Resume ────────────────────────────────────────────────────────────────
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.isfile(last_ckpt) else None
    if ckpt_path:
        log.info(f"Resuming from: {ckpt_path}")
    else:
        log.info("No last.ckpt found. Starting from scratch.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    max_epochs    = 2 if args.smoke_test else t.get("max_epochs", 20)
    limit_batches = 2 if args.smoke_test else 1.0

    log.info(f"Trainer topology: devices={devices}  num_nodes={num_nodes}  strategy={strategy}")

    # limit_val_batches: from YAML when set, else 1.0 (full val). Smoke test overrides.
    limit_val_batches_cfg = t.get("limit_val_batches", 1.0)
    if args.smoke_test:
        limit_val_batches_cfg = limit_batches  # 2 steps in smoke test

    # precision: bf16-mixed uses Tensor Cores on A30/A100; fall back to 32 if not set
    precision = t.get("precision", "32")

    trainer = pl.Trainer(
        accelerator          = t.get("accelerator", "auto"),
        devices              = devices,
        num_nodes            = num_nodes,
        strategy             = strategy,
        precision            = precision,
        max_epochs           = max_epochs,
        val_check_interval   = t.get("val_check_interval", 1.0),
        log_every_n_steps    = t.get("log_every_n_steps", 50),
        default_root_dir     = out_dir,
        logger               = TensorBoardLogger(save_dir=out_dir),
        callbacks            = callbacks,
        gradient_clip_val    = 1.0,
        limit_train_batches  = limit_batches,
        limit_val_batches    = limit_val_batches_cfg,
    )

    log.info(f"Training: max_epochs={max_epochs}  devices={devices}  smoke_test={args.smoke_test}")
    trainer.fit(lm, datamodule=dm, ckpt_path=ckpt_path)

    log.info(f"Training complete. Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
