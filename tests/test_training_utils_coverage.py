from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.training_utils import (
    AlignmentDataset,
    SCARTrainer,
    SemanticCompressionDataset,
    TrainingConfig,
    create_synthetic_training_data,
)


class _TinyCompressor(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_reconstruction: bool = False):
        reconstructed = self.proj(x)
        compressed = reconstructed[:, : max(1, reconstructed.shape[1] // 2)]
        if return_reconstruction:
            return compressed, reconstructed
        return compressed

    def compute_preservation_loss(self, targets, reconstructed):
        return ((targets - reconstructed) ** 2).mean()


class _TinyAlignment(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, d, use_projection: bool = True):
        if use_projection:
            q = self.proj(q)
            d = self.proj(d)
        # per-sample distance for trainer's contrastive objective
        loss = ((q - d) ** 2).mean(dim=1)
        return loss, {"mean": loss.mean().item()}


def _tiny_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=1,
        weight_decay=0.0,
        log_interval=100,
        eval_interval=100,
        scheduler_type="step",
        device="cpu",
    )


def test_dataset_contracts_and_synthetic_data_shapes():
    emb = np.random.randn(6, 8).astype(np.float32)
    ds = SemanticCompressionDataset(emb)
    assert len(ds) == 6
    item = ds[0]
    assert set(item.keys()) == {"embedding", "target"}

    queries = np.random.randn(6, 8).astype(np.float32)
    positives = np.random.randn(6, 8).astype(np.float32)
    negatives = np.random.randn(6, 8).astype(np.float32)
    align_ds = AlignmentDataset(queries, positives, negatives)
    assert len(align_ds) == 6
    assert set(align_ds[0].keys()) == {"query", "positive", "negative"}

    tr, ev = create_synthetic_training_data(num_samples=20, embedding_dim=8)
    assert tr.shape == (20, 8)
    assert ev.shape == (4, 8)


def test_trainer_compressor_train_eval_checkpoint(tmp_path: Path, monkeypatch):
    np.random.seed(42)
    torch.manual_seed(42)
    config = _tiny_config(tmp_path)
    model = _TinyCompressor(dim=8)
    trainer = SCARTrainer(model=model, config=config, model_type="compressor")

    train = SemanticCompressionDataset(np.random.randn(8, 8).astype(np.float32))
    eval_ds = SemanticCompressionDataset(np.random.randn(4, 8).astype(np.float32))
    history = trainer.train_compressor(train, eval_ds)
    assert history["train_losses"]
    assert history["eval_losses"]

    ckpt = tmp_path / "compressor.pt"
    trainer.save_checkpoint(str(ckpt))
    assert ckpt.exists()

    real_torch_load = torch.load
    monkeypatch.setattr(
        "src.training_utils.torch.load",
        lambda *args, **kwargs: real_torch_load(*args, weights_only=False, **kwargs),
    )
    trainer.load_checkpoint(str(ckpt))
    assert trainer.global_step > 0
    assert trainer.current_epoch == 0


def test_trainer_alignment_train_eval(tmp_path: Path):
    np.random.seed(7)
    torch.manual_seed(7)
    config = _tiny_config(tmp_path)
    config.scheduler_type = "cosine"

    model = _TinyAlignment(dim=8)
    trainer = SCARTrainer(model=model, config=config, model_type="alignment")

    q = np.random.randn(8, 8).astype(np.float32)
    p = np.random.randn(8, 8).astype(np.float32)
    n = np.random.randn(8, 8).astype(np.float32)
    train = AlignmentDataset(q, p, n)
    eval_ds = AlignmentDataset(q[:4], p[:4], n[:4])

    history = trainer.train_alignment(train, eval_ds)
    assert history["train_losses"]
    assert history["eval_losses"]
