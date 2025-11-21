"""
Training Utilities for SCAR Modules

Provides training infrastructure for:
1. Learnable Semantic Compressor (SCAR Section 3.2)
2. Semantic Alignment Module (SCAR Section 3.3)

Training follows SCAR paper methodology with semantic preservation loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for SCAR training"""

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 1e-5

    # Loss weights (inspired by SCAR)
    preservation_weight: float = 1.0  # L_pres weight (Eq. 4)
    alignment_weight: float = 0.5  # L_align weight (Eq. 8)

    # Optimization
    warmup_epochs: int = 2
    scheduler_type: str = "cosine"  # "cosine" or "step"

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticCompressionDataset(Dataset):
    """
    Dataset for training SCAR's learnable compressor.

    Creates training samples from document embeddings with:
    - Original embeddings as input
    - Same embeddings as reconstruction target (self-supervised)
    """

    def __init__(self, embeddings: np.ndarray, chunk_texts: Optional[List[str]] = None):
        """
        Initialize dataset.

        Args:
            embeddings: Document embeddings [N, embedding_dim]
            chunk_texts: Optional text chunks for reference
        """
        self.embeddings = torch.from_numpy(embeddings).float()
        self.chunk_texts = chunk_texts

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embeddings[idx],
            "target": self.embeddings[idx],  # Self-supervised reconstruction
        }


class AlignmentDataset(Dataset):
    """
    Dataset for training SCAR's semantic alignment module.

    Creates (query, relevant_doc, irrelevant_doc) triplets for:
    - Contrastive learning
    - Alignment guidance training
    """

    def __init__(
        self,
        query_embeddings: np.ndarray,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
    ):
        """
        Initialize alignment dataset.

        Args:
            query_embeddings: Query embeddings [N, dim]
            positive_embeddings: Relevant document embeddings [N, dim]
            negative_embeddings: Irrelevant document embeddings [N, dim]
        """
        self.queries = torch.from_numpy(query_embeddings).float()
        self.positives = torch.from_numpy(positive_embeddings).float()
        self.negatives = torch.from_numpy(negative_embeddings).float()

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "query": self.queries[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
        }


class SCARTrainer:
    """
    Trainer for SCAR modules following the paper's training methodology.

    Training objectives (from SCAR paper):
    1. Semantic Preservation Loss (Eq. 4): L_pres = ||Fs - Uk(Fc)||^2
    2. Semantic Alignment Loss (Eq. 8): L_align = ||Hs - Pt||^2
    3. Combined: L_total = L_CE + δ * L_align (SCAR uses δ=0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_type: str = "compressor",  # "compressor" or "alignment"
    ):
        """
        Initialize trainer.

        Args:
            model: SCAR module to train
            config: Training configuration
            model_type: Type of model ("compressor" or "alignment")
        """
        self.model = model.to(config.device)
        self.config = config
        self.model_type = model_type
        self.device = config.device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = None  # Will be set when training starts

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_training_steps // 3, gamma=0.1
            )

    def train_compressor(
        self,
        train_dataset: SemanticCompressionDataset,
        eval_dataset: Optional[SemanticCompressionDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the learnable semantic compressor.

        Implements SCAR's semantic preservation objective (Eq. 4).

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training history with losses
        """
        print(f"\n{'='*70}")
        print(f"Training Learnable Semantic Compressor (SCAR Section 3.2)")
        print(f"{'='*70}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        num_training_steps = len(train_loader) * self.config.num_epochs
        self._setup_scheduler(num_training_steps)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_compressor_epoch(train_loader)

            print(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {epoch_loss:.6f}")

            # Evaluation
            if eval_dataset is not None:
                eval_loss = self._evaluate_compressor(eval_dataset)
                print(f"  Eval Loss: {eval_loss:.6f}")
                self.eval_losses.append(eval_loss)

                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    print(f"  ✓ New best model (loss: {eval_loss:.6f})")

        return {"train_losses": self.train_losses, "eval_losses": self.eval_losses}

    def _train_compressor_epoch(self, train_loader: DataLoader) -> float:
        """Train compressor for one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            embeddings = batch["embedding"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass with reconstruction
            compressed, reconstructed = self.model(embeddings, return_reconstruction=True)

            # SCAR Equation 4: L_pres = ||Fs - Uk(Fc)||^2
            loss = self.model.compute_preservation_loss(targets, reconstructed)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            epoch_losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.global_step += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return np.mean(epoch_losses)

    def _evaluate_compressor(self, eval_dataset: SemanticCompressionDataset) -> float:
        """Evaluate compressor on validation set"""
        self.model.eval()
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)

        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                embeddings = batch["embedding"].to(self.device)
                targets = batch["target"].to(self.device)

                compressed, reconstructed = self.model(embeddings, return_reconstruction=True)

                loss = self.model.compute_preservation_loss(targets, reconstructed)
                eval_losses.append(loss.item())

        return np.mean(eval_losses)

    def train_alignment(
        self,
        train_dataset: AlignmentDataset,
        eval_dataset: Optional[AlignmentDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the semantic alignment module.

        Implements SCAR's alignment objective (Eq. 8) with contrastive learning.

        Args:
            train_dataset: Training dataset with (query, pos, neg) triplets
            eval_dataset: Optional evaluation dataset

        Returns:
            Training history
        """
        print(f"\n{'='*70}")
        print(f"Training Semantic Alignment Module (SCAR Section 3.3)")
        print(f"{'='*70}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        num_training_steps = len(train_loader) * self.config.num_epochs
        self._setup_scheduler(num_training_steps)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_alignment_epoch(train_loader)

            print(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {epoch_loss:.6f}")

            if eval_dataset is not None:
                eval_loss = self._evaluate_alignment(eval_dataset)
                print(f"  Eval Loss: {eval_loss:.6f}")
                self.eval_losses.append(eval_loss)

                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    print(f"  ✓ New best model (loss: {eval_loss:.6f})")

        return {"train_losses": self.train_losses, "eval_losses": self.eval_losses}

    def _train_alignment_epoch(self, train_loader: DataLoader) -> float:
        """Train alignment module for one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            queries = batch["query"].to(self.device)
            positives = batch["positive"].to(self.device)
            negatives = batch["negative"].to(self.device)

            # Compute alignment loss for positive pairs (SCAR Eq. 8)
            pos_loss, _ = self.model(queries, positives, use_projection=True)

            # Compute alignment loss for negative pairs (should be high)
            neg_loss, _ = self.model(queries, negatives, use_projection=True)

            # Contrastive objective: minimize pos_loss, maximize neg_loss
            # Margin-based triplet loss
            margin = 0.5
            loss = torch.relu(pos_loss - neg_loss + margin).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            epoch_losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.global_step += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return np.mean(epoch_losses)

    def _evaluate_alignment(self, eval_dataset: AlignmentDataset) -> float:
        """Evaluate alignment module"""
        self.model.eval()
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)

        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                queries = batch["query"].to(self.device)
                positives = batch["positive"].to(self.device)
                negatives = batch["negative"].to(self.device)

                pos_loss, _ = self.model(queries, positives, use_projection=True)
                neg_loss, _ = self.model(queries, negatives, use_projection=True)

                margin = 0.5
                loss = torch.relu(pos_loss - neg_loss + margin).mean()
                eval_losses.append(loss.item())

        return np.mean(eval_losses)

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_loss = checkpoint["best_loss"]

        print(f"✓ Checkpoint loaded from {filepath}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best loss: {self.best_loss:.6f}")


def create_synthetic_training_data(
    num_samples: int = 1000, embedding_dim: int = 384
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for demonstration.

    In practice, use real document embeddings from your corpus.

    Args:
        num_samples: Number of synthetic samples
        embedding_dim: Embedding dimension

    Returns:
        train_embeddings, eval_embeddings
    """
    print(f"Creating {num_samples} synthetic training samples...")

    # Generate synthetic embeddings (normally distributed, then normalized)
    train_embeddings = np.random.randn(num_samples, embedding_dim)
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

    eval_embeddings = np.random.randn(num_samples // 5, embedding_dim)
    eval_embeddings = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

    return train_embeddings, eval_embeddings


# Example usage
if __name__ == "__main__":
    """
    Demonstrate SCAR training pipeline
    """
    print("=" * 70)
    print("SCAR Training Pipeline Demo")
    print("=" * 70)

    # Import SCAR modules
    from scar_compressor import LearnableSemanticCompressor, SemanticAlignmentModule

    # Training config
    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=5,
        preservation_weight=1.0,
        alignment_weight=0.5,
    )

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")

    # Create synthetic data
    train_embeddings, eval_embeddings = create_synthetic_training_data(
        num_samples=1000, embedding_dim=384
    )

    # Create datasets
    train_dataset = SemanticCompressionDataset(train_embeddings)
    eval_dataset = SemanticCompressionDataset(eval_embeddings)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Eval: {len(eval_dataset)}")

    # Initialize and train compressor
    compressor = LearnableSemanticCompressor(
        input_dim=384,
        compressed_dim=96,  # 4× compression like SCAR
    )

    trainer = SCARTrainer(model=compressor, config=config, model_type="compressor")

    history = trainer.train_compressor(train_dataset, eval_dataset)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final eval loss: {history['eval_losses'][-1]:.6f}")
    print(f"Best eval loss: {trainer.best_loss:.6f}")

    # Save checkpoint
    trainer.save_checkpoint("scar_compressor_demo.pt")

    print("\n✓ Demo complete! Check scar_compressor_demo.pt for saved model.")
