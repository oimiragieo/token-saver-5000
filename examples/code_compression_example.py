"""
Code Compression Example

Demonstrates how to use Semantic Modulator for source code compression.

Features:
- AST-based intelligent chunking (functions, classes)
- Code skeleton generation
- Semantic code search
- Dependency analysis
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.code_compressor import CodeSemanticCompressor


# Sample Python code (a complete module)
SAMPLE_MODULE = '''
"""
Neural Network Training Module

This module provides utilities for training deep learning models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output classes
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_data_loader(dataset, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    """
    Evaluate model on validation/test set.

    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

class Trainer:
    """
    Model trainer with early stopping and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_val_loss = float('inf')

    def train(self, epochs=10, patience=3):
        """
        Train model with early stopping.

        Args:
            epochs: Number of epochs
            patience: Early stopping patience

        Returns:
            Training history
        """
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = train_epoch(
                self.model, self.train_loader,
                self.criterion, self.optimizer
            )

            # Evaluate
            val_loss, val_acc = evaluate_model(
                self.model, self.val_loader, self.criterion
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(f"best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        return history

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
'''


def main():
    print("=" * 80)
    print("CODE COMPRESSION EXAMPLE - Neural Network Training Module")
    print("=" * 80)

    # Initialize code compressor
    compressor = CodeSemanticCompressor()

    # Ingest code
    print("\n[1] INGESTING CODE FILE")
    stats = compressor.ingest_code_file(
        code=SAMPLE_MODULE,
        file_id="nn_trainer",
        filepath="nn_trainer.py",
        metadata={
            "module": "neural network training",
            "language": "python",
            "lines": len(SAMPLE_MODULE.split("\n")),
        },
    )

    print("\n📊 Compression Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Functions: {stats['chunk_types']['functions']}")
    print(f"   Classes: {stats['chunk_types']['classes']}")
    print(f"   Imports: {stats['chunk_types']['imports']}")
    print(f"   Graph nodes: {stats['graph_nodes']}")
    print(f"   Graph edges: {stats['graph_edges']}")

    # Generate skeleton
    print("\n" + "=" * 80)
    print("[2] CODE SKELETON (High-Level Overview)")
    print("=" * 80)
    skeleton = compressor.generate_code_skeleton("nn_trainer", show_top_n=5)
    print(skeleton)

    # Semantic code search examples
    print("\n" + "=" * 80)
    print("[3] SEMANTIC CODE SEARCH")
    print("=" * 80)

    queries = [
        "How do I train the model?",
        "How do I evaluate accuracy?",
        "What is the network architecture?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)

        results = compressor.search_code(query, "nn_trainer", top_k=2)

        for i, (chunk_id, score) in enumerate(results, 1):
            chunk = compressor.chunks[chunk_id]
            print(f"\n{i}. {chunk_id} (score: {score:.3f})")
            print(f"   Type: {chunk.chunk_type}")
            print(f"   Name: {chunk.name}")
            if chunk.docstring:
                doc_preview = chunk.docstring.split("\n")[0][:70]
                print(f"   Doc: {doc_preview}...")
            print(f"   Lines: {chunk.start_line}-{chunk.end_line}")

    # Retrieve specific code chunks
    print("\n" + "=" * 80)
    print("[4] RETRIEVING SPECIFIC FUNCTIONS")
    print("=" * 80)

    # Find the train_epoch function
    print("\n📦 Retrieving: train_epoch function")
    print("-" * 80)

    # Search for it first
    results = compressor.search_code("train for one epoch", "nn_trainer", top_k=1)
    if results:
        chunk_id = results[0][0]
        code = compressor.get_code_chunk(chunk_id, include_context=False)
        print(code)

    # Token savings analysis
    print("\n" + "=" * 80)
    print("[5] TOKEN SAVINGS ANALYSIS")
    print("=" * 80)

    # Count tokens in original code
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    original_tokens = len(tokenizer.encode(SAMPLE_MODULE))

    # Count tokens in skeleton
    skeleton_tokens = len(tokenizer.encode(skeleton))

    # Count tokens in search results (typical workflow)
    # User sees skeleton + retrieves 2-3 relevant functions
    typical_workflow_tokens = skeleton_tokens

    for query in queries[:2]:  # Simulate retrieving 2 functions
        results = compressor.search_code(query, "nn_trainer", top_k=1)
        if results:
            chunk = compressor.chunks[results[0][0]]
            typical_workflow_tokens += len(tokenizer.encode(chunk.code))

    print(f"\n📊 Token Usage:")
    print(f"   Original code: {original_tokens:,} tokens")
    print(f"   Skeleton only: {skeleton_tokens:,} tokens")
    print(
        f"   Typical workflow (skeleton + 2 functions): {typical_workflow_tokens:,} tokens"
    )
    print(
        f"\n   Savings with skeleton: {(1 - skeleton_tokens/original_tokens)*100:.1f}%"
    )
    print(
        f"   Savings with workflow: {(1 - typical_workflow_tokens/original_tokens)*100:.1f}%"
    )

    # Use case examples
    print("\n" + "=" * 80)
    print("[6] PRACTICAL USE CASES")
    print("=" * 80)

    print("\n✅ Use Case 1: Code Review")
    print("   - Ingest entire codebase")
    print("   - Review skeleton to understand structure")
    print("   - Search for 'security issues' or 'error handling'")
    print("   - Retrieve only relevant functions for review")
    print("   → Save 80-90% of tokens vs reading entire codebase")

    print("\n✅ Use Case 2: Documentation Generation")
    print("   - Ingest source files")
    print("   - Extract all function signatures + docstrings from skeleton")
    print("   - Generate API documentation")
    print("   → Only use docstrings, ignore implementation details")

    print("\n✅ Use Case 3: Debugging")
    print("   - Ingest project code")
    print("   - Search for 'where does X get called?'")
    print("   - Find dependencies automatically")
    print("   - Retrieve only relevant call chain")
    print("   → Focus on relevant code, save 90%+ tokens")

    print("\n✅ Use Case 4: AI-Assisted Coding")
    print("   - Feed skeleton to AI (low tokens)")
    print("   - AI asks for specific functions when needed")
    print("   - Progressive retrieval keeps context small")
    print("   → Work with large codebases in limited context windows")

    print("\n" + "=" * 80)
    print("✅ Code compression complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
