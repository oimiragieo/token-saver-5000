"""
Multi-Modal Compression Example

Demonstrates compressing mixed content: text, code, AND images.

Use cases:
- Documentation with diagrams
- Code repositories with screenshots
- Technical papers with figures
- Project READMEs with images

Requires: pip install Pillow (for image handling)
Optional: pip install clip-ViT-B-32 (for CLIP image embeddings)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal_compressor import MultiModalCompressor


def create_sample_image():
    """Create a sample image for demonstration"""
    try:
        from PIL import Image, ImageDraw
        import io

        # Create a simple diagram image
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)

        # Draw a simple neural network diagram
        draw.rectangle([50, 50, 150, 100], outline="black", width=2)
        draw.text((70, 70), "Input Layer", fill="black")

        draw.rectangle([200, 50, 300, 100], outline="black", width=2)
        draw.text((210, 70), "Hidden Layer", fill="black")

        draw.rectangle([50, 150, 150, 200], outline="black", width=2)
        draw.text((60, 170), "Output Layer", fill="black")

        # Draw arrows
        draw.line([(150, 75), (200, 75)], fill="black", width=2)
        draw.line([(150, 175), (200, 75)], fill="black", width=2)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    except ImportError:
        print("⚠️  PIL not installed, using placeholder")
        return b"placeholder_image_data"


def main():
    print("=" * 80)
    print("MULTI-MODAL COMPRESSION EXAMPLE")
    print("Text + Code + Images in Unified Semantic Graph")
    print("=" * 80)

    # Initialize multi-modal compressor
    print("\n[1] INITIALIZING MULTI-MODAL COMPRESSOR")
    compressor = MultiModalCompressor(
        use_clip_for_images=True,  # Try to use CLIP for images
        use_codebert_for_code=False,  # Use general model for simplicity
    )

    # Prepare multi-modal content (simulating a project)
    print("\n[2] PREPARING MIXED CONTENT")

    content_items = [
        # README text
        {
            "type": "text",
            "content": """
# Neural Network Project

This project implements a neural network for image classification.

## Features
- Modular architecture
- PyTorch backend
- Training utilities
- Visualization tools

## Performance
The model achieves 95% accuracy on CIFAR-10 after 50 epochs.
            """,
            "metadata": {"file": "README.md", "section": "overview"},
        },
        # Training code
        {
            "type": "code",
            "content": '''
def train_model(model, train_loader, optimizer, epochs=10):
    """
    Train neural network model.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer (e.g., Adam, SGD)
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} complete")

    return model
            ''',
            "metadata": {"file": "train.py", "function": "train_model"},
        },
        # Model architecture code
        {
            "type": "code",
            "content": '''
class ConvNet(nn.Module):
    """Convolutional Neural Network for image classification"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
            ''',
            "metadata": {"file": "model.py", "class": "ConvNet"},
        },
        # Results documentation
        {
            "type": "text",
            "content": """
## Training Results

Final test accuracy: 95.2%
Training time: 2 hours on GPU
Best epoch: 48

The model shows strong generalization with minimal overfitting.
Confusion matrix shows good performance across all classes.
            """,
            "metadata": {"file": "RESULTS.md"},
        },
        # Architecture diagram (image)
        {
            "type": "image",
            "content": create_sample_image(),
            "metadata": {
                "file": "architecture.png",
                "description": "Network architecture diagram",
            },
        },
    ]

    print(f"   Prepared {len(content_items)} items:")
    print(f"     {sum(1 for x in content_items if x['type'] == 'text')} text documents")
    print(f"     {sum(1 for x in content_items if x['type'] == 'code')} code files")
    print(f"     {sum(1 for x in content_items if x['type'] == 'image')} images")

    # Ingest all content
    print("\n[3] INGESTING MULTI-MODAL CONTENT")
    stats = compressor.ingest_mixed_content(
        content_items=content_items,
        project_id="nn_project",
        similarity_threshold=0.65,  # Lower threshold for cross-modal connections
    )

    print("\n📊 Ingestion Statistics:")
    for key, value in stats.items():
        if key != "cross_modal_connections":
            print(f"   {key}: {value}")

    if stats.get("cross_modal_connections"):
        print("   cross_modal_connections:")
        for conn_type, count in stats["cross_modal_connections"].items():
            print(f"     {conn_type}: {count}")

    # Generate project summary
    print("\n" + "=" * 80)
    print("[4] PROJECT SUMMARY")
    print("=" * 80)
    summary = compressor.generate_multimodal_summary("nn_project")
    print(summary)

    # Cross-modal search examples
    print("\n" + "=" * 80)
    print("[5] CROSS-MODAL SEARCH EXAMPLES")
    print("=" * 80)

    # Example 1: Find code related to "training"
    print("\n🔍 Example 1: Text query → Find related CODE")
    print("-" * 80)
    query = "how to train the neural network"
    results = compressor.search_cross_modal(
        query=query,
        query_type="text",
        project_id="nn_project",
        filter_modality="code",  # Only return code
        top_k=2,
    )

    print(f"Query: '{query}'")
    print("Filter: CODE only\n")
    for i, (node_id, score, modality) in enumerate(results, 1):
        node_data = compressor.get_node_content(node_id)
        preview = node_data["content"][:80].replace("\n", " ")
        file = node_data["metadata"].get("file", "unknown")
        print(f"{i}. {file} (score: {score:.3f})")
        print(f"   {preview}...\n")

    # Example 2: Find text documentation about architecture
    print("\n🔍 Example 2: Code query → Find related TEXT")
    print("-" * 80)
    query = "class ConvNet"
    results = compressor.search_cross_modal(
        query=query,
        query_type="code",
        project_id="nn_project",
        filter_modality="text",  # Only return text
        top_k=2,
    )

    print(f"Query: '{query}'")
    print("Filter: TEXT only\n")
    for i, (node_id, score, modality) in enumerate(results, 1):
        node_data = compressor.get_node_content(node_id)
        preview = node_data["content"][:100].replace("\n", " ")
        file = node_data["metadata"].get("file", "unknown")
        print(f"{i}. {file} (score: {score:.3f})")
        print(f"   {preview}...\n")

    # Example 3: Find images related to architecture
    print("\n🔍 Example 3: Text query → Find related IMAGES")
    print("-" * 80)
    query = "neural network architecture diagram"
    results = compressor.search_cross_modal(
        query=query,
        query_type="text",
        project_id="nn_project",
        filter_modality="image",  # Only return images
        top_k=1,
    )

    print(f"Query: '{query}'")
    print("Filter: IMAGES only\n")
    for i, (node_id, score, modality) in enumerate(results, 1):
        node_data = compressor.get_node_content(node_id)
        file = node_data["metadata"].get("file", "unknown")
        description = node_data["metadata"].get("description", "No description")
        print(f"{i}. {file} (score: {score:.3f})")
        print(f"   Description: {description}")
        print(f"   Content type: {node_data['content_type']}")

    # Practical use cases
    print("\n" + "=" * 80)
    print("[6] PRACTICAL USE CASES")
    print("=" * 80)

    print("\n✅ Use Case 1: Documentation with Diagrams")
    print("   Scenario: Technical documentation with architecture diagrams")
    print("   Workflow:")
    print("     1. Ingest markdown docs + PNG/SVG diagrams")
    print("     2. Ask: 'Show me the architecture diagram'")
    print("     3. AI uses cross-modal search to find relevant image")
    print("     4. Returns base64-encoded image for display")
    print("   → Compress docs 90%, retrieve images on demand")

    print("\n✅ Use Case 2: Code + Screenshots")
    print("   Scenario: Tutorial with code and UI screenshots")
    print("   Workflow:")
    print("     1. Ingest code files + screenshot images")
    print("     2. Ask: 'Show code for the login screen'")
    print("     3. AI finds related screenshot AND code")
    print("     4. Progressive retrieval: skeleton → code → screenshot")
    print("   → Show only relevant code + images, save 85%+ tokens")

    print("\n✅ Use Case 3: Research Papers")
    print("   Scenario: Scientific paper with equations and figures")
    print("   Workflow:")
    print("     1. Ingest PDF text + extract figures as images")
    print("     2. Ask: 'Explain the main results'")
    print("     3. AI retrieves text sections + relevant figures")
    print("     4. Combined text + image response")
    print("   → Compress text, retrieve figures contextually")

    print("\n✅ Use Case 4: Project README")
    print("   Scenario: GitHub README with badges, diagrams, screenshots")
    print("   Workflow:")
    print("     1. Ingest README.md + all referenced images")
    print("     2. Semantic graph connects text to images")
    print("     3. Query: 'How does it work?' → text + architecture diagram")
    print("     4. Query: 'Show example usage' → code + screenshot")
    print("   → Intelligent multi-modal retrieval")

    # Token savings with images
    print("\n" + "=" * 80)
    print("[7] TOKEN SAVINGS ANALYSIS")
    print("=" * 80)

    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Calculate total text tokens (code + text)
    total_text_tokens = 0
    for item in content_items:
        if item["type"] in ["text", "code"]:
            total_text_tokens += len(tokenizer.encode(item["content"]))

    # Images are typically sent as base64, which is 4/3 * bytes
    # In token terms, ~1.5 tokens per byte (rough estimate)
    total_image_tokens = 0
    for item in content_items:
        if item["type"] == "image":
            # Base64 encoding increases size by 33%
            base64_size = len(item["content"]) * 4 / 3
            # Rough token estimate (1 token ≈ 4 chars)
            total_image_tokens += base64_size / 4

    # Summary shows only skeleton
    summary_tokens = len(tokenizer.encode(summary))

    # Typical workflow: skeleton + 2 code retrievals (no images unless asked)
    workflow_tokens = summary_tokens + (total_text_tokens * 0.15)  # ~15% of text

    print("\n📊 Token Usage Comparison:")
    print(f"   Full project (text): {total_text_tokens:,} tokens")
    print(f"   Full project (images): ~{int(total_image_tokens):,} tokens (base64)")
    print(f"   Total if sent raw: ~{int(total_text_tokens + total_image_tokens):,} tokens")
    print("\n   With compression:")
    print(f"     Summary only: {summary_tokens:,} tokens")
    print(f"     Typical workflow: ~{int(workflow_tokens):,} tokens")
    print("\n   Savings:")
    print(f"     Summary: {(1 - summary_tokens/(total_text_tokens + total_image_tokens))*100:.1f}%")
    print(
        f"     Workflow: {(1 - workflow_tokens/(total_text_tokens + total_image_tokens))*100:.1f}%"
    )

    print("\n" + "=" * 80)
    print("✅ Multi-modal compression complete!")
    print("=" * 80)

    print("\n💡 Key Insights:")
    print("   • Text, code, and images in UNIFIED semantic graph")
    print("   • Cross-modal search (text → code, code → images, etc.)")
    print("   • Images retrieved on demand (not sent by default)")
    print("   • ~85-95% token savings even with images")
    print("   • Perfect for documentation, tutorials, papers, READMEs")

    print("\n📦 Installation Notes:")
    print("   • Basic: Works with text + code (no extra deps)")
    print("   • Images: pip install Pillow (for image handling)")
    print("   • CLIP: pip install clip-ViT-B-32 (for better image search)")


if __name__ == "__main__":
    main()
