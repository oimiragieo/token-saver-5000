# examples/ Directory

## Overview
Runnable examples demonstrating various features of the Semantic Modulator, including basic usage, SCAR enhancements, code compression, and multi-modal document processing.

## Files

### 1. **`example_usage.py`** (12,745 bytes)
**Purpose**: Basic usage demonstration of semantic compression workflow

**Features Demonstrated**:
- Document ingestion and semantic graph creation
- Skeleton view generation
- Semantic search
- Adaptive fidelity modulation (5 levels)
- Token savings calculation
- Statistics retrieval

**Example Flow**:
```python
# 1. Initialize
compressor = SemanticCompressor()

# 2. Ingest document
result = compressor.ingest_file(document_text, "paper_1")
print(f"Compressed: {result.total_tokens} → {result.skeleton_tokens} tokens")

# 3. Read skeleton (80-95% token reduction)
skeleton = compressor.read_skeleton("paper_1")

# 4. Search semantically
results = compressor.search_semantic("quantum error threshold", "paper_1", top_k=3)

# 5. Retrieve at different fidelity levels
abstract = compressor.modulate_region(results, FidelityLevel.ABSTRACT)
structure = compressor.modulate_region(results, FidelityLevel.STRUCTURE)
raw = compressor.modulate_region(results, FidelityLevel.RAW)
```

**Sample Document**: Research paper on quantum error correction (~500 tokens)

**Output Example**:
```
=== SEMANTIC SKELETON: paper_1 ===
Total nodes: 8 | Skeleton nodes: 2
Compression: 20% of content shown

[paper_1_n0] ⭐ ANCHOR (importance: 0.185)
  Summary: Quantum Error Correction is essential for practical quantum computing...
  Key entities: Quantum, Error, Correction

[paper_1_n2] 📦 Detail hidden (use modulate_region to expand)
```

**Run Command**:
```bash
python examples/example_usage.py
```

**Key Learnings**:
- Skeleton first, details on demand (progressive disclosure)
- 5 fidelity levels for adaptive token budgets
- Search → Retrieve workflow

---

### 2. **`scar_demo.py`** (9,061 bytes)
**Purpose**: SCAR-enhanced retrieval demonstration

**SCAR Features Demonstrated**:
- Learnable embedding compression (384D → 96D, 4× compression)
- Semantic alignment guidance for better search
- Adaptive fidelity based on alignment scores
- Compression statistics

**Example Flow**:
```python
# 1. Initialize base compressor
base_compressor = SemanticCompressor()
base_compressor.ingest_file(document, "quantum_doc")

# 2. Wrap with SCAR enhancements
scar = SCAREnhancedCompressor(
    base_compressor=base_compressor,
    use_learnable_compression=True,
    use_alignment_guidance=True,
    compression_ratio=4.0
)

# 3. Alignment-guided search (better than cosine similarity alone)
results = scar.search_with_alignment(
    query="What is the error threshold for surface codes?",
    file_id="quantum_doc",
    top_k=3,
    alignment_weight=0.5  # Balance similarity + alignment
)

# 4. Adaptive modulation (fidelity adapts to alignment score)
content = scar.adaptive_modulate(
    query="error threshold",
    file_id="quantum_doc",
    alignment_threshold=0.7
)
# High alignment → RAW (full detail)
# Medium alignment → STRUCTURE
# Low alignment → ABSTRACT
```

**Sample Document**: Quantum computing paper with complex error correction concepts

**Output Example**:
```
=== SCAR ADAPTIVE MODULATION ===
Query: What is the error threshold for surface codes?

🔥 Alignment Score: 0.85 → Fidelity: RAW
[quantum_doc_n2] Full Content:
--- BEGIN ---
The error threshold for surface codes is approximately 1%.
This makes them practical for near-term quantum computers.
--- END ---

⭐ Alignment Score: 0.62 → Fidelity: STRUCTURE
[quantum_doc_n4] Structure:
  Summary: Recent advances in code concatenation have improved efficiency.
  Entities: concatenation, efficiency
  Tokens: 42
  Importance: 0.156
```

**Run Command**:
```bash
python examples/scar_demo.py
```

**Key Learnings**:
- SCAR alignment improves search quality
- Adaptive fidelity saves tokens on less relevant content
- 4× embedding compression reduces memory footprint

---

### 3. **`code_compression_example.py`** (10,366 bytes)
**Purpose**: Code-specific compression with AST analysis

**Features Demonstrated**:
- AST-based code parsing (Python, JavaScript)
- Function and class extraction
- Dependency graph creation
- Code skeleton generation (signatures only)
- Semantic code search

**Example Flow**:
```python
# 1. Initialize code compressor
code_compressor = CodeSemanticCompressor()

# 2. Ingest Python code
sample_code = '''
import numpy as np
from sklearn.metrics import accuracy_score

def preprocess_data(data, normalize=True):
    """Preprocess input data for model training."""
    if normalize:
        data = (data - np.mean(data)) / np.std(data)
    return data

class NeuralNetwork:
    """Simple neural network implementation."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network"""
        hidden = np.dot(x, self.weights1)
        output = np.dot(hidden, self.weights2)
        return output
'''

stats = code_compressor.ingest_code_file(
    code=sample_code,
    file_id="neural_net",
    filepath="neural_net.py"
)

# 3. Generate code skeleton (signatures, not bodies)
skeleton = code_compressor.generate_code_skeleton("neural_net")

# 4. Semantic code search
results = code_compressor.search_code(
    "How do I train the model?",
    "neural_net",
    top_k=3
)

# 5. Retrieve full function
content = code_compressor.get_code_chunk(results[0][0])
```

**Output Example**:
```
=== CODE SKELETON: neural_net ===
Total chunks: 4 | Showing: 2

📦 IMPORTS:
   numpy, sklearn.metrics.accuracy_score

⭐ FUNCTION: forward (importance: 0.287)
   Signature: def forward(self, x):
   Doc: Forward pass through the network...
   Lines: 15-18

⭐ CLASS: NeuralNetwork (importance: 0.245)
   Doc: Simple neural network implementation...
   Methods: __init__, forward
   Lines: 10-18

📦 2 additional chunks hidden
   Use search_code() or modulate_code() to retrieve them
```

**Run Command**:
```bash
python examples/code_compression_example.py
```

**Supported Languages**:
- ✅ Python (full AST support)
- ✅ JavaScript / TypeScript (regex-based)
- ⚠️ Java, C++, Go, Rust (line-based fallback)

**Key Learnings**:
- AST parsing preserves code structure better than text chunking
- Function signatures provide context without token overhead
- Dependency graphs show code relationships

---

### 4. **`multimodal_example.py`** (13,297 bytes)
**Purpose**: Multi-modal document processing (text + code + images)

**Features Demonstrated**:
- Unified semantic graph across modalities
- Cross-modal semantic connections
- CLIP-based image embedding
- Cross-modal search
- Multi-modal project summary

**Example Flow**:
```python
# 1. Initialize multi-modal compressor
compressor = MultiModalCompressor(
    use_clip_for_images=True,
    use_codebert_for_code=False
)

# 2. Prepare mixed content
content_items = [
    {
        'type': 'text',
        'content': 'This project implements a neural network for image classification using PyTorch.',
        'metadata': {'file': 'README.md'}
    },
    {
        'type': 'code',
        'content': '''
def train_model(model, data_loader, epochs=10):
    """Train the neural network model"""
    for epoch in range(epochs):
        for batch in data_loader:
            loss = model(batch)
            loss.backward()
    return model
''',
        'metadata': {'file': 'train.py', 'function': 'train_model'}
    },
    {
        'type': 'image',
        'content': image_bytes,  # PNG/JPEG bytes
        'metadata': {'file': 'architecture_diagram.png'}
    }
]

# 3. Ingest mixed content
stats = compressor.ingest_mixed_content(content_items, "ml_project")

# 4. Cross-modal search
# Example: Find code related to "training"
results = compressor.search_cross_modal(
    query="training neural networks",
    query_type="text",
    project_id="ml_project",
    filter_modality="code",  # Only return code
    top_k=3
)

# Example: Find images related to code
results = compressor.search_cross_modal(
    query=code_snippet,
    query_type="code",
    filter_modality="image",  # Only return images
    top_k=2
)

# 5. Generate multi-modal summary
summary = compressor.generate_multimodal_summary("ml_project")
```

**Output Example**:
```
=== MULTI-MODAL PROJECT: ml_project ===
Total items: 5

📄 TEXT DOCUMENTS (2):
  ml_project_n0: This project implements a neural network for image... (importance: 0.245)
  ml_project_n3: The model achieves 95% accuracy on the test set... (importance: 0.198)

💻 CODE FILES (2):
  ml_project_n1 (train.py): def train_model(model, data_loader, epochs=10):... (importance: 0.287)
  ml_project_n4 (model.py): class NeuralNetwork:... (importance: 0.225)

🖼️  IMAGES (1):
  ml_project_n2 (architecture_diagram.png): 245.3KB (importance: 0.156)

🔗 TOP CROSS-MODAL CONNECTIONS:
  ml_project_n0 (text) ↔ ml_project_n1 (code): 0.823
  ml_project_n1 (code) ↔ ml_project_n2 (image): 0.756
  ml_project_n0 (text) ↔ ml_project_n2 (image): 0.712
```

**Run Command**:
```bash
python examples/multimodal_example.py
```

**Use Cases**:
- Documentation with diagrams
- Code repositories with README + screenshots
- Research papers with figures
- Tutorial content (text + code + visuals)

**Key Learnings**:
- CLIP enables text ↔ image search
- Cross-modal connections reveal relationships
- Unified embedding space for all modalities

**Note**: Full image support requires CLIP installation:
```bash
pip install sentence-transformers[clip]
```

---

## Running All Examples

### Run individually
```bash
python examples/example_usage.py
python examples/scar_demo.py
python examples/code_compression_example.py
python examples/multimodal_example.py
```

### Run all with output
```bash
for script in examples/*.py; do
    echo "=== Running $script ==="
    python "$script"
    echo ""
done
```

---

## Example Dependencies

**Basic Examples** (example_usage.py, code_compression_example.py):
- sentence-transformers
- networkx
- tiktoken

**SCAR Example** (scar_demo.py):
- + torch
- + torch.nn

**Multi-modal Example** (multimodal_example.py):
- + PIL (Pillow)
- + clip-ViT-B-32 (optional, for images)

---

## Expected Runtime

- `example_usage.py`: ~5-10 seconds (embedding model load + inference)
- `scar_demo.py`: ~10-15 seconds (+ SCAR module initialization)
- `code_compression_example.py`: ~5-8 seconds
- `multimodal_example.py`: ~15-20 seconds (+ CLIP model load)

---

## Learning Path

**Recommended Order**:
1. **Start here**: `example_usage.py` - Learn basic workflow
2. **Code-specific**: `code_compression_example.py` - If compressing code
3. **Advanced**: `scar_demo.py` - Learn SCAR enhancements
4. **Multi-modal**: `multimodal_example.py` - If working with images

---

## Customization

All examples can be modified:
- Change `SAMPLE_DOCUMENT` to your own content
- Adjust `similarity_threshold` for graph density
- Modify `skeleton_ratio` for compression amount
- Experiment with fidelity levels for token budget

---

## Troubleshooting

**Issue**: Model download slow
**Fix**: Models are cached in `~/.cache/huggingface/` after first run

**Issue**: CLIP not available
**Fix**: Multi-modal example will warn but run without image support

**Issue**: Out of memory
**Fix**: Reduce document size or use CPU-only mode
