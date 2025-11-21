# docs/ Directory

## Overview
Research paper analyses and implementation notes explaining the academic foundations for this project. Contains summaries of three key papers that inspire the semantic compression approach.

## Files

### 1. **`CODE_AND_IMAGES.md`** (11,702 bytes)
**Purpose**: Guide for code and image compression features

**Topics Covered**:
- Multi-modal compression architecture
- Code-specific compression with AST parsing
- Image compression using CLIP embeddings
- Cross-modal search capabilities
- Use cases and examples

**Key Sections**:
- **Code Compression**: AST-based parsing for Python, regex for JavaScript
  - Function/class extraction
  - Dependency graph creation
  - Code skeleton generation (signatures only)

- **Image Compression**: CLIP (vision-language model)
  - Image → embedding conversion
  - Cross-modal connections (text ↔ image, code ↔ image)
  - Base64 encoding for retrieval

- **Multi-Modal Projects**: Unified semantic graphs
  - Example: Documentation with diagrams
  - Example: Code repository with screenshots
  - Cross-modal search queries

**Practical Examples**:
```python
# Code compression
code_compressor.ingest_code_file(python_code, "module.py")
skeleton = code_compressor.generate_code_skeleton("module")

# Multi-modal
compressor.ingest_mixed_content([
    {'type': 'text', 'content': readme},
    {'type': 'code', 'content': source_code},
    {'type': 'image', 'content': diagram_bytes}
], project_id="my_project")
```

**Audience**: Users wanting to compress codebases or projects with images

---

### 2. **`SCAR_PAPER_SUMMARY.md`** (10,018 bytes)
**Purpose**: Analysis of SCAR paper and adaptation to text compression

**Paper**: "Semantic Context Matters: Improving Conditioning for Autoregressive Models" (arXiv:2511.14063v1)

**Original Paper Context**:
- SCAR improves vision-language models (VLMs)
- Compresses visual features from 1024 tokens → 256 tokens
- Uses learnable compression + semantic alignment

**Key Concepts**:

#### 1. Compressed Semantic Prefilling (Section 3.2)
**Original**: Compress vision features via parallel downsampling
- Ck(Fs): Convolutional downsampling
- Rk(Fs): Residual downsampling
- Fc = Ck(Fs) + Rk(Fs)

**Our Adaptation**: Compress text embeddings
- Input: 384D embeddings (sentence-transformers)
- Output: 96D compressed embeddings (4× compression)
- Architecture: Parallel Linear branches

**Formula**:
```
Branch1: Linear(384→256) → LayerNorm → GELU → Linear(256→96)
Branch2: Linear(384→96)
Compressed = Branch1 + Branch2
```

**Training**: Reconstruction loss (Eq. 4)
- L_pres = ||Fs - Uk(Fc)||²
- Ensures semantic information is preserved

#### 2. Semantic Alignment Guidance (Section 3.3)
**Original**: Align AR model hidden states with target image semantics
- L_align = ||Hs - Pt||²
- Dense, in-context guidance

**Our Adaptation**: Align retrieved nodes with query semantics
- Improves search relevance
- Optional learnable projection for better alignment
- Combines with cosine similarity for hybrid scoring

**Benefits**:
- Better retrieval (alignment score > pure similarity)
- Adaptive fidelity (high alignment → RAW, low → ABSTRACT)

#### 3. Results from Paper
- 4× compression with minimal quality loss
- Improved downstream task performance
- Validated on vision-language benchmarks

**Our Implementation**:
- `LearnableSemanticCompressor`: Parallel downsampling
- `SemanticAlignmentModule`: Alignment guidance
- `SCAREnhancedCompressor`: Integration layer

**Code Example**:
```python
scar = SCAREnhancedCompressor(
    base_compressor,
    use_learnable_compression=True,
    use_alignment_guidance=True,
    compression_ratio=4.0
)

# Better search with alignment
results = scar.search_with_alignment(query, alignment_weight=0.5)

# Adaptive fidelity
content = scar.adaptive_modulate(query, alignment_threshold=0.7)
```

**Audience**: Researchers, advanced users wanting to understand SCAR integration

---

### 3. **`JSCCM_PAPER_ANALYSIS.md`** (16,961 bytes)
**Purpose**: Analysis of JSCCM paper and application to semantic compression

**Paper**: "Joint Semantic-Channel Coding for Image Transmission" (arXiv:2511.15699v1)

**Original Paper Context**:
- JSCCM = Joint Semantic-Channel Coding for wireless image transmission
- Adapts compression to "channel conditions" (SNR)
- Multi-rate allocation strategy

**Key Concepts**:

#### 1. Semantic Communication Theory
**Principle**: Transmit meaning, not bits
- Traditional: Optimize bit error rate
- Semantic: Optimize task performance

**Analogy to Our System**:
- Wireless channel SNR → Context window availability
- Channel capacity → Token budget
- Image transmission → Document retrieval

#### 2. Multi-Rate Allocation (Section IV-C)
**Original**: Dynamically select compression rate based on SNR
- High SNR → Less compression (better quality)
- Low SNR → More compression (better reliability)

**Our Adaptation**: `AdaptiveRateAllocator`
- High context availability → Higher skeleton_ratio
- Low context availability → Lower skeleton_ratio
- Gumbel-Softmax for differentiable rate selection

**Rate Levels**: [0.10, 0.15, 0.20, 0.25, 0.30]

**Decision Factors**:
1. Document complexity (graph density, clustering, entropy)
2. Context window availability (remaining tokens / max tokens)
3. Query priority (user-specified importance)

#### 3. Two-Branch Architecture (Fig. 3)
**Original**: Parallel JSCC encoders
- Main branch: Global structure (high importance)
- Auxiliary branch: Local details (lower importance)

**Our Adaptation**: `MultiLevelSemanticEncoder`
- Main: Top 15% of nodes (always include)
- Auxiliary: Next 25% of nodes (include if space allows)
- Detail: Remaining nodes (only if plenty of space)

**Progressive Inclusion**:
```
Always: Main branch (essential structure)
If space > 2000 tokens: + Auxiliary branch
If space > 5000 tokens: + Detail nodes
```

#### 4. Channel Adapter Concept
**Original**: Adapt modulation to channel SNR in real-time

**Our Adaptation**: `ContextWindowAdapter`
```python
adapter.adapt_to_context_window(
    file_id="doc",
    available_tokens=5000,  # Low availability
    max_tokens=100000
)
# → Selects lower skeleton_ratio (more compression)
```

**Diagnostics**:
```
Context Availability: 5000 / 100000 (5.0%)
Document Complexity: 0.723
Selected Skeleton Ratio: 10% (level 0)
Reason: Low availability → aggressive compression
```

**Key Insight**: JSCCM's "channel conditions" maps perfectly to LLM context windows

**Our Implementation**:
- `AdaptiveRateAllocator`: Learned rate selection
- `ContextWindowAdapter`: Adaptive skeleton generation
- `MultiLevelSemanticEncoder`: Two-branch architecture

**Audience**: Researchers understanding the communication theory foundation

---

### 4. **`FPQE_PAPER_ANALYSIS.md`** (19,413 bytes)
**Purpose**: Analysis of FPQE paper and structure-preserving compression

**Paper**: "Fidelity-Preserving Quantization Encoding" (arXiv:2511.15695v1)

**Original Paper Context**:
- FPQE for image compression
- Key finding: **SSIM > MSE/PSNR** for predicting downstream performance
- Structure preservation matters more than pixel accuracy

**Key Concepts**:

#### 1. Quality Metrics for Compression
**Traditional Metrics** (pixel-level):
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- Problem: Don't correlate well with task performance

**Structural Metrics** (perceptual):
- SSIM (Structural Similarity Index)
- Advantage: Predicts downstream task success

**Paper's Key Result**:
> "We find that SSIM-based optimization produces better results for downstream tasks than MSE-based optimization, even at the same compression ratio."

#### 2. SSIM Components (for visual images)
Formula: SSIM(x,y) = l(x,y)^α · c(x,y)^β · s(x,y)^γ

1. **Luminance (l)**: Brightness comparison
   - l(x,y) = (2·μ_x·μ_y + c1) / (μ_x² + μ_y² + c1)

2. **Contrast (c)**: Dynamic range comparison
   - c(x,y) = (2·σ_x·σ_y + c2) / (σ_x² + σ_y² + c2)

3. **Structure (s)**: Correlation comparison
   - s(x,y) = (σ_xy + c2) / (σ_x·σ_y + c2)

#### 3. Adaptation to Semantic Graphs
**Our `SemanticSSIM` Implementation**:

1. **Luminance** → **Information Density**
   - Visual: Average brightness
   - Semantic: Average node importance (PageRank)
   - Measures: Are high/low importance distributions preserved?

2. **Contrast** → **Importance Range**
   - Visual: Brightness variance
   - Semantic: Importance variance
   - Measures: Is the dynamic range preserved?

3. **Structure** → **Graph Connectivity**
   - Visual: Pixel correlations
   - Semantic: Graph structure preservation
   - Sub-metrics:
     - Edge preservation: Fraction of edges retained
     - Community preservation: Clustering coefficient similarity
     - Centrality preservation: PageRank correlation

**Formula Adaptation**:
```python
# Luminance: Average importance preservation
mu_orig = mean(original_importance)
mu_comp = mean(compressed_importance)
luminance = (2·mu_orig·mu_comp + c1) / (mu_orig² + mu_comp² + c1)

# Contrast: Variance preservation
sigma_orig = std(original_importance)
sigma_comp = std(compressed_importance)
contrast = (2·sigma_orig·sigma_comp + c2) / (sigma_orig² + sigma_comp² + c2)

# Structure: Graph preservation
edge_preservation = preserved_edges / total_edges
clustering_similarity = compressed_clustering / original_clustering
centrality_preservation = corr(orig_pagerank, comp_pagerank)
structure = 0.5·edge + 0.25·clustering + 0.25·centrality

# Combined SSIM
ssim = luminance^0.33 · contrast^0.33 · structure^0.34
```

#### 4. SSIM Interpretation
**Score Ranges** (validated empirically):
- **SSIM ≥ 0.9**: Excellent preservation
  - Minimal structure loss
  - Safe for all use cases

- **SSIM 0.7-0.9**: Good preservation
  - Minor structure degradation
  - Acceptable for most tasks

- **SSIM 0.5-0.7**: Acceptable preservation
  - Noticeable structure loss
  - May impact complex queries
  - Consider reducing compression

- **SSIM < 0.5**: Poor preservation
  - Significant structure loss
  - Reduce compression ratio
  - Or increase skeleton_ratio

**Usage in Code**:
```python
ssim_calculator = SemanticSSIM()
ssim_score, components = ssim_calculator.calculate_ssim(
    graph,
    original_nodes,
    compressed_nodes
)

if ssim_score >= 0.7:
    print("✅ Good quality compression")
else:
    print("⚠️ Consider reducing compression")
```

#### 5. Why SSIM Matters
**FPQE's Core Insight**: Structure preservation → Better task performance

**Applied to Our System**:
- Preserving semantic structure → Better retrieval accuracy
- Preserving importance distribution → Critical info not lost
- Preserving graph connectivity → Relationships maintained

**Validation**:
- SSIM > 0.7: Retrieval accuracy remains high
- SSIM < 0.5: Retrieval degrades noticeably

**Our Implementation**:
- `SemanticSSIM`: SSIM calculator for graphs
- `calculate_ssim()`: Compute all components
- `interpret_ssim_score()`: Actionable guidance

**Audience**: Researchers understanding quality metrics and validation

---

## Research Papers Summary

| Paper | arXiv | Key Contribution | Our Adaptation |
|-------|-------|------------------|----------------|
| **JSCCM** | 2511.15699v1 | Adaptive rate allocation for channel conditions | Context window adaptation, multi-level encoding |
| **FPQE** | 2511.15695v1 | SSIM > MSE for structure preservation | Semantic SSIM for graph quality |
| **SCAR** | 2511.14063v1 | Learnable compression + alignment guidance | Learnable embeddings, alignment search |

---

## Reading Order

**For New Users**:
1. `CODE_AND_IMAGES.md` - Practical features
2. `SCAR_PAPER_SUMMARY.md` - Advanced compression

**For Researchers**:
1. `JSCCM_PAPER_ANALYSIS.md` - Communication theory foundation
2. `FPQE_PAPER_ANALYSIS.md` - Quality metrics
3. `SCAR_PAPER_SUMMARY.md` - Neural compression
4. `CODE_AND_IMAGES.md` - Implementation details

---

## Mathematical Notation

**Common Symbols**:
- `μ` (mu): Mean
- `σ` (sigma): Standard deviation
- `||·||`: L2 norm
- `⊙`: Element-wise product
- `Fs`: Source features (original)
- `Fc`: Compressed features
- `L_pres`: Preservation loss
- `L_align`: Alignment loss

---

## Additional Resources

**Referenced Papers** (download from arXiv):
- JSCCM: https://arxiv.org/abs/2511.15699v1
- FPQE: https://arxiv.org/abs/2511.15695v1
- SCAR: https://arxiv.org/abs/2511.14063v1

**Related Work**:
- Semantic communication theory
- Joint source-channel coding
- Visual SSIM (Zhou Wang et al., 2004)
- CLIP (Radford et al., 2021)
