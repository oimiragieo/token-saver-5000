# SCAR Paper Implementation Summary

## Paper Details
**Title:** Semantic Context Matters: Improving Conditioning for Autoregressive Models
**arXiv:** 2511.14063v1 [cs.CV]
**Published:** November 18, 2025
**Authors:** Dongyang Jin, Ryan Xu, Jianhao Zeng, Rui Lan, Yancheng Bai, Lei Sun, Xiangxiang Chu (Amap, Alibaba Group)
**License:** arXiv.org perpetual non-exclusive license

---

## Core Problem

Autoregressive (AR) models for image generation struggle with:
1. **Weak conditioning** - poor instruction adherence
2. **Inefficient conditioning** - high computational cost
3. **Visual artifacts** - especially in image editing tasks

Existing approaches fall into two categories:
- **Decoding-stage injection** (e.g., ControlAR): Strong control but disrupts AR process, causes artifacts
- **Prefilling-stage conditioning** (e.g., EditAR): Flexible but uses inefficient VQ tokens with shallow semantics

---

## SCAR Solution

**SCAR** (Semantic-Context-driven AutoregRessive) introduces two key innovations:

### 1. Compressed Semantic Prefilling (Section 3.2)

**Problem:** VQ token prefixes are:
- Inefficient (1024 tokens for 512×512 image)
- Semantically sparse (low-level visual features)

**Solution:**
- Use Vision Foundation Model (DINOv2) to extract high-level semantic features
- Learnable compression module `Pk(·)` compresses features 4× (1024 → 256 tokens)
- Trained with semantic preservation loss:

```
Fc = Ck(Fs) + Rk(Fs)              # Equation 3: Parallel compression
L_pres = ||Fs - Uk(Fc)||^2         # Equation 4: Reconstruction loss
```

**Benefits:**
- **4× fewer tokens** while preserving semantics
- **23.9% less GPU memory**
- **1.42× faster training**
- Robust to compression (VQ tokens degrade sharply when compressed)

### 2. Semantic Alignment Guidance (Section 3.3)

**Problem:** Text instructions are sparse, providing weak guidance for dense visual token generation.

**Solution:**
- Use target image's semantic features (from VFM) as dense guidance
- Align AR model's internal hidden states with target semantics during training:

```
Pt = Pk(E(It))                    # Equation 6: Target semantics
Hs = G(S)[1:Lc, :]                # Equation 7: Model hidden states
L_align = ||Hs - Pt||^2            # Equation 8: Alignment loss
```

**Benefits:**
- Dense, in-context learning signal
- Improves semantic accuracy
- Better instruction following

### Combined Training Objective

```
L_total = L_CE + δ * L_align
```

Where:
- `L_CE`: Standard cross-entropy loss for token prediction
- `δ = 0.5`: Alignment weight (found optimal through ablation)

---

## Results

### C2I Controllable Generation (ImageNet-256)

| Method | Paradigm | Canny FID↓ | Depth FID↓ | HED FID↓ |
|--------|----------|------------|------------|----------|
| ControlAR | Next-token | 7.69 | 4.19 | - |
| ControlVAR | Next-set | 7.85 | 6.50 | - |
| CAR | Next-set | 8.30 | 6.90 | 5.60 |
| **SCAR (VAR-d20)** | Next-set | **1.97** | **3.29** | **1.51** |
| **SCAR (LlamaGen-L)** | Next-token | **2.69** | **2.69** | **2.67** |

**SCAR achieves 2-4× better FID** while maintaining competitive control accuracy.

### T2I Controllable Generation (MultiGen-20M, 512×512)

| Method | HED FID↓ | HED SSIM↑ | Depth FID↓ | Canny FID↓ |
|--------|----------|-----------|------------|------------|
| ControlNet | 15.41 | 76.21 | 17.76 | 14.73 |
| ControlNet++ | 15.01 | 80.97 | 16.66 | 18.23 |
| ControlAR | 10.53 | 85.63 | 14.61 | 17.51 |
| EditAR | - | - | 15.97 | 13.91 |
| **SCAR** | **8.41** | **83.09** | **13.77** | **10.82** |

**SCAR outperforms all methods** in image quality (FID) while maintaining strong control.

### Instruction Editing (PIE-Bench)

| Method | Structure Distance↓ | PSNR↑ | LPIPS↓ | CLIP Similarity↑ |
|--------|---------------------|-------|--------|------------------|
| InstructPix2Pix | 107.43 | 16.69 | 271.33 | 23.49 |
| MGIE | 67.41 | 21.20 | 142.25 | 24.28 |
| SEED-X-Edit | 61.69 | 18.80 | 173.63 | 25.51 |
| ControlAR* | 116.99 | 14.63 | 289.34 | 24.07 |
| EditAR | 39.43 | 21.32 | 117.15 | 24.87 |
| **SCAR** | **30.98** | **22.59** | **105.09** | **26.07** |

**SCAR reduces structure distance by 21.4%** and improves CLIP similarity by 4.8% over EditAR.

---

## Key Ablations

### Compression Ratio (Table 5)

| Compression | HED FID↓ | HED SSIM↑ | Depth FID↓ |
|-------------|----------|-----------|------------|
| 1× (no compression) | 9.29 | 81.95 | 14.61 |
| **4× (SCAR)** | **9.43** | **81.76** | **14.70** |
| 16× (too aggressive) | 10.74 | 79.66 | 16.10 |

**4× compression maintains quality** while significantly reducing cost.

### Image Encoder (Table 6)

| Encoder | Params | HED FID↓ | Depth FID↓ |
|---------|--------|----------|------------|
| ViT-S | 21.8M | 12.37 | 17.56 |
| SAM-B | 89.6M | 10.20 | 15.57 |
| CLIP-B | 149.6M | 13.78 | 18.35 |
| **DINOv2-B** | **86.6M** | **9.43** | **14.70** |

**DINOv2 provides best semantic features** for compression.

### Alignment Weight δ (Figure 7, 8)

| δ | Instruction Fidelity | Visual Quality |
|---|---------------------|----------------|
| 0.0 | Poor (misses instructions) | Good |
| 0.3 | Fair | Good |
| **0.5** | **Excellent** | **Excellent** |
| 1.0 | Excellent | Poor (artifacts, distortion) |

**δ = 0.5 provides optimal balance.**

---

## Implementation Details

### Training

- **Dataset:** ImageNet-256 (C2I), MultiGen-20M (T2I), SEED-Edit-Unsplash (editing)
- **Image Encoder:** DINOv2-B (frozen)
- **Text Encoder:** T5 (for T2I/editing)
- **AR Backbones:** VAR (next-set), LlamaGen (next-token)
- **Epochs:** 10 (C2I VAR), 20 (C2I LlamaGen), 4 (T2I), 2 (editing)
- **Hardware:** 8× NVIDIA H20 GPUs
- **Resolution:** 256×256 (C2I), 512×512 (T2I/editing)

### Architecture

- **Compression module:** Parallel downsampling (Conv + Residual)
- **Upsampling (training only):** Pixel shuffle
- **Attention mask:** Bidirectional between prefix and text, causal for VQ tokens
- **Loss weights:** L_CE + 0.5 * L_align

---

## Adaptation to Text/Document Compression

### Our Implementation

We adapt SCAR's concepts from **image AR models** to **text/document compression for LLMs**:

| SCAR (Images) | Our Adaptation (Text) |
|---------------|----------------------|
| DINOv2 vision encoder | Sentence-Transformers text encoder |
| 1024 → 256 vision tokens | 384D → 96D embeddings |
| Image semantic features | Document semantic graph |
| AR model hidden states | Retrieved node embeddings |
| Target image alignment | Query-document alignment |
| Controllable generation | Adaptive retrieval |

### Key Modules

1. **`LearnableSemanticCompressor`** (SCAR Section 3.2)
   - Compresses sentence embeddings 4× (384D → 96D)
   - Parallel compression branches (like SCAR's Eq. 3)
   - Reconstruction loss for semantic preservation (like SCAR's Eq. 4)

2. **`SemanticAlignmentModule`** (SCAR Section 3.3)
   - Aligns retrieved documents with query semantics
   - L2 alignment loss (like SCAR's Eq. 8)
   - Learnable projection for better alignment

3. **`SCAREnhancedCompressor`**
   - Integration layer for existing `SemanticCompressor`
   - Alignment-guided search
   - Adaptive fidelity based on alignment scores

### Training

- **Compressor Training:** Self-supervised reconstruction (L_pres)
- **Alignment Training:** Contrastive learning on (query, relevant_doc, irrelevant_doc) triplets
- **Framework:** PyTorch with AdamW optimizer
- **Utilities:** `training_utils.py` with `SCARTrainer` class

---

## References

### SCAR Paper

```bibtex
@article{jin2025scar,
  title={Semantic Context Matters: Improving Conditioning for Autoregressive Models},
  author={Jin, Dongyang and Xu, Ryan and Zeng, Jianhao and Lan, Rui and Bai, Yancheng and Sun, Lei and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2511.14063},
  year={2025}
}
```

### Related Work Mentioned in SCAR

- **AR Models:** LlamaGen, VAR, AiM
- **Diffusion Models:** ControlNet, ControlNet++, UniControl, T2I-Adapter
- **Vision Foundation Models:** DINOv2, CLIP, SAM
- **Editing Methods:** InstructPix2Pix, EditAR, ControlAR

---

## Future Directions

From SCAR paper Section 5:

1. **Scaling:** Larger parameter sizes following AR scaling laws
2. **Multimodal:** Extension to unified multimodal models (UMM)
3. **Video Editing:** Temporal extension of SCAR concepts

From our implementation:

1. **Real Training Data:** Train on actual document corpora instead of synthetic data
2. **Multi-Domain:** Adapt to different document types (code, legal, scientific)
3. **Online Learning:** Update compression/alignment models as documents are ingested
4. **Hybrid Compression:** Combine SCAR with existing graph-based methods

---

## Code Structure

```
src/
├── scar_compressor.py      # Main SCAR modules
│   ├── LearnableSemanticCompressor
│   ├── SemanticAlignmentModule
│   └── SCAREnhancedCompressor
├── training_utils.py       # Training infrastructure
│   ├── SCARTrainer
│   ├── SemanticCompressionDataset
│   └── AlignmentDataset
├── semantic_compressor.py  # Base compressor (existing)
└── adaptive_rate_allocator.py  # Adaptive allocation (existing)

examples/
└── scar_demo.py            # Full demonstration

docs/
└── SCAR_PAPER_SUMMARY.md   # This file
```

---

## Conclusion

SCAR demonstrates that **semantic context is crucial** for effective conditioning in autoregressive models. By using:

1. **Compressed semantic prefilling** (efficient, high-level features)
2. **Semantic alignment guidance** (dense, in-context learning)

SCAR achieves state-of-the-art results in both controllable generation and instruction editing, while being compatible with multiple AR paradigms.

Our adaptation brings these concepts to **text/document compression**, enabling:
- More efficient semantic compression
- Better query-document alignment
- Adaptive retrieval based on semantic relevance

The principles are universal: **compress semantically, align densely, adapt dynamically**.
