# Survey: VLM-Based Image/Document Retrieval Beyond ColPali (2024--2026)

**Compiled: 2026-03-28**

This survey covers the latest research on Vision-Language Model (VLM) based document/image retrieval systems, with special attention to models that combine retrieval with spatial grounding (knowing WHERE information resides within a document or image).

---

## Table of Contents

1. [Foundational Late-Interaction Retrieval](#1-foundational-late-interaction-retrieval)
2. [Efficiency & Compression for Multi-Vector Retrieval](#2-efficiency--compression-for-multi-vector-retrieval)
3. [Vision-Based RAG Pipelines](#3-vision-based-rag-pipelines)
4. [Retrieval + Spatial Grounding](#4-retrieval--spatial-grounding)
5. [Universal Multimodal Retrieval](#5-universal-multimodal-retrieval)
6. [Grounding-Capable VLMs (Complementary)](#6-grounding-capable-vlms-complementary)
7. [Surveys & Benchmarks](#7-surveys--benchmarks)
8. [Gap Analysis: Retrieval + Location Awareness](#8-gap-analysis-retrieval--location-awareness)

---

## 1. Foundational Late-Interaction Retrieval

### 1.1 ColPali: Efficient Document Retrieval with Vision Language Models

- **Authors:** Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Celine Hudelot, Pierre Colombo
- **Venue/Year:** ICLR 2025 (arXiv: July 2024)
- **Key Contribution:** Introduces the paradigm of using VLMs with late-interaction (ColBERT-style) scoring for document page retrieval directly from images, eliminating the need for OCR pipelines. Establishes the ViDoRe benchmark.
- **Method:** Uses PaliGemma-3B (SigLIP vision encoder + Gemma 2B LLM) to encode document page images into ~1,024 patch embeddings. Queries are encoded by the same model's text pathway. Relevance is computed via MaxSim late interaction between query token embeddings and document patch embeddings.
- **Limitations:** Operates at page-level granularity only -- no sub-page localization. Storage-heavy due to ~1,024 128-dim vectors per page. Limited to the visual/linguistic capacity of PaliGemma-3B.
- **Retrieval + Location Awareness:** ColPali's per-patch similarity scores implicitly encode spatial information. The original paper demonstrated interpretability heatmaps showing which image patches contribute most to retrieval decisions. However, it does NOT output bounding boxes or explicit region coordinates.

### 1.2 ColQwen2: Late-Interaction Retrieval with Qwen2-VL

- **Authors:** Same team (Illuin Tech / ViDoRe collaboration)
- **Venue/Year:** Released alongside ColPali v1.2 (late 2024)
- **Key Contribution:** Replaces PaliGemma with Qwen2-VL-2B as the backbone, yielding +5.3 nDCG@5 over ColPali. Supports dynamic image resolutions (no aspect-ratio distortion) and up to 768 image patches.
- **Method:** Same late-interaction framework as ColPali but with a stronger VLM backbone. Handles variable aspect ratios natively.
- **Limitations:** Same page-level granularity limitation. Larger patch counts increase memory requirements.
- **Retrieval + Location Awareness:** Same implicit spatial awareness through patch-level similarity as ColPali, but no explicit grounding output.

### 1.3 Nemotron ColEmbed V2 (NVIDIA)

- **Authors:** Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Oliver Holworthy, Benedikt Schifferer, Zhiding Yu, Yauhen Babakhin, Radek Osmulski, et al.
- **Venue/Year:** arXiv, February 2026
- **Key Contribution:** State-of-the-art on ViDoRe V3 leaderboard (NDCG@10 = 63.42 for 8B model). Releases three model variants (3B/4B/8B) built on NVIDIA Eagle 2 and Qwen3-VL backbones.
- **Method:** Employs cluster-based sampling, hard-negative mining, bidirectional attention, late interaction, and model merging. Three variants based on different VLM backbones.
- **Limitations:** Large model sizes (3B--8B). Same page-level retrieval paradigm. Storage overhead of multi-vector embeddings persists.
- **Retrieval + Location Awareness:** Inherits patch-level spatial information from the late-interaction mechanism but does not produce explicit spatial outputs.

### 1.4 CausalEmbed: Auto-Regressive Multi-Vector Generation for Visual Document Embedding

- **Authors:** Jiahao Huo, Yu Huang, Yibo Yan, Ye Pan, Yi Cao, Mingdong Ou, Philip S. Yu, Xuming Hu
- **Venue/Year:** arXiv, January 2026
- **Key Contribution:** Reframes multi-vector embedding creation as an auto-regressive generation process rather than direct patch extraction. Achieves 30--155x reduction in token count while maintaining competitive performance.
- **Method:** Fine-tunes a pre-trained MLLM to sequentially synthesize compact latent vectors, using iterative margin loss during contrastive training. Enables flexible test-time scaling of representation size.
- **Limitations:** Auto-regressive generation adds inference latency compared to single-pass patch extraction. Novel paradigm with less ecosystem support.
- **Retrieval + Location Awareness:** The generated vectors are abstract latent representations -- they lose the direct spatial correspondence that patch-level embeddings have. Spatial grounding would require additional mapping.

### 1.5 jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval

- **Authors:** Jina AI team
- **Venue/Year:** ACL MRL Workshop 2025 (arXiv: June 2025)
- **Key Contribution:** A 3.8B parameter model supporting BOTH single-vector and multi-vector (late-interaction) embeddings from a single backbone, for text, images, and cross-modal search. Introduces the JVDR benchmark for visually rich document retrieval.
- **Method:** Built on Qwen2.5-VL-3B-Instruct. Uses three task-specific LoRA adapters (60M each) for retrieval, text-matching, and code tasks. Single-vector mode produces 2048-dim embeddings (truncatable to 128 via MRL); multi-vector mode produces 128-dim per-token embeddings for late interaction.
- **Limitations:** Dual-mode adds architectural complexity. Multi-vector mode still incurs storage overhead.
- **Retrieval + Location Awareness:** Multi-vector mode preserves patch-level spatial correspondence. No explicit grounding or bounding box output.

---

## 2. Efficiency & Compression for Multi-Vector Retrieval

### 2.1 HPC-ColPali: Hierarchical Patch Compression for ColPali

- **Authors:** (Not fully specified in search results)
- **Venue/Year:** SCITEPRESS 2025 / arXiv June 2025
- **Key Contribution:** Addresses the storage bottleneck of ColPali's multi-vector embeddings through a three-pronged compression strategy.
- **Method:** (1) K-Means quantization compresses patch embeddings to 1-byte centroid indices (up to 32x storage reduction); (2) attention-guided dynamic pruning retains only top-p% most salient patches (up to 60% computation reduction with <2% nDCG@10 loss); (3) optional binary encoding for Hamming-distance search.
- **Limitations:** Lossy compression; aggressive pruning degrades performance. Quantization clusters may merge semantically distinct patches.
- **Retrieval + Location Awareness:** Attention-guided pruning reveals which patches are most informative, providing implicit spatial signal. Pruned patches retain their spatial coordinates.

### 2.2 DocPruner: Storage-Efficient Multi-Vector Visual Document Retrieval

- **Authors:** (arXiv: 2509.23883)
- **Venue/Year:** arXiv, September 2025
- **Key Contribution:** First framework using adaptive patch-level embedding pruning for VDR. Achieves 50--60% storage reduction with negligible retrieval degradation.
- **Method:** Leverages per-document attention score distributions to compute a document-specific statistical threshold for pruning. Dynamically adjusts pruning ratio based on information density and document complexity.
- **Limitations:** Pruning is document-specific, complicating batch processing. Performance degrades sharply at compression rates above 60%.
- **Retrieval + Location Awareness:** Retained patches preserve their spatial position, allowing post-hoc spatial reasoning over the pruned set.

### 2.3 Sculpting the Vector Space: Prune-then-Merge Framework

- **Authors:** (arXiv: 2602.19549)
- **Venue/Year:** arXiv, February 2026
- **Key Contribution:** Two-stage approach that extends the near-lossless compression frontier from 50--60% (DocPruner) to 60--70%, and avoids the sharp performance cliff at high compression rates (80%+).
- **Method:** Stage 1 (Prune): adaptive pruning filters out low-information patches. Stage 2 (Merge): hierarchical merging compresses the remaining set by summarizing semantic content without noise-induced feature dilution. Evaluated with ColQwen2.5, ColNomic, and Jina-v4 across 29 VDR datasets.
- **Limitations:** Two-stage pipeline adds complexity. Merging stage may lose fine-grained spatial distinctions between nearby patches.
- **Retrieval + Location Awareness:** The pruning stage retains spatially informative patches, but the merging stage may aggregate spatial information, reducing localization granularity.

### 2.4 ColParse: Beyond the Grid -- Layout-Informed Multi-Vector Retrieval

- **Authors:** Yibo Yan, Mingdong Ou, Yi Cao, Xin Zou, Shuliang Liu, Jiahao Huo, Yu Huang, James Kwok, Xuming Hu
- **Venue/Year:** arXiv, March 2026
- **Key Contribution:** Achieves >95% storage reduction while IMPROVING performance by replacing the uniform patch grid with layout-aware sub-image embeddings from a document parsing model.
- **Method:** Uses a document parsing model to segment pages into semantically meaningful regions (paragraphs, tables, figures, etc.). Each region is encoded into a compact sub-image embedding, fused with a global page-level vector to create a structurally-aware multi-vector representation.
- **Limitations:** Depends on quality of the document parsing model. Adds a parsing step to the indexing pipeline.
- **Retrieval + Location Awareness:** **HIGH RELEVANCE.** Because it uses document parsing to identify semantically meaningful regions with known spatial extents, the representation inherently carries layout/spatial information. This is one of the closest papers to the retrieval + location awareness goal.

---

## 3. Vision-Based RAG Pipelines

### 3.1 VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents

- **Authors:** Shi Yu, Chaoyue Tang, et al. (OpenBMB)
- **Venue/Year:** ICLR 2025 (arXiv: October 2024)
- **Key Contribution:** First end-to-end VLM-based RAG pipeline that embeds documents directly as images for retrieval, then feeds retrieved page images to a VLM generator. Achieves 20--40% end-to-end gain over text-based RAG.
- **Method:** VisRAG-Ret uses a VLM to encode document page screenshots into dense embeddings for retrieval. VisRAG-Gen uses a generative VLM to produce answers from retrieved page images, preserving all visual information (layout, figures, tables).
- **Limitations:** Page-level retrieval granularity. Does not identify specific regions within retrieved pages. High computational cost for VLM-based generation on multiple page images.
- **Retrieval + Location Awareness:** Retrieves at page level. No sub-page grounding.

### 3.2 VisRAG 2.0 (EVisRAG): Evidence-Guided Multi-Image Reasoning

- **Authors:** OpenBMB team
- **Venue/Year:** arXiv, October 2025
- **Key Contribution:** Extends VisRAG with per-image evidence recording and aggregation. Introduces RS-GRPO (Reward-Scoped Group Relative Policy Optimization) to jointly optimize visual perception and reasoning.
- **Method:** After retrieving images, the model first observes each image and records per-image evidence, then derives the final answer from aggregated evidence. RS-GRPO binds fine-grained rewards to scope-specific tokens.
- **Limitations:** Still operates at page-level retrieval. Evidence is recorded per-image, not per-region.
- **Retrieval + Location Awareness:** Evidence recording provides implicit grounding (which page has relevant information), but does not localize within pages.

### 3.3 VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents

- **Authors:** Ryota Tanaka et al.
- **Venue/Year:** CVPR 2025 (arXiv: April 2025)
- **Key Contribution:** Proposes self-supervised pre-training tasks that adapt VLMs for retrieval by compressing visual information into dense token representations while aligning with textual content. Introduces OpenDocVQA benchmark.
- **Method:** VDocRetriever encodes document images into dense embeddings. VDocGenerator processes retrieved document images to generate answers. Novel pre-training aligns visual tokens with document text without requiring labeled data.
- **Limitations:** Page-level retrieval. Self-supervised pre-training adds training overhead.
- **Retrieval + Location Awareness:** Retrieves whole pages. No spatial grounding output.

### 3.4 ViDoRAG: Visual Document RAG via Dynamic Iterative Reasoning Agents

- **Authors:** Alibaba NLP team
- **Venue/Year:** EMNLP 2025 (arXiv: February 2025)
- **Key Contribution:** Multi-agent RAG framework with iterative exploration, summarization, and reflection for complex reasoning over visual documents. Introduces the ViDoSeek benchmark. Outperforms existing methods by >10%.
- **Method:** Uses GMM-based hybrid strategy for multi-modal retrieval. Actor-critic multi-agent paradigm iteratively refines retrieval and reasoning. Enhances noise robustness of generation.
- **Limitations:** Complex multi-agent architecture. High latency due to iterative reasoning. Page-level retrieval.
- **Retrieval + Location Awareness:** The iterative reasoning helps identify relevant information but does not produce spatial coordinates within documents.

### 3.5 RegionRAG: Region-Level Retrieval-Augmented Generation

- **Authors:** (arXiv: 2510.27261)
- **Venue/Year:** arXiv, October 2025
- **Key Contribution:** **Shifts retrieval from document-level to region-level.** Achieves +10.02% R@1 improvement on average while using only 71.42% of visual tokens compared to page-level methods.
- **Method:** During training, uses a hybrid supervision strategy combining labeled bounding box data and pseudo-labels from unlabeled data to identify query-relevant patches. Clusters salient patches via BFS to produce visual region crops. A dual-objective contrastive loss trains the retriever.
- **Limitations:** Requires bounding box annotations (at least partially). BFS-based region detection may produce inconsistent region boundaries. Region quality depends on patch saliency estimation.
- **Retrieval + Location Awareness:** **HIGH RELEVANCE.** Directly retrieves at the region level, meaning each retrieved result corresponds to a spatial region within a document page. This is one of the most relevant papers for retrieval + location awareness.

---

## 4. Retrieval + Spatial Grounding (Core Target Area)

### 4.1 Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation

- **Authors:** Athos Georgiou
- **Venue/Year:** arXiv, December 2025 (revised January 2026)
- **Key Contribution:** **The most directly relevant paper.** Proposes a hybrid architecture that bridges late-interaction multimodal retrieval (ColPali/ColQwen) with OCR-based systems to provide spatially grounded retrieval results with bounding box coordinates.
- **Method:** Uses ColPali/ColQwen-style models to generate patch-level similarity scores, then propagates these scores to OCR-extracted text regions through formalized coordinate mapping between vision transformer patch grids and OCR bounding boxes. Introduces intersection metrics (IoU-weighted) for relevance propagation. Works at inference time with no additional training.
- **Results:** On BBox-DocVQA, ColQwen3-4B with percentile-50 thresholding achieved 59.7% hit rate at IoU@0.5 (84.4% at IoU@0.25). Reduces context tokens by 28.8% vs. all OCR regions and 52.3% vs. full-page image tokens.
- **Limitations:** Depends on OCR quality for region extraction. Only grounds to OCR-detected text regions (not arbitrary visual regions like figures or charts). IoU thresholding is a hyperparameter requiring tuning.
- **Retrieval + Location Awareness:** **HIGHEST RELEVANCE.** This paper directly addresses the retrieval + location awareness problem. It retrieves relevant document pages AND outputs bounding box coordinates for the specific regions within those pages that are most relevant to the query. Open-source implementation called "Snappy."

### 4.2 EaGERS: Spatially Grounded Explanations in VLMs for Document VQA

- **Authors:** (arXiv: 2507.12490)
- **Venue/Year:** arXiv, July 2025
- **Key Contribution:** Training-free pipeline that generates natural language explanations, grounds them to spatial sub-regions via multimodal embedding similarities over a configurable grid, and re-queries the model on masked images.
- **Method:** (1) VLM generates a spatial natural language explanation from image + question; (2) Image is segmented into an m x n grid; (3) Embeddings of the explanation and each sub-region are obtained using BLIP, CLIP, and ALIGN; (4) Majority voting selects the most relevant regions; (5) Image is masked to retain only selected regions for final answer generation.
- **Limitations:** Grid-based segmentation is coarse and may not align with document layout. Multi-model pipeline (needs BLIP + CLIP + ALIGN + VLM). Not designed for retrieval -- focused on VQA.
- **Retrieval + Location Awareness:** Provides spatial grounding within a single document page but is not a retrieval system. Could potentially be adapted as a post-retrieval grounding step.

### 4.3 BBox-DocVQA: Bounding-Box-Grounded Dataset for Document VQA

- **Authors:** (arXiv: 2511.15090)
- **Venue/Year:** arXiv, November 2025
- **Key Contribution:** A large-scale benchmark (3.6K documents, 32K QA pairs) where every answer is grounded to explicit bounding boxes. Reveals that state-of-the-art VLMs (GPT, Qwen2.5-VL, InternVL) still struggle with spatial grounding.
- **Method:** Automated "Segment-Judge-Generate" pipeline: segment model for region segmentation, VLM for semantic judgment, advanced VLM for QA generation, followed by human verification. Covers single/multi-region and single/multi-page scenarios.
- **Limitations:** Dataset only (not a model). Limited to academic papers from arXiv. Bounding boxes are for text/table/figure regions, not arbitrary visual elements.
- **Retrieval + Location Awareness:** **HIGH RELEVANCE** as a benchmark. Provides the evaluation framework needed for testing retrieval + location awareness systems. Fine-tuning on BBox-DocVQA improves both bounding box localization and answer generation.

---

## 5. Universal Multimodal Retrieval

### 5.1 MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs

- **Authors:** NVIDIA team
- **Venue/Year:** ICLR 2025 (arXiv: November 2024)
- **Key Contribution:** MLLM fine-tuned as a bi-encoder retriever across 10 datasets with 16 retrieval tasks. State-of-the-art on M-BEIR (multimodal) AND MTEB (text-only) retrieval benchmarks simultaneously.
- **Method:** Fine-tunes an MLLM as a dense bi-encoder retriever. Proposes modality-aware hard negative mining to mitigate modality bias. Also explores zero-shot MLLM reranking via prompting.
- **Limitations:** Single-vector embeddings lack fine-grained patch-level interaction. No spatial grounding. Dense retrieval only.
- **Retrieval + Location Awareness:** Retrieves whole documents/images. No spatial localization.

### 5.2 GME: Bridging Modalities -- Improving Universal Multimodal Retrieval by MLLMs

- **Authors:** Zhang et al.
- **Venue/Year:** CVPR 2025
- **Key Contribution:** Develops a training data synthesis pipeline to construct large-scale fused-modal training data, and builds General Multimodal Embedder (GME) for universal multimodal retrieval. Constructs UMRB benchmark.
- **Method:** MLLM-based dense retriever with synthetic multimodal training data that addresses modality imbalance in existing training sets. Trains on diverse query-candidate pairs spanning text, image, and fused modalities.
- **Limitations:** Single-vector dense retrieval. No late-interaction or spatial awareness. Synthetic data quality may vary.
- **Retrieval + Location Awareness:** Retrieves whole items. No spatial localization.

### 5.3 UniIR: Training and Benchmarking Universal Multimodal Information Retrievers

- **Authors:** TIGER Lab
- **Venue/Year:** ECCV 2024
- **Key Contribution:** Single retrieval system jointly trained on 10 multimodal-IR datasets for 8 distinct retrieval tasks. Introduces M-BEIR benchmark. Demonstrates instruction tuning as key to cross-task generalization.
- **Method:** Instruction-guided multimodal retriever built on CLIP/BLIP2 backbones. User instructions specify the desired retrieval task. Multi-task training enables zero-shot generalization to new tasks.
- **Limitations:** Based on older CLIP/BLIP2 backbones (not VLMs). Single-vector embeddings. No document-specific features (layout awareness, etc.).
- **Retrieval + Location Awareness:** Retrieves whole items. No spatial localization.

### 5.4 GENIUS: A Generative Framework for Universal Multimodal Search

- **Authors:** Sung-Yeon Kim et al. (Amazon)
- **Venue/Year:** CVPR 2025
- **Key Contribution:** Generative retrieval approach for multimodal search with >99% storage reduction over embedding-based methods. Introduces modality-decoupled semantic quantization.
- **Method:** Transforms multimodal data into discrete IDs via residual quantization (first level encodes modality, subsequent levels capture semantics). An autoregressive decoder generates these IDs at query time. Query augmentation interpolates between query and target.
- **Limitations:** Generative retrieval paradigm is less mature than embedding-based. Discrete IDs may lose fine-grained information. No spatial awareness.
- **Retrieval + Location Awareness:** Retrieves whole items. No spatial localization. The discrete ID paradigm is fundamentally at odds with spatial grounding.

### 5.5 Qwen3-VL-Embedding & Qwen3-VL-Reranker

- **Authors:** Qwen Team, Alibaba Group
- **Venue/Year:** arXiv, January 2026
- **Key Contribution:** Unified framework for multimodal retrieval and ranking, achieving first place on MMEB-V2 (score: 77.8 for 8B model). Handles text, images, screenshots, and video.
- **Method:** Built on Qwen3-VL foundation models. Dense single-vector embedding for retrieval, with a separate reranker model for candidate refinement. Trained on JinaVDR and ViDoRe v3 datasets.
- **Limitations:** Single-vector embeddings (not multi-vector/late-interaction). Separate retriever and reranker models.
- **Retrieval + Location Awareness:** Retrieves whole documents. No spatial localization, though the underlying Qwen3-VL has grounding capabilities that are not exposed through the embedding interface.

---

## 6. Grounding-Capable VLMs (Complementary -- Not Retrieval-Native)

These models do spatial grounding but are NOT retrieval systems. They are relevant as potential components in a retrieval + grounding pipeline.

### 6.1 SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models

- **Authors:** An-Chieh Cheng, Hongxu Yin, Yang Fu, et al. (UC San Diego, NVIDIA)
- **Venue/Year:** NeurIPS 2024
- **Key Contribution:** Enhances VLMs' spatial perception through 3D scene graph-based data curation and a depth-integration plugin module. Can perceive relative directions and distances between user-specified regions.
- **Limitations:** Requires user-specified region proposals as input. Not a retrieval system.
- **Retrieval + Location Awareness:** Strong spatial reasoning but must be paired with a retrieval system.

### 6.2 Groma: Localized Visual Tokenization for Grounding MLLMs

- **Authors:** Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, Xiaojuan Qi (HKU, ByteDance)
- **Venue/Year:** ECCV 2024
- **Key Contribution:** Decomposes images into regions of interest via a region proposer, encodes them into region tokens, and grounds textual output by referring to these tokens. Avoids explicit coordinate regression.
- **Limitations:** Not a retrieval system. Region proposer adds inference overhead.
- **Retrieval + Location Awareness:** Excellent grounding capability (region-level). Could serve as a post-retrieval grounding module.

### 6.3 DocCogito: Layout Cognition and Step-Level Grounded Reasoning for Document Understanding

- **Authors:** (arXiv: 2603.07494)
- **Venue/Year:** arXiv, March 2026
- **Key Contribution:** OCR-free document understanding framework that couples global layout cognition with step-level region-grounded reasoning. SOTA on 4 of 6 benchmarks.
- **Method:** Lightweight layout tower produces global layout prior tokens. Visual-Semantic Chain (VSC) supervises intermediate reasoning aligned with evidence regions. Progressive training: layout perception pretraining, VSC-guided cold start, rejection sampling, and GRPO.
- **Limitations:** Not a retrieval system. Focused on document understanding/QA.
- **Retrieval + Location Awareness:** Strong region-grounded reasoning within documents. Could be paired with a retrieval system for end-to-end retrieval + grounding.

### 6.4 LayoutLLM: Layout Instruction Tuning for Document Understanding

- **Authors:** Luo et al. (Alibaba Research)
- **Venue/Year:** CVPR 2024
- **Key Contribution:** Layout-aware pre-training at document/region/segment levels with LayoutCoT (Layout Chain-of-Thought) to focus on question-relevant regions.
- **Limitations:** Requires OCR + layout input. Not a retrieval system.
- **Retrieval + Location Awareness:** LayoutCoT enables region-level focus, relevant for post-retrieval grounding.

---

## 7. Surveys & Benchmarks

### 7.1 Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey

- **Venue/Year:** AACL-IJCNLP 2025 (arXiv: January 2025)
- **Key Insight:** Organizes MLLMs into three roles for VRD retrieval: (1) Modality-Unifying Captioners, (2) Multimodal Embedders, and (3) End-to-End Representers. Compares retrieval granularity, information fidelity, latency, and compatibility with reranking and grounding.

### 7.2 DSE: Unifying Multimodal Retrieval via Document Screenshot Embedding

- **Authors:** Xueguang Ma et al.
- **Venue/Year:** EMNLP 2024 (arXiv: June 2024)
- **Key Contribution:** Early demonstration that directly encoding document screenshots with a VLM (Phi-3-vision, 4B) produces effective dense retrieval embeddings. Outperforms BM25 by 17 points on Wiki-SS.
- **Limitations:** Single-vector embeddings (no late interaction). Vulnerable to pixel poisoning attacks (shown by follow-up work in January 2025).

### 7.3 FLMR / PreFLMR: Fine-Grained Late-Interaction Multi-Modal Retrieval

- **Authors:** Weizhe Lin, Jingbiao Mei, Jinghong Chen, Bill Byrne
- **Venue/Year:** FLMR: NeurIPS 2023; PreFLMR: ACL 2024
- **Key Contribution:** Precursor to ColPali-style approaches. Incorporates token-level visual and textual features into multi-dimensional embeddings with cross-modality late interaction for RA-VQA. PreFLMR introduces the M2KR benchmark for general-purpose multimodal retrievers.

---

## 8. Gap Analysis: Retrieval + Location Awareness

### Current State of the Art

The goal of combining **retrieval** (finding the right document/page) with **location awareness** (knowing WHERE within that document/page the answer resides) is addressed to varying degrees by the papers surveyed:

| Paper | Retrieval | Sub-Page Localization | Bounding Boxes | Training-Free |
|-------|-----------|----------------------|----------------|---------------|
| ColPali/ColQwen2 | Yes (page-level) | Implicit (heatmaps) | No | N/A |
| Patch-to-Region (Georgiou) | Yes (page-level) | Yes (OCR regions) | Yes | Yes |
| RegionRAG | Yes (region-level) | Yes (patch clusters) | Partial | No |
| ColParse | Yes (region-level) | Yes (parsed regions) | Implicit | No |
| EaGERS | No (VQA only) | Yes (grid cells) | No | Yes |
| BBox-DocVQA | Benchmark only | Benchmark only | Yes | N/A |
| DocCogito | No (understanding) | Yes (VSC regions) | Yes | No |
| Groma | No (understanding) | Yes (region tokens) | Yes | No |

### Key Gap

No single system currently provides:
1. Large-scale retrieval over a document corpus (finding the right page among millions)
2. Fine-grained region localization with bounding box coordinates within the retrieved page
3. End-to-end training without requiring OCR

**The closest approaches are:**

- **Patch-to-Region Relevance Propagation** (Georgiou, 2025): Combines ColPali/ColQwen retrieval with OCR-based region grounding at inference time. Most complete current solution but depends on OCR.
- **RegionRAG** (2025): Shifts retrieval to region level with patch clustering, but uses coarse BFS-based region detection and requires bounding box supervision during training.
- **ColParse** (2026): Uses document parsing for layout-informed retrieval with implicit region awareness and >95% storage reduction, but does not explicitly output bounding boxes.

### Recommended Architecture for Retrieval + Location Awareness

Based on this survey, the most promising approach would combine:
1. **Retrieval stage:** ColQwen2/Nemotron ColEmbed V2 for page-level retrieval
2. **Localization stage:** Patch-to-Region Relevance Propagation (Georgiou) to map patch similarities to OCR regions with bounding boxes
3. **OR** ColParse-style layout parsing for region-level retrieval with structural awareness
4. **Optional grounding refinement:** DocCogito or Groma for step-level grounded reasoning on the retrieved regions

---

## Sources

- [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [ColPali GitHub (illuin-tech)](https://github.com/illuin-tech/colpali)
- [VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [VisRAG 2.0 (arXiv:2510.09733)](https://arxiv.org/abs/2510.09733)
- [DSE (arXiv:2406.11251)](https://arxiv.org/abs/2406.11251)
- [UniIR (ECCV 2024)](https://dl.acm.org/doi/10.1007/978-3-031-73021-4_23)
- [MM-Embed (arXiv:2411.02571)](https://arxiv.org/abs/2411.02571)
- [GME / Bridging Modalities (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Modalities_Improving_Universal_Multimodal_Retrieval_by_Multimodal_Large_Language_CVPR_2025_paper.pdf)
- [GENIUS (arXiv:2503.19868)](https://arxiv.org/abs/2503.19868)
- [VDocRAG (arXiv:2504.09795)](https://arxiv.org/abs/2504.09795)
- [ViDoRAG (arXiv:2502.18017)](https://arxiv.org/abs/2502.18017)
- [RegionRAG (arXiv:2510.27261)](https://arxiv.org/abs/2510.27261)
- [Spatially-Grounded Document Retrieval (arXiv:2512.02660)](https://arxiv.org/abs/2512.02660)
- [HPC-ColPali (arXiv:2506.21601)](https://arxiv.org/abs/2506.21601)
- [DocPruner (arXiv:2509.23883)](https://arxiv.org/abs/2509.23883)
- [Prune-then-Merge (arXiv:2602.19549)](https://arxiv.org/abs/2602.19549)
- [ColParse / Beyond the Grid (arXiv:2603.01666)](https://arxiv.org/abs/2603.01666)
- [CausalEmbed (arXiv:2601.21262)](https://arxiv.org/abs/2601.21262)
- [Nemotron ColEmbed V2 (arXiv:2602.03992)](https://arxiv.org/abs/2602.03992)
- [jina-embeddings-v4 (arXiv:2506.18902)](https://arxiv.org/abs/2506.18902)
- [Qwen3-VL-Embedding (arXiv:2601.04720)](https://arxiv.org/abs/2601.04720)
- [BBox-DocVQA (arXiv:2511.15090)](https://arxiv.org/abs/2511.15090)
- [EaGERS (arXiv:2507.12490)](https://arxiv.org/abs/2507.12490)
- [SpatialRGPT (arXiv:2406.01584)](https://arxiv.org/abs/2406.01584)
- [Groma (arXiv:2404.13013)](https://arxiv.org/abs/2404.13013)
- [DocCogito (arXiv:2603.07494)](https://arxiv.org/abs/2603.07494)
- [LayoutLLM (CVPR 2024)](https://arxiv.org/abs/2404.05225)
- [PreFLMR (ACL 2024)](https://aclanthology.org/2024.acl-long.289/)
- [FLMR (NeurIPS 2023)](https://arxiv.org/abs/2309.17133)
- [Roles of MLLMs Survey (arXiv:2601.03262)](https://arxiv.org/abs/2601.03262)
- [Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
