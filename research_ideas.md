# Research Ideas: Retrieval with Spatial Localization in Visual Documents

## Literature Landscape Summary

The core research question is: **How can we build image/document retrieval systems that not only find the right image but also provide precise location information (bounding boxes, regions, coordinates) about WHERE the relevant information is within the retrieved image?**

### Survey 1: VLM-Based Image/Document Retrieval
Key papers surveyed: ColPali (ICLR 2025), ColQwen2, DSE (EMNLP 2024), VisRAG (ICLR 2025), UniIR (ECCV 2024), MM-Embed (ICLR 2025), GENIUS (CVPR 2025), VDocRAG (CVPR 2025), ViDoRAG (EMNLP 2025), FLMR/PreFLMR (NeurIPS 2023 / ACL 2024), RegionRAG, DocPruner, HPC-ColPali, CausalEmbed, Jina-ColBERT-v2, ModernVBERT, BBox-DocVQA, VISA (ACL 2025), Bridging Modalities / GME (CVPR 2025), M3DocRAG, Nemotron ColEmbed V2, Qwen3-VL-Embedding.

### Survey 2: Visual Grounding and Localization
Key papers surveyed: Ferret/Ferret-v2 (ICLR 2024 / COLM 2024), Florence-2 (CVPR 2024), Kosmos-2, Shikra, Qwen-VL series (up to Qwen3-VL), Grounding DINO 1.5 (ECCV 2024), GLaMM (CVPR 2024), LLaVA-Grounding (ECCV 2024), SpatialRGPT (NeurIPS 2024), GroundingGPT (ACL 2024), CogVLM, SPHINX-X, VGent, GroundSight, mPLUG-DocOwl, SAM 2.

### Survey 3: Document Understanding and Retrieval
Key papers surveyed: ColPali family, ViDoRe V1-V3 benchmarks, LayoutLLM (CVPR 2024), DocLLM (ACL 2024), Pix2Struct, UDOP, LayoutLMv3, M3DocRAG, REAL-MM-RAG (ACL 2025), MMDocBench, MMLongBench-Doc, VisDoM (NAACL 2025), UniDoc-Bench, Marten (CVPR 2025), DocLayout-YOLO, NL-DIR (CVPR 2025).

---

## Gap Analysis

After thorough analysis of these three research streams, the following critical gaps emerge:

1. **Retrieval and grounding remain disjoint tasks.** ColPali-style retrievers find the right page but cannot tell you WHERE on the page. Grounding models (Ferret, Florence-2, Grounding DINO) can localize objects but do not operate in a retrieval context over large corpora.

2. **The only bridge is post-hoc and ad-hoc.** The Spatially-Grounded Document Retrieval paper (Georgiou, 2025) maps ColPali patch scores to OCR regions at inference time -- but this requires a separate OCR pipeline, is not trained end-to-end, and achieves only 59.7% hit rate at IoU@0.5.

3. **Region-level retrieval is emerging but immature.** RegionRAG proposes region-level retrieval units but requires separate bounding box supervision and a complex dual-objective loss. No model jointly learns to retrieve and localize in a single pass.

4. **No benchmark jointly evaluates retrieval + localization.** ViDoRe measures page-level nDCG. BBox-DocVQA measures grounding on single pages. No benchmark scores a system on "given a query and 10,000 pages, retrieve the right page AND point to the right region."

5. **Token-level patch embeddings are rich but underexploited for spatial reasoning.** ColPali produces 1024 patch embeddings per page with spatial correspondence to image regions, but this spatial structure is used only for late-interaction scoring, not for localization.

6. **Efficiency is a concern.** Multi-vector representations create massive storage (1024 x 128-dim per page). Pruning approaches (DocPruner, HPC-ColPali) reduce storage but may destroy spatially informative patches.

7. **Document grounding models lack retrieval awareness.** Models like Marten, BBox-DocVQA baselines, and VISA operate on already-retrieved documents. They do not jointly optimize for finding the relevant document AND localizing within it.

---

## Research Ideas

### Idea 1: ColPali-Ground -- End-to-End Document Retrieval with Patch-Level Spatial Grounding

**One-line summary:** Extend ColPali's late-interaction framework to simultaneously retrieve relevant document pages and output bounding boxes over the most relevant regions, trained end-to-end with a joint retrieval-grounding objective.

**Motivation:** ColPali generates spatially-structured patch embeddings that implicitly encode WHERE information is on a page. The late-interaction score (MaxSim) between each query token and every document patch already computes a soft spatial attention map. Yet this spatial signal is discarded after scoring -- only the aggregate page-level score is used. This is wasteful: the patch-level similarities contain enough information to localize relevant regions without any additional model or OCR pipeline. The post-hoc approach of Georgiou (2025) demonstrated that propagating patch scores to OCR regions can achieve localization, but this was an inference-time hack, not a trained capability. No existing work trains a ColPali-style model to be explicitly aware that its patch scores will be used for both retrieval and localization.

**Method:**
- Architecture: Start with ColQwen2 (Qwen2-VL backbone). Add a lightweight localization head -- a small MLP that takes the top-K query-patch similarity vectors (including their 2D positional indices on the patch grid) and regresses bounding box coordinates [x1, y1, x2, y2] in normalized image space.
- Training stage 1 (Retrieval pretraining): Standard ColPali contrastive training on query-page pairs from the ViDoRe training pipeline. This establishes strong patch embeddings.
- Training stage 2 (Joint retrieval + grounding fine-tuning): Use BBox-DocVQA, VISA (Wiki-VISA, Paper-VISA), and a newly curated subset of DocVQA/InfographicVQA with region annotations. The loss is: L = L_retrieval (contrastive) + lambda * L_grounding (smooth-L1 on predicted vs. ground-truth bounding boxes, computed only on the retrieved positive page).
- Key innovation: The grounding loss backpropagates through the same patch embeddings used for retrieval, forcing the model to learn patch representations that are simultaneously good for page-level ranking AND sub-page localization. This is a form of multi-task representation learning where both tasks share the same late-interaction embedding space.
- Inference: For a given query, (1) compute late-interaction scores against all indexed pages to retrieve top-K pages, (2) for each top-K page, run the localization head on the query-patch similarity matrix to predict bounding boxes. Total overhead is minimal since the similarity matrix is already computed during retrieval.

**Expected contribution:** First model to jointly optimize retrieval and grounding in a single multi-vector embedding space. Demonstrates that ColPali-style patch embeddings can serve double duty. Establishes a new evaluation protocol: Retrieval@K + IoU@0.5 as a joint metric.

**Related work differentiation:** Differs from Georgiou (2025) which is inference-time only and requires OCR. Differs from RegionRAG which uses separate region proposals and a different retriever. Differs from VISA which only does attribution on already-retrieved documents without joint training. Differs from Marten which does mask generation for DocVQA but not in a retrieval setting.

**Potential venues:** CVPR 2027, ECCV 2026, NeurIPS 2026, ICLR 2027.

**Feasibility:** Medium. Requires BBox-DocVQA data (available, 32K QA pairs with bounding boxes), VISA datasets (available), and the ColPali training infrastructure (open source). The localization head is a simple addition. Main engineering effort is curating more region-annotated data and designing the joint training schedule. A single A100-80G or H100 should suffice for fine-tuning ColQwen2-2B. Timeline: 3-4 months.

**Risk factors:** The patch grid resolution (e.g., 32x32 for a 1024px image) may limit localization precision for small text regions. The grounding loss could interfere with retrieval performance if lambda is not carefully tuned. Ground-truth bounding boxes in existing datasets may not align perfectly with patch grid boundaries, requiring interpolation.

---

### Idea 2: GroundedViDoRe -- A Benchmark for Joint Document Retrieval and Region Localization

**One-line summary:** Construct the first large-scale benchmark that jointly evaluates document retrieval accuracy AND spatial localization quality, enabling fair comparison of grounded retrieval systems.

**Motivation:** The field currently lacks a benchmark that captures the full retrieval + localization pipeline. ViDoRe (V1/V2/V3) evaluates only page-level nDCG. BBox-DocVQA evaluates grounding on single given pages. VISA provides bounding box annotations but only for a small set of documents. No benchmark asks the fundamental question: "Given 10K+ pages and a natural language query, can you find the right page AND point to the right region?" Without such a benchmark, progress on this combined task cannot be measured or compared.

**Method:**
- Data construction pipeline (inspired by BBox-DocVQA's Segment-Judge-Generate):
  1. Collect a diverse corpus: 50K pages from 5 domains (academic papers from arXiv, financial reports, technical manuals, infographics/slides, medical documents). Use existing corpora (OpenDocVQA, ViDoRe V3 pages, PubLayNet).
  2. Region extraction: Run document layout analysis (DocLayout-YOLO or Florence-2) to detect semantic regions (text blocks, tables, figures, headers, captions) with bounding boxes.
  3. Query generation: For each region, use a strong VLM (GPT-4o or Qwen2.5-VL-72B) to generate natural language questions whose answer requires information from THAT specific region. Generate 3 difficulty levels: (a) keyword-match (easy), (b) paraphrased (medium), (c) reasoning-required (hard), following the REAL-MM-RAG difficulty scheme.
  4. Human verification: Have annotators verify 20% of query-region pairs for correctness, adjust bounding boxes, and filter out ambiguous queries.
  5. Negative pages: Include hard negatives (visually similar pages from the same document, pages with related but non-answering content).
- Evaluation metrics:
  - Retrieval: nDCG@5, Recall@1, Recall@5 (page level)
  - Localization: IoU@0.25, IoU@0.5, IoU@0.75 (bounding box on the correct page)
  - Joint: Grounded-Recall@K = fraction of queries where the correct page is in top-K AND the predicted bounding box has IoU >= threshold with the ground truth region
  - Efficiency: latency per query, index size
- Baseline evaluations: (1) ColPali + post-hoc OCR propagation (Georgiou method), (2) ColPali retrieval + Florence-2 grounding as a pipeline, (3) Retrieve-then-Ground with Qwen2.5-VL, (4) RegionRAG if available.

**Expected contribution:** Fills a fundamental evaluation gap. Provides the community with a standardized way to measure grounded retrieval. The benchmark itself is a significant resource contribution. The baseline analysis will reveal how far current systems are from solving this problem and which components are the bottleneck.

**Related work differentiation:** ViDoRe is page-level only. BBox-DocVQA is single-page grounding. VISA is small-scale and VQA-focused. MMDocBench evaluates fine-grained understanding but not retrieval. GroundedViDoRe is the first to combine large-scale retrieval with per-region localization evaluation.

**Potential venues:** NeurIPS 2026 Datasets and Benchmarks Track, CVPR 2027, ACL 2026, EMNLP 2026.

**Feasibility:** High. The core components are available: document corpora (ViDoRe V3, OpenDocVQA), layout detectors (DocLayout-YOLO, Florence-2), query generation (GPT-4o API), evaluation code (MTEB framework, standard IoU computation). Main cost is human annotation (estimate: 200 annotator-hours for verification of 10K query-region pairs at \~$5K). The benchmark can be built incrementally. Timeline: 4-5 months.

**Risk factors:** Inter-annotator agreement on region boundaries may be low for ambiguous layouts. Queries generated by VLMs may have bias toward easily-described regions (e.g., tables and figures) while underrepresenting text-heavy regions. The benchmark may become stale quickly as models improve. Mitigation: include a hidden test set and a leaderboard with periodic updates.

---

### Idea 3: Spatial Token Distillation -- Teaching Text Retrievers to Inherit Patch-Level Spatial Knowledge from VLMs

**One-line summary:** Distill the spatial localization knowledge embedded in ColPali-style patch embeddings into a lightweight text-only retriever that can simultaneously retrieve documents and predict bounding boxes from OCR+layout features alone, without needing a vision encoder at inference.

**Motivation:** ColPali-style multi-vector retrievers require processing every page through a vision encoder (3B+ parameters) and storing 1024 vectors per page, which is prohibitively expensive for enterprise-scale deployment (millions of pages). Meanwhile, layout-aware text models like DocLLM and LayoutLLM show that bounding box coordinates from OCR can capture substantial spatial information. The key insight is: a VLM-based retriever is the best TEACHER for spatial understanding, but a text+layout model can be a much more efficient STUDENT that retains the spatial awareness at a fraction of the cost. This bridges the efficiency of text-based retrieval with the spatial awareness of vision-based retrieval.

**Method:**
- Teacher model: A trained ColQwen2 model that produces patch embeddings E_v in R^{N_patches x d} for each document page.
- Student model: A layout-aware encoder (ModernVBERT architecture or a smaller LM) that takes as input OCR tokens + their bounding box coordinates (from an existing OCR pipeline) and produces token embeddings E_t in R^{N_tokens x d}.
- Spatial distillation loss: For each OCR token, identify which vision patches overlap with that token's bounding box. Compute a soft assignment matrix A where A[i,j] = IoU(bbox_token_i, bbox_patch_j) / sum_k IoU(bbox_token_i, bbox_patch_k). Then the distillation loss is: L_distill = sum_i ||E_t[i] - sum_j A[i,j] * E_v[j]||^2. This forces each text token embedding to approximate the weighted combination of vision patch embeddings that overlap its spatial position.
- Retrieval loss: Standard contrastive loss on the student's multi-vector representations, using late interaction (ColBERT-style MaxSim).
- Localization at inference: Given a query, compute MaxSim between query tokens and student's document token embeddings. The top-scoring document tokens already have bounding box coordinates (from OCR). Cluster the top-K scoring tokens spatially (using DBSCAN on their bounding box centers) to produce region-level bounding boxes.
- Training data: Use the same documents with both OCR+layout extraction and ColQwen2 patch embeddings. No additional annotation needed.

**Expected contribution:** Demonstrates that spatial understanding can transfer from vision to text+layout modality through distillation. Achieves near-VLM localization quality with text-encoder efficiency (10-50x faster, 10-100x smaller index). Novel spatial distillation loss that respects geometric correspondence between OCR tokens and vision patches.

**Related work differentiation:** Differs from ModernVBERT which trains a smaller VLM from scratch (still needs a vision encoder). Differs from Georgiou (2025) which maps patches to OCR at inference without learning. Differs from DocLLM which uses layout for understanding but not retrieval. This is the first to distill spatial retrieval knowledge from a VLM into a text-only model.

**Potential venues:** ICLR 2027, ACL 2026, EMNLP 2026, NeurIPS 2026.

**Feasibility:** Medium-High. Requires running ColQwen2 to extract patch embeddings on the training corpus (one-time cost, parallelizable). Requires an OCR pipeline (e.g., Tesseract, EasyOCR, or PaddleOCR -- all free). The student model can be small (250M-1B parameters) and trained on a single A100. The spatial assignment matrix A can be precomputed. Timeline: 3-4 months.

**Risk factors:** OCR errors may degrade the spatial correspondence. Documents with complex layouts where OCR ordering is wrong will confuse the student. The approach depends on having an OCR pipeline, partially negating the "OCR-free" advantage of ColPali. Mitigation: use a high-quality OCR engine and filter training samples where OCR confidence is low.

---

### Idea 4: Hierarchical Multi-Granularity Retrieval with Adaptive Region Zoom

**One-line summary:** Design a two-stage retrieval system that first retrieves at the page level using coarse representations, then performs adaptive region-level re-retrieval by "zooming in" to relevant sub-regions, using a single model with shared parameters across granularities.

**Motivation:** Current retrieval systems operate at a fixed granularity -- either page-level (ColPali, DSE) or, experimentally, region-level (RegionRAG). But the optimal granularity depends on the query: "What year was the company founded?" may require a single cell in a table, while "Summarize the methodology" spans an entire section. A hierarchical approach that dynamically selects the right zoom level would be more natural and efficient. This is also inspired by how Ferret-v2 uses any-resolution grounding and R-VLM uses a two-stage zoom-in approach for GUI grounding.

**Method:**
- Architecture: A single VLM encoder (Qwen2-VL or PaliGemma) with a shared vision-language backbone.
- Indexing: For each document page, create a multi-granularity index:
  - Level 0 (page): Average pool all patch embeddings into a single vector (like BiPali).
  - Level 1 (quadrant): Divide the page into a 2x2 grid. Average pool patches in each quadrant to get 4 vectors per page.
  - Level 2 (region): Divide into a 4x4 grid, yielding 16 vectors per page.
  - Level 3 (patch): Full ColPali-style 1024 vectors per page (stored lazily or for top candidates only).
- Retrieval pipeline:
  1. Stage 1 (Coarse retrieval): Use Level 0 vectors with a standard ANN index (HNSW) to retrieve top-100 pages. Fast because it is single-vector bi-encoder.
  2. Stage 2 (Region localization): For top-100 pages, load Level 1 or Level 2 vectors. Compute late-interaction scores at the quadrant/region level. Re-rank pages and identify the most relevant regions within each page.
  3. Stage 3 (Fine localization, optional): For top-10 pages, load Level 3 (full patch) vectors or re-encode at higher resolution. Run fine-grained MaxSim to produce precise bounding boxes by clustering the top-scoring patches.
- Training: Multi-granularity contrastive learning. At each level, the positive is the region containing the answer and negatives are other regions at the same level from the same page (hard in-page negatives) and from other pages. The shared backbone ensures representations are consistent across levels.
- Adaptive zoom policy: Train a lightweight classifier (small MLP on the query embedding) that predicts which zoom level is needed, so that easy queries stop at Level 1 while hard queries proceed to Level 3. This amortizes computational cost.

**Expected contribution:** First hierarchical retrieval system for visual documents that operates across multiple granularities with a single model. Achieves a better latency-accuracy tradeoff than flat ColPali (which always computes full patch interactions) or flat bi-encoder (which loses fine-grained information). The adaptive zoom policy is a novel contribution to retrieval efficiency.

**Related work differentiation:** RegionRAG uses a fixed region granularity and requires separate region proposals. ColPali is flat (page-level only). HPC-ColPali and DocPruner prune patches for efficiency but do not change the retrieval granularity. This work introduces a principled multi-level hierarchy. The zoom-in idea draws from Ferret-v2's any-resolution grounding but applies it to retrieval rather than VQA.

**Potential venues:** CVPR 2027, NeurIPS 2026, ICLR 2027, SIGIR 2026.

**Feasibility:** Medium. The multi-level index is straightforward to build (average pooling at different grid resolutions). The main research challenge is the multi-granularity training: region-level annotations are needed, which can be obtained from BBox-DocVQA, layout detection outputs, or synthetic annotations. The adaptive zoom classifier adds complexity. Estimated compute: 2-4 A100s for training. Timeline: 4-6 months.

**Risk factors:** The fixed grid (2x2, 4x4) may not align with natural document regions (e.g., a table spanning 3/4 of the page). Mitigation: use soft pooling regions informed by layout detection rather than hard grids. The multi-level index increases total storage (1 + 4 + 16 = 21 vectors per page for Levels 0-2), though this is still much less than ColPali's 1024 vectors. The adaptive policy may be hard to train with limited data showing diverse query complexities.

---

### Idea 5: Retrieval-Aware Grounding Pre-Training via Contrastive Patch-Region Alignment

**One-line summary:** Pre-train a vision-language model with a novel objective that jointly learns (a) contrastive retrieval from large document corpora and (b) patch-to-region alignment using layout annotations, producing a foundation model that can both retrieve and ground out of the box.

**Motivation:** Current VLMs are pre-trained either for general vision-language understanding (e.g., Qwen2-VL, PaliGemma) or fine-tuned separately for retrieval (ColPali) or grounding (Ferret, Florence-2). No model is pre-trained from the start with the joint objective of retrieval + spatial grounding. This means that combining retrieval and grounding always requires either a pipeline of separate models or a post-hoc fusion. A foundation model pre-trained with both objectives would produce representations that are inherently good for both tasks, avoiding the need for complex multi-stage pipelines.

**Method:**
- Pre-training data:
  - Retrieval pairs: 10M+ query-page pairs, sourced from ViDoRe training data, Wikipedia screenshot + NQ pairs (DSE-style), academic paper screenshots + generated queries.
  - Layout annotations: For all pages in the retrieval corpus, run a layout detector (Florence-2 or DocLayout-YOLO) to extract regions with bounding boxes and semantic labels (text, table, figure, header, etc.).
- Pre-training objectives (3 losses, jointly optimized):
  1. **Contrastive retrieval loss** (L_ret): Standard in-batch contrastive loss between query embeddings and page-level document embeddings (average-pooled patch embeddings). This teaches the model to discriminate relevant vs. irrelevant pages.
  2. **Patch-region alignment loss** (L_align): For each detected layout region R_k with bounding box B_k, compute the average patch embedding of patches inside B_k. Then maximize cosine similarity between this region embedding and the text content of that region (extracted via OCR or from the original text). This teaches individual patches to encode the content of the layout region they belong to.
  3. **Spatial grounding loss** (L_ground): Given a query that refers to a specific region (e.g., "the table in the bottom half"), the model must select the correct patches. This is implemented as a binary cross-entropy loss on patch selection, where ground-truth is derived from the overlap between query-relevant region bounding boxes and the patch grid.
- Architecture: Qwen2-VL-2B with a projection head for embedding and a lightweight grounding decoder (cross-attention between query tokens and patch embeddings, followed by MLP predicting bounding box coordinates).
- Total loss: L = L_ret + alpha * L_align + beta * L_ground.

**Expected contribution:** First pre-training recipe that produces "retrieval-grounding aware" patch embeddings. This creates a new type of foundation model for document intelligence that can serve as a drop-in replacement for ColPali while also providing localization. The three-objective pre-training is the core technical novelty.

**Related work differentiation:** ColPali trains only for retrieval. Florence-2 pre-trains for grounding but not retrieval. VDocRAG proposes pre-training for retrieval but without spatial grounding. Marten adds mask generation as auxiliary but for VQA, not retrieval. This is the first to combine contrastive retrieval, layout-aware patch alignment, and spatial grounding in a single pre-training recipe.

**Potential venues:** ICLR 2027, NeurIPS 2026, CVPR 2027.

**Feasibility:** Medium-Low. This is the most resource-intensive idea. Pre-training on 10M+ pages requires significant compute (estimate: 32-64 A100-hours per epoch, with 3-5 epochs needed). Layout detection on the full corpus is a one-time cost but substantial. However, all components are individually well-understood. A smaller-scale proof of concept on 1M pages with a 2B model is feasible on an academic budget. Timeline: 5-7 months.

**Risk factors:** Three losses may compete: retrieval loss pushes for discriminative page-level representations, while grounding loss pushes for fine-grained spatial specificity. These may not naturally align. Careful loss weighting and curriculum (e.g., start with retrieval, then add grounding) will be critical. The quality of automatically generated layout annotations will affect L_align. The scale of pre-training data may be insufficient to fully realize the benefits.

---

### Idea 6: Sparse Grounding Tokens -- Efficient Localization via Learnable Spatial Anchors in Document Retrieval

**One-line summary:** Replace ColPali's dense 1024-patch representation with a small set (e.g., 16-32) of learnable "grounding tokens" that are spatially aware, achieving competitive retrieval with dramatically reduced index size while enabling localization through each token's learned spatial receptive field.

**Motivation:** The central tension in multi-vector document retrieval is between spatial richness (more vectors = better localization) and efficiency (fewer vectors = faster retrieval, smaller index). DocPruner and HPC-ColPali try to prune redundant patches post-hoc, but this is suboptimal because the model was never trained to concentrate information into fewer tokens. Meanwhile, perceiver/resampler architectures (used in Flamingo, BLIP-2) compress visual tokens through learnable queries, but these lose spatial correspondence. The key idea is: what if we train learnable tokens that (a) each specialize in a spatial region of the page and (b) together provide enough information for retrieval AND localization?

**Method:**
- Architecture: Start with a VLM encoder (Qwen2-VL-2B). After the vision encoder produces N_patch patch embeddings with positions, introduce K learnable spatial anchor tokens (K=16 or 32). Each anchor token q_k has a trainable 2D position (mu_k, sigma_k) initialized on a regular grid over the image.
- Spatial attention: Each anchor attends to all patches, but with a position-dependent Gaussian attention bias: attn_weight(q_k, patch_j) = softmax(q_k^T patch_j / sqrt(d) + gaussian(pos_j; mu_k, sigma_k)). This encourages each anchor to attend primarily to patches near its spatial center while still allowing global information flow.
- After one or two cross-attention layers, the K anchor embeddings become the multi-vector representation of the page (replacing the 1024 patch vectors of ColPali).
- Training: (1) Contrastive late-interaction loss between query token embeddings and the K anchor embeddings. (2) Spatial regularization loss: the learned positions (mu_k, sigma_k) should cover the page uniformly (minimize KL divergence with a uniform 2D distribution) to prevent anchor collapse. (3) Localization loss: given a query, the anchor whose learned position (mu_k) is closest to the ground-truth region should have the highest similarity score. This teaches the model to use anchor positions for localization.
- Localization at inference: After computing MaxSim between query tokens and anchor embeddings, the anchor with the highest similarity score reveals the approximate location (its mu_k). The sigma_k gives the spatial extent. This directly produces a localization heatmap without additional computation.

**Expected contribution:** Novel architecture that reconciles the efficiency-localization tradeoff. Achieves 32-64x compression over ColPali (16-32 vectors instead of 1024) while retaining spatial localization capability. The learnable spatial anchors with Gaussian attention are a new architectural primitive for spatially-aware representation learning.

**Related work differentiation:** DocPruner and HPC-ColPali prune post-hoc without spatial awareness. Perceiver/resampler architectures compress but lose spatial structure. ModernVBERT uses a smaller model but still produces many tokens. This work designs an architecture specifically to compress into spatially-meaningful tokens. Closest in spirit to DETR's object queries, but applied to document retrieval rather than detection.

**Potential venues:** NeurIPS 2026, ICLR 2027, CVPR 2027, ECCV 2026.

**Feasibility:** Medium. The architecture is relatively simple (cross-attention with position-dependent bias). Training requires the same data as ColPali plus region annotations for the localization loss. The main challenge is ensuring that K=16-32 tokens are sufficient for retrieval quality (ColPali uses 1024). Ablation studies on K will be critical. A single A100 suffices for training the 2B model. Timeline: 3-5 months.

**Risk factors:** With only 16-32 tokens, the model may lose fine-grained information needed for queries about small details (e.g., a specific number in a table cell). The Gaussian attention bias may over-constrain the model's ability to attend to distant but semantically relevant patches. The spatial regularization loss may conflict with the retrieval loss (e.g., if most queries are about a specific page region, the model wants to concentrate anchors there, but the regularization pushes for uniform coverage). The approach assumes that localization can be approximated by a single anchor position, which may not hold for queries that require information from multiple disjoint regions.

---

### Idea 7: UnifiedRetGround -- A Single Autoregressive VLM that Retrieves, Grounds, and Answers in One Forward Pass

**One-line summary:** Train a single autoregressive vision-language model that, given a query and a set of candidate document images, outputs in a single generation: (1) the index of the most relevant page, (2) bounding box coordinates of the evidence region, and (3) the answer -- unifying retrieval, grounding, and generation into one model.

**Motivation:** The current approach to grounded document QA is a 3-stage pipeline: retrieve (ColPali) then ground (Ferret/Florence-2) then generate (GPT-4o/Qwen2.5-VL). Each stage has its own model, its own errors, and its own latency. Errors compound: if retrieval fails, grounding and generation fail too. Recent models like Kosmos-2 and Qwen3-VL have shown that a single autoregressive model can generate both text and bounding box coordinates. And models like GENIUS show that generative approaches can work for retrieval (generating document identifiers autoregressively). The ambitious insight is: these capabilities can be unified.

**Method:**
- Architecture: A strong VLM (Qwen2.5-VL-7B or InternVL-7B) as the base model.
- Input format: The model receives a text query and N candidate page images (N=5-20, pre-filtered by a lightweight first-stage retriever like BiPali). All page images are concatenated as a multi-image input with special <PAGE_1>, <PAGE_2>, ... separator tokens.
- Output format (structured generation): The model generates a JSON-like output:
  ```
  {"page": 3, "bbox": [0.12, 0.45, 0.67, 0.82], "answer": "The revenue increased by 15%"}
  ```
  The page index selects which input page contains the answer. The bbox coordinates are normalized [0,1] values. The answer is a text response.
- Training:
  - Phase 1: Multi-page QA fine-tuning on existing datasets (M3DocVQA, MP-DocVQA) where ground-truth includes the correct page index.
  - Phase 2: Grounded QA fine-tuning on BBox-DocVQA, VISA datasets where ground-truth includes both page and bounding box.
  - Phase 3: Joint fine-tuning with a combined loss: cross-entropy on page selection token + smooth-L1 on bounding box tokens + cross-entropy on answer tokens. All in the standard autoregressive next-token-prediction framework with special token types for coordinates (following Kosmos-2 / Qwen-VL's approach of discretizing coordinates into location tokens).
- Inference pipeline: BiPali retrieves top-20 pages (fast, single-vector). UnifiedRetGround processes these 20 pages and the query in one forward pass, outputting the page, bbox, and answer.

**Expected contribution:** First model to perform page selection, spatial grounding, and answer generation in a single autoregressive forward pass. Eliminates the need for separate retrieval and grounding models after the initial filtering stage. Shows that the "retrieve-ground-generate" pipeline can be collapsed into a single model, simplifying deployment and reducing error compounding. The structured generation format is a practical contribution for building grounded document QA systems.

**Related work differentiation:** GENIUS does generative retrieval but without grounding. Kosmos-2 does grounding but not retrieval. M3DocRAG uses ColPali + Qwen2-VL as separate models. VISA does attribution but not page selection. ViDoRAG uses multi-agent reasoning but not a single unified model. This is the first end-to-end model for the full pipeline.

**Potential venues:** ICLR 2027, NeurIPS 2026, ACL 2026, CVPR 2027.

**Feasibility:** Medium. The key question is whether a 7B VLM can reliably process 20 page images (each ~1024 tokens = 20K visual tokens total) plus a query in one forward pass. This may require: (a) aggressive image compression (lower resolution, 512px), (b) a strong long-context model (Qwen2.5-VL supports 32K tokens), or (c) reducing N to 5-10 pages. Training requires multi-page datasets with bounding box annotations, which exist but may need augmentation. Compute: ~4 A100s for 7B model fine-tuning. Timeline: 4-6 months.

**Risk factors:** The biggest risk is context length: 20 pages at high resolution may exceed the model's effective context window, degrading quality. Reducing to 5 pages limits the re-ranking capability. The model may learn shortcuts (always selecting page 1, or generating generic bounding boxes). The structured output format requires careful training to avoid degenerate outputs. Also, for truly large-scale retrieval (millions of pages), the first-stage filter (BiPali) becomes the bottleneck, and this model only helps with the re-ranking and grounding stages.

---

## Summary Ranking

| Idea | Novelty | Feasibility | Impact | Risk | Recommended Priority |
|------|---------|-------------|--------|------|---------------------|
| 1. ColPali-Ground | Medium-High | High | High | Medium | 1st -- solid incremental contribution with clear path |
| 2. GroundedViDoRe Benchmark | Medium | Very High | Very High | Low | 2nd -- high community impact, enables all other work |
| 3. Spatial Token Distillation | High | Medium-High | High | Medium | 3rd -- novel and practical |
| 4. Hierarchical Multi-Granularity | Medium-High | Medium | High | Medium | 4th -- good engineering contribution |
| 5. Retrieval-Aware Pre-Training | Very High | Low-Medium | Very High | High | 5th -- ambitious flagship, high reward |
| 6. Sparse Grounding Tokens | Very High | Medium | High | High | 6th -- creative architecture, risky |
| 7. UnifiedRetGround | High | Medium | Very High | High | 7th -- ambitious end-to-end, deployment story |

**Recommended strategy:** Start with Ideas 1 and 2 in parallel (both achievable in 3-4 months, complement each other perfectly -- the benchmark enables evaluation of the model). Then pursue Idea 3 or 6 as the next paper, and Ideas 5 or 7 as longer-term ambitious projects.
