# Expert Reviewer Evaluation: Retrieval with Spatial Localization in Visual Documents

**Reviewer Profile:** Area Chair level, CVPR/NeurIPS/ICLR. 100+ papers reviewed, 50+ published.
**Date of Review:** 2026-03-28

---

## Idea 1: ColPali-Ground -- End-to-End Document Retrieval with Patch-Level Spatial Grounding

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 7/10 | The core technical idea -- backpropagating a grounding loss through shared patch embeddings used for retrieval -- is sound. The assumption that MaxSim similarity maps contain localizable spatial signals is empirically supported by Georgiou (2025). However, the smooth-L1 regression on patch indices conflates discrete grid positions with continuous bounding box coordinates, and the authors need to rigorously justify that a lightweight MLP over top-K similarity vectors is sufficient for precise regression. |
| Novelty | 6/10 | This is a natural and somewhat predictable extension of ColPali. The idea of adding a grounding head on top of a retrieval model has been explored in text retrieval (e.g., passage highlighting). The multi-task training of retrieval + localization is not conceptually novel; similar joint objectives exist in object detection + retrieval settings. What is new is applying it specifically to the ColPali embedding space. A reviewer familiar with this area would characterize this as solid but incremental. |
| Significance | 7/10 | Addresses a real and well-articulated gap. If successful, this would be the first model to demonstrate that late-interaction multi-vector embeddings can serve dual purposes. The proposed joint evaluation metric (Retrieval@K + IoU@0.5) is itself a useful contribution. Practitioners in document AI would find this directly useful. |
| Clarity | 8/10 | The idea is extremely well-articulated. The two-stage training pipeline is clear, the architecture is concrete, and the inference procedure is easy to follow. A paper could be written directly from this description. The differentiation from related work is precise and honest. |
| Experimental Design | 6/10 | The described experiments are necessary but not sufficient. The baselines (Georgiou post-hoc, pipeline approaches) are appropriate. However, the proposal lacks: (a) ablation on the localization head architecture, (b) analysis of how grounding loss weight lambda affects retrieval performance (the key tension), (c) evaluation across different document types (text-heavy vs. figure-heavy), (d) a failure mode analysis. The BBox-DocVQA dataset alone (32K pairs) may be insufficient for robust training. |
| Reproducibility | 8/10 | All components are available: ColQwen2 is open source, BBox-DocVQA is public, VISA datasets are accessible. The localization head is a simple MLP. The training recipe is standard. A competent lab could reproduce this in 2-3 months. |
| Weakness Severity | 6/10 | The patch grid resolution (32x32) is a genuine limitation for small text regions -- this is acknowledged but not adequately addressed. The reliance on BBox-DocVQA as the primary grounding data source is a bottleneck: 32K pairs across limited domains may not generalize. The lambda-tuning sensitivity is a real concern but not a fatal flaw. |

**Overall Score: 6.5/10**
**Predicted Verdict: Borderline Accept**

### Strengths
- **S1:** Clean and well-motivated idea that addresses a clear gap between retrieval and grounding literature. The observation that MaxSim already computes a soft spatial attention map is insightful.
- **S2:** High feasibility -- all data, models, and training infrastructure exist. Single-GPU fine-tuning on ColQwen2-2B is realistic.
- **S3:** Practical inference design: the localization head adds negligible overhead since the similarity matrix is already computed during retrieval. This is deployment-friendly.

### Weaknesses
- **W1:** Limited novelty. Adding an MLP grounding head to an existing retrieval model and training with a joint loss is a standard recipe (multi-task learning). Reviewers at top venues would likely flag this as "straightforward extension" rather than a genuine conceptual advance.
- **W2:** Patch grid resolution bottleneck. At 32x32 patches for a 1024px image, each patch covers ~32x32 pixels. For dense text at 8pt font, a single line may span 2-3 patches vertically, making precise bounding box regression inherently noisy. The paper would need to demonstrate that this resolution is sufficient or propose a multi-resolution solution.
- **W3:** Training data scarcity for grounding. BBox-DocVQA (32K) + VISA are small-scale. The idea mentions "a newly curated subset of DocVQA/InfographicVQA with region annotations" but this data curation effort is hand-waved. Without sufficient grounding data, the model may learn a degenerate localization (e.g., always predicting the center of the page).

### Questions to Authors
- **Q1:** What is the retrieval performance degradation (nDCG@5 on ViDoRe) when the grounding loss is added? If there is a Pareto tradeoff between retrieval accuracy and grounding precision, how do you propose to navigate it?
- **Q2:** How does the localization head handle queries whose answer spans multiple disjoint regions on a page (e.g., a comparison query referencing two different tables)? The current design appears to predict a single bounding box.

### Missing References
- DETR (Carion et al., 2020) -- the original work on using learnable queries for spatial prediction, conceptually related to regressing bounding boxes from attention patterns.
- GroundingGPT (Li et al., 2024) -- a model that unifies grounding with language generation; relevant architecture comparison.
- DePlot / ChartQA -- document grounding benchmarks for chart understanding that could provide additional training data.

### Suggested Experiments
1. **Retrieval-Grounding Pareto Curve:** Sweep lambda from 0.0 to 1.0 and plot retrieval nDCG@5 vs. grounding IoU@0.5. This is the single most important experiment to establish that joint training does not severely degrade either task.
2. **Resolution Ablation:** Compare localization accuracy at different input resolutions (512px, 1024px, 2048px) and patch grid sizes. This directly addresses W2 and would reveal the practical limits of patch-level grounding.

---

## Idea 2: GroundedViDoRe -- A Benchmark for Joint Document Retrieval and Region Localization

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 7/10 | The benchmark design follows established best practices (Segment-Judge-Generate pipeline from BBox-DocVQA, difficulty levels from REAL-MM-RAG, standard IoU metrics). The proposed joint metric Grounded-Recall@K is well-defined and meaningful. The 20% human verification rate is reasonable but may be insufficient for ensuring quality at scale. |
| Novelty | 5/10 | Benchmark papers inherently have lower novelty scores. The idea of combining retrieval and localization metrics is natural, not surprising. The data construction pipeline reuses existing techniques (layout detection + VLM query generation + human verification). The individual components are all borrowed. What is new is the specific combination and the explicit joint evaluation framing. |
| Significance | 8/10 | This fills a genuine and well-documented gap. Every other idea in this document implicitly needs this benchmark for evaluation. The community would benefit substantially. Benchmark papers at NeurIPS D&B track have high citation potential. The inclusion of 5 diverse domains and 3 difficulty levels adds lasting value. |
| Clarity | 8/10 | The description is thorough and specific. Corpus sources, pipeline stages, metric definitions, and baselines are all concretely specified. The risk factors (inter-annotator agreement, VLM bias toward easily-described regions) are honestly acknowledged. |
| Experimental Design | 7/10 | The four proposed baselines cover the relevant design space (post-hoc, pipeline, generative). The metrics are comprehensive (page-level, box-level, joint). Missing: (a) human upper bound estimation, (b) cross-domain analysis (performance on papers vs. financial docs vs. medical), (c) error decomposition (how often does the system find the right page but wrong region, vs. wrong page entirely?). |
| Reproducibility | 9/10 | All components are publicly available. The construction pipeline is well-defined and reproducible. The estimated cost ($5K for annotation) is realistic for academic budgets. The benchmark itself, once released, is trivially reproducible by other labs. |
| Weakness Severity | 5/10 | The core weakness of benchmark papers is always: does anyone actually use it? Beyond that, the VLM-generated queries may have systematic biases that reduce the benchmark's validity as a test of real-world grounded retrieval. The 50K pages / 5 domains is large but may still underrepresent edge cases (handwritten documents, low-resolution scans, non-English text). |

**Overall Score: 6.0/10**
**Predicted Verdict: Borderline Accept (Accept at NeurIPS D&B track)**

### Strengths
- **S1:** Fills a clear, well-motivated evaluation gap. The paper writes itself: "no existing benchmark jointly evaluates retrieval + localization for document images."
- **S2:** High practical value. Every team working on grounded document retrieval needs a standard benchmark. First-mover advantage is real for benchmarks.
- **S3:** Feasible and cost-effective. Unlike model papers that need expensive GPU training, this primarily requires annotation effort and pipeline engineering -- achievable on a modest academic budget.

### Weaknesses
- **W1:** Limited novelty for a top-tier venue. Benchmark papers at CVPR/NeurIPS main tracks need either a surprising finding from baseline analysis or a genuinely novel evaluation paradigm. "We combined existing retrieval metrics with existing localization metrics" may not clear the novelty bar at the main conference. The D&B track is a better fit.
- **W2:** VLM-generated queries may not represent real user information needs. Studies have shown that synthetic queries from large LMs tend toward specific patterns (overly precise, unnaturally detailed, lacking the ambiguity of real queries). Without a substantial portion of organically collected queries, the benchmark may measure performance on artificial tasks.
- **W3:** Scale limitations. 50K pages is large for this specific task but small compared to web-scale retrieval benchmarks (MS MARCO has 8.8M passages). Models that work well at 50K may fail at 1M+ pages. The benchmark may not reveal scaling behavior.

### Questions to Authors
- **Q1:** What percentage of the generated queries have unambiguous ground-truth regions? For a query like "What is the main finding?", the relevant region could be the abstract, the conclusion, or a figure -- how do you handle multi-region ground truth?
- **Q2:** How do you ensure the benchmark remains challenging as models improve? ViDoRe V1 was quickly saturated, leading to V2 and V3. What is the plan for benchmark longevity?

### Missing References
- MS MARCO (Bajaj et al., 2016) -- the gold standard for retrieval benchmarks; this paper should discuss how GroundedViDoRe's construction differs.
- Natural Questions (Kwiatkowski et al., 2019) -- an example of a large-scale QA benchmark with well-designed annotation guidelines; relevant for methodology.
- LVIS (Gupta et al., 2019) -- a detection benchmark that addressed long-tail distribution issues; relevant for ensuring domain diversity.
- DocBench (Zou et al., 2024) -- recent document understanding benchmark that should be differentiated.

### Suggested Experiments
1. **Human Performance Upper Bound:** Have 3-5 human annotators perform the full grounded retrieval task on a 500-query subset. Report human Grounded-Recall@K. This establishes the ceiling and reveals inherent task ambiguity.
2. **Error Decomposition Analysis:** For each baseline, decompose failures into (a) retrieval failures (wrong page), (b) localization failures (right page, wrong region), (c) joint failures. This is the most valuable analysis a benchmark paper can provide -- it tells the community exactly where to focus research effort.

---

## Idea 3: Spatial Token Distillation -- Teaching Text Retrievers to Inherit Patch-Level Spatial Knowledge

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 7/10 | The spatial distillation loss using IoU-based soft assignment between OCR tokens and vision patches is mathematically sound and geometrically motivated. The DBSCAN clustering for inference-time localization is a reasonable heuristic. The core assumption -- that VLM patch embeddings contain distillable spatial knowledge -- is plausible. However, the assumption that OCR token embeddings can faithfully represent the same information as vision patch embeddings (which include visual features like fonts, colors, layout cues) is a strong one that needs empirical validation. |
| Novelty | 7/10 | The spatial distillation loss (IoU-weighted alignment between text token and vision patch embeddings) is genuinely novel. While knowledge distillation from VLMs to text models exists (e.g., LLM2Vec, TinyBERT), spatially-structured distillation that preserves geometric correspondence is new. The idea of creating a text-only model that can do spatial localization is counterintuitive and surprising, which is a hallmark of novelty. |
| Significance | 7/10 | Addresses a real practical problem: enterprise deployment at scale. If this works, it could make grounded document retrieval 10-100x more efficient. The efficiency claims (10-50x faster, 10-100x smaller index) would be impactful for industry adoption. However, the significance is somewhat diminished by the dependence on OCR, which ColPali was explicitly designed to eliminate. |
| Clarity | 7/10 | The method is well-described, especially the spatial assignment matrix and distillation loss. The training pipeline is clear. Some ambiguity around: (a) how the soft assignment matrix handles overlapping OCR tokens, (b) how DBSCAN parameters are chosen at inference, (c) what happens when OCR ordering differs from reading order. |
| Experimental Design | 6/10 | The implicit experimental plan is underdeveloped. Key missing experiments: (a) comparison against simply running ColQwen2 at lower resolution (a naive baseline), (b) analysis of distillation quality vs. OCR error rate, (c) ablation on the student model size, (d) comparison on domains where OCR is known to struggle (diagrams, handwritten text, non-Latin scripts). The efficiency claims need rigorous latency/memory profiling, not just asymptotic estimates. |
| Reproducibility | 7/10 | Components are available (ColQwen2, OCR pipelines, training data). The soft assignment matrix computation is well-defined. However, the specific OCR engine choice, preprocessing steps, and DBSCAN parameters could significantly affect results, introducing hidden variability. |
| Weakness Severity | 5/10 | The OCR dependency is a significant weakness. ColPali's primary selling point was being "OCR-free." This idea reintroduces OCR as a core requirement, which may be perceived as a step backward. Additionally, for visually rich documents (infographics, charts, diagrams), OCR produces little or no text, making this approach fundamentally limited to text-heavy documents. This is not a fatal flaw but substantially narrows the applicability. |

**Overall Score: 6.0/10**
**Predicted Verdict: Borderline Accept**

### Strengths
- **S1:** Novel and technically interesting spatial distillation loss. The IoU-weighted soft assignment between modalities is elegant and geometrically principled. This loss function alone could find applications beyond this specific paper.
- **S2:** Addresses a genuine deployment bottleneck. The 10-100x efficiency improvement would make grounded retrieval practical for enterprise-scale document collections (millions of pages).
- **S3:** No additional annotation needed -- the distillation uses only the teacher's outputs and OCR coordinates. This makes the approach scalable and easy to apply to new domains.

### Weaknesses
- **W1:** Reintroduces OCR dependency. The entire motivation of ColPali was to bypass OCR pipelines. This work effectively says "OCR is needed after all, but only during training/indexing." While pragmatic, reviewers may view this as philosophically regressive. For visually rich documents where OCR fails, the approach breaks down.
- **W2:** Information loss during distillation may be severe. Vision patches encode visual features (font style, background color, spatial layout, graphical elements) that have no counterpart in OCR text tokens. The student may learn good spatial correspondence but lose the visual discriminative power that makes VLM-based retrieval strong in the first place.
- **W3:** The DBSCAN clustering for localization is ad-hoc. The number of clusters, the distance threshold, and the minimum points parameter all need tuning, and the optimal values likely vary across document types. This weakens the "end-to-end" narrative.

### Questions to Authors
- **Q1:** What is the retrieval performance gap between the student and teacher across different document types? Specifically, on visually rich documents (infographics, slides with diagrams), where OCR captures only a fraction of the visual content, how much performance is lost?
- **Q2:** How does this approach compare to simply running ColQwen2 at a lower resolution (e.g., 256px instead of 1024px) for efficiency? This naive baseline achieves similar efficiency gains -- can spatial distillation meaningfully outperform it?

### Missing References
- TinyBERT (Jiao et al., 2020) -- knowledge distillation for text encoders; the standard reference for distillation in NLP.
- LLM2Vec (BehnamGhader et al., 2024) -- converting LLMs to text encoders; related approach of transferring knowledge across model types.
- LayoutLMv3 (Huang et al., 2022) -- a layout-aware pre-training approach that uses both text and layout features; a relevant baseline for the student architecture.
- ERNIE-Layout (Peng et al., 2022) -- another layout-aware model that should be considered as a student architecture alternative.

### Suggested Experiments
1. **Document Type Stratification:** Evaluate the student-teacher gap separately on (a) text-heavy documents, (b) table-heavy documents, (c) figure-heavy documents, (d) mixed documents. This reveals exactly where the distillation succeeds and fails, and whether the approach should be recommended only for specific document types.
2. **OCR Quality Sensitivity:** Intentionally degrade OCR quality (add character-level noise, drop random words, shuffle reading order) and measure the impact on both retrieval and localization. This establishes robustness bounds and informs deployment decisions.

---

## Idea 4: Hierarchical Multi-Granularity Retrieval with Adaptive Region Zoom

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 6/10 | The hierarchical pooling strategy is technically sound for the retrieval component. Average pooling at different grid resolutions is a valid way to create multi-granularity representations. However, the adaptive zoom policy (MLP classifier on query embedding) is underspecified and potentially unsound: it assumes the optimal zoom level can be predicted from the query alone, without seeing any documents. In practice, the optimal granularity depends on both the query AND the document layout. The fixed grid assumption (2x2, 4x4) is a known weakness acknowledged by the authors. |
| Novelty | 5/10 | Hierarchical retrieval is a well-studied concept in information retrieval (passage retrieval with document-level pre-filtering, multi-resolution image search). The specific application to visual document retrieval with patch pooling at different granularities is new in this domain but borrows heavily from established ideas. The adaptive zoom policy is reminiscent of early-exit mechanisms in neural networks and cascaded classifiers, which are decades old. Reviewers would likely say: "The individual components are known; the novelty is in the combination." |
| Significance | 6/10 | The latency-accuracy tradeoff is practically important. Being able to stop at a coarse level for easy queries is valuable. However, the improvement over simply running ColPali (which is already efficient with MaxSim) may be marginal unless the corpus is very large. The practical significance depends entirely on the magnitude of speedup vs. accuracy loss, which is an empirical question. |
| Clarity | 6/10 | The hierarchical structure is clear, but the training procedure is underdeveloped. How exactly is the multi-granularity contrastive learning implemented? How are region-level positives and negatives defined for arbitrary grid cells? What is the training data for the adaptive zoom classifier? These details are left vague. |
| Experimental Design | 5/10 | The description mentions no specific experiments, baselines, or datasets. For a systems-oriented paper, the experimental design must include: (a) latency profiling at each level, (b) accuracy at each level independently and combined, (c) comparison against ColPali with different numbers of pruned patches (the obvious baseline), (d) comparison against ANN-based approaches like HNSW over single-vector representations. Without these, the contribution is unclear. |
| Reproducibility | 6/10 | The architecture is conceptually reproducible (average pooling at different grid levels). However, the training recipe for multi-granularity contrastive learning and the adaptive zoom policy involve many design choices (loss weights, number of levels, grid sizes, training data for the policy) that are not specified. |
| Weakness Severity | 4/10 | The fixed grid is a serious limitation. Real documents have complex layouts where semantic regions do not align with regular grids. A 2x2 grid that splits a table across two quadrants will produce nonsensical quadrant-level representations. The authors acknowledge this but the proposed mitigation (soft pooling informed by layout detection) essentially transforms this into a different, more complex system. The adaptive zoom policy training is poorly defined and may not work. |

**Overall Score: 5.0/10**
**Predicted Verdict: Borderline Reject**

### Strengths
- **S1:** Intuitive hierarchical design that mirrors how humans search documents (skim page, then zoom into relevant region). The coarse-to-fine retrieval cascade is a well-motivated systems design.
- **S2:** Efficiency story is compelling: Level 0 single-vector search over millions of pages is fast, and finer levels are only applied to top candidates. Storage for Levels 0-2 (21 vectors/page) is dramatically smaller than ColPali (1024 vectors/page).
- **S3:** The framework is modular -- each level can be independently improved or replaced, and the number of levels can be tuned for different latency budgets.

### Weaknesses
- **W1:** Fixed grid pooling is fundamentally misaligned with document structure. Documents are organized by semantic regions (paragraphs, tables, figures), not by spatial quadrants. A table spanning the bottom 60% of a page would be split across quadrants in a 2x2 grid, producing meaningless region embeddings. This is not just a minor issue -- it undermines the core architecture.
- **W2:** Unclear advantage over simpler baselines. ColPali with HNSW pre-filtering (encode a single vector per page for first-stage, then full MaxSim for re-ranking) achieves a similar coarse-to-fine effect without the complexity of multi-granularity training. The paper would need to convincingly demonstrate that the intermediate levels (quadrant, 4x4 grid) provide meaningful signal beyond what coarse retrieval + fine re-ranking already offers.
- **W3:** The adaptive zoom policy is the weakest component. Predicting the needed granularity from the query alone is ill-posed: the same query ("What is the revenue?") might need full-page context in one document and a single table cell in another. Query-only zoom prediction is fundamentally limited.

### Questions to Authors
- **Q1:** How does the hierarchical approach compare against the simple baseline of ColPali with HNSW pre-filtering (single-vector first stage) followed by full MaxSim re-ranking on top-K? This baseline achieves coarse-to-fine retrieval without any new architecture.
- **Q2:** For the 4x4 grid (Level 2), what percentage of ground-truth answer regions in BBox-DocVQA are cleanly contained within a single grid cell? If most answers span multiple cells, the intermediate levels provide little localization value.

### Missing References
- Matryoshka Representation Learning (Kusupati et al., 2022) -- multi-granularity embeddings where different dimensions capture different levels of detail; a more principled approach to hierarchical representations.
- COIL (Gao et al., 2021) -- contextualized exact match in retrieval, which explores the tradeoff between dense and sparse retrieval; relevant for the efficiency discussion.
- Cascaded retrieval approaches (Nogueira & Cho, 2019) -- classic two-stage retrieve-then-rerank; the paper should position against this established paradigm.

### Suggested Experiments
1. **Grid Alignment Analysis:** Measure what percentage of ground-truth bounding boxes in BBox-DocVQA/VISA are fully contained within a single cell at each grid level (2x2, 4x4, 8x8). This directly quantifies the information loss from grid-based pooling and reveals whether layout-aligned pooling is essential.
2. **Ablation Without Adaptive Policy:** Compare the hierarchical retrieval with a fixed policy (always use all 3 levels) against the learned adaptive policy. If the fixed policy performs comparably, the adaptive component adds complexity without benefit, and the paper should focus on the hierarchical architecture alone.

---

## Idea 5: Retrieval-Aware Grounding Pre-Training via Contrastive Patch-Region Alignment

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 6/10 | Each of the three losses is individually well-motivated and technically sound. The concern is their interaction. L_ret pushes for page-level discriminative representations (patches should collectively encode page identity). L_align pushes for region-level content encoding (each patch encodes its local content). L_ground pushes for query-conditioned spatial selection. These three objectives pull patch embeddings in different directions. The assumption that they can be harmonized through loss weighting is optimistic. The approach also relies on automatically generated layout annotations (from Florence-2 / DocLayout-YOLO), whose quality directly affects L_align -- introducing a hard dependency on an imperfect upstream model. |
| Novelty | 8/10 | This is the most conceptually ambitious idea. A pre-training recipe that jointly produces retrieval-capable and grounding-capable patch embeddings is genuinely new. No existing work combines contrastive retrieval, layout-aware alignment, and spatial grounding in a single pre-training framework. The three-objective formulation represents a distinct contribution to the pre-training methodology literature. |
| Significance | 8/10 | If successful, this would establish a new category of foundation model for document intelligence. The idea that pre-training should be "retrieval-grounding aware from day one" is a meaningful paradigm shift. The resulting model could serve as a backbone for multiple downstream tasks. This has the potential for high citation impact. |
| Clarity | 6/10 | The three losses are clearly defined individually, but the training recipe (how to schedule losses, curriculum strategy, warmup procedures) is vague. The description mentions "careful loss weighting and curriculum" but does not specify what this entails. The scale of pre-training (10M+ pages) is stated but the actual feasibility on academic compute is unclear. |
| Experimental Design | 5/10 | The proposal lacks specificity on experiments. For a pre-training paper, the experimental bar is extremely high: (a) pre-training then evaluation on multiple downstream tasks (retrieval, grounding, VQA), (b) comparison against separately pre-trained models (ColPali for retrieval, Florence-2 for grounding), (c) scaling analysis (performance vs. pre-training data size, model size), (d) representation analysis (what do the patch embeddings learn?). None of these are specified. |
| Reproducibility | 4/10 | 10M+ pages of pre-training data, 32-64 A100-hours per epoch, 3-5 epochs -- this is 100-320 A100-hours minimum, which is borderline for academic reproducibility. The layout annotation on the full corpus adds substantial engineering overhead. The three-loss interaction makes hyperparameter tuning expensive. An industry lab could reproduce this; most academic labs cannot without significant resources. |
| Weakness Severity | 4/10 | The multi-loss optimization challenge is serious. Pre-training papers that try to combine too many objectives often find that the losses interfere, and the final model is worse than models trained with individual losses. This is especially likely when the losses operate at different granularities (page vs. region vs. patch). The authors acknowledge this risk but offer only vague mitigation. Additionally, the reliance on automatic layout annotations introduces noise into the core training signal, and the scale requirements may prevent thorough ablation studies. |

**Overall Score: 5.5/10**
**Predicted Verdict: Borderline (likely Reject at top venues without strong empirical results)**

### Strengths
- **S1:** Highest conceptual novelty among all 7 ideas. The vision of a foundation model pre-trained for joint retrieval and grounding is compelling and forward-looking. If this were demonstrated to work, it would be a landmark paper.
- **S2:** The three-loss formulation is well-motivated: each loss targets a specific capability (page discrimination, region content encoding, spatial selection). The framework is intellectually coherent.
- **S3:** The potential impact is very high. A successful model would become the standard backbone for document retrieval systems, replacing the current ColPali family.

### Weaknesses
- **W1:** Multi-objective optimization risk. Pre-training with 3 losses at different granularities (page, region, patch) is extremely challenging. Without extensive ablation showing that all three losses contribute positively and do not interfere, reviewers will be skeptical. This is the kind of idea that sounds great on paper but often fails in practice due to optimization difficulties.
- **W2:** Compute requirements limit thoroughness. At 100+ A100-hours for a single run, the authors can afford only a handful of experiments. Pre-training papers at top venues (e.g., Florence-2, BLIP-2) typically include extensive scaling curves, ablations on each component, and evaluation on 10+ downstream tasks. Meeting this bar on an academic budget will be extremely difficult.
- **W3:** Layout annotation quality bottleneck. L_align requires accurate region-level annotations for the entire 10M+ page corpus. Automatic layout detectors have non-trivial error rates (10-20% for complex documents). Training on noisy region labels may produce a model that learns to ignore the alignment signal, effectively reducing the approach to standard ColPali pre-training.

### Questions to Authors
- **Q1:** Can you provide results from a small-scale pilot (e.g., 100K pages, 2B model, single A100)? Pre-training papers need to show that the approach works at small scale before arguing it will work at large scale. Without pilot results, this is a speculative proposal.
- **Q2:** What happens when L_align and L_ret conflict? Specifically, if two pages have visually similar regions but different overall content, L_ret wants their page embeddings to be different while L_align wants their region embeddings to be similar. How is this tension resolved?

### Missing References
- SigLIP (Zhai et al., 2023) -- a pre-training approach that modifies contrastive loss; relevant for understanding loss design choices.
- UniCL (Yang et al., 2022) -- unified contrastive learning combining multiple objectives; directly relevant methodology.
- METER (Dou et al., 2022) -- multi-objective VLP that studies loss interactions empirically; important reference for the optimization challenge.
- DINOv2 (Oquab et al., 2023) -- self-supervised visual pre-training with spatial awareness; relevant for the patch-level representation learning aspect.

### Suggested Experiments
1. **Loss Ablation Matrix (3x3):** Train 8 model variants: each individual loss alone (3 runs), each pair (3 runs), all three together (1 run), and no pre-training baseline (1 run). Evaluate each on retrieval (ViDoRe nDCG@5) and grounding (BBox-DocVQA IoU@0.5). This 8-run experiment is the minimum needed to establish that the three-loss formulation is beneficial. Do this at small scale (1M pages, 2B model) first.
2. **Scaling Curve:** Plot retrieval and grounding performance as a function of pre-training data size (100K, 500K, 1M, 5M, 10M pages). This reveals whether the approach benefits from scale or saturates early, which determines practical significance.

---

## Idea 6: Sparse Grounding Tokens -- Efficient Localization via Learnable Spatial Anchors

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 7/10 | The architecture is technically sound. Gaussian-biased cross-attention from learnable anchor tokens to patch embeddings is well-defined. The spatial regularization (KL divergence with uniform distribution) is a reasonable way to prevent anchor collapse. The localization via anchor positions (mu_k) is clean and elegant. The main concern is whether K=16-32 is sufficient for retrieval quality -- this is an empirical question but the theoretical justification (information bottleneck) is missing. |
| Novelty | 8/10 | This is architecturally creative. The idea of learnable spatial anchors with trainable positions and receptive fields is novel in the document retrieval context. The connection to DETR's object queries is apt but the application is distinct. The Gaussian attention bias that creates spatially-grounded tokens is an interesting architectural primitive. The explicit spatial regularization to prevent collapse is also a nice touch. |
| Significance | 7/10 | The 32-64x compression over ColPali (16-32 tokens vs. 1024) is highly significant for deployment. If the retrieval quality holds, this would make multi-vector document retrieval practical at much larger scale. The dual-purpose tokens (efficient retrieval + localization) address two problems simultaneously. |
| Clarity | 7/10 | The architecture is well-described. The attention mechanism, training losses, and inference procedure are all concrete. Some ambiguity around: (a) how exactly the anchor positions are initialized and optimized, (b) whether sigma_k is per-anchor or shared, (c) the exact cross-attention architecture (number of layers, heads, dimensions). |
| Experimental Design | 5/10 | The critical experiment -- ablation on K (number of anchors) -- is mentioned but no other experiments are described. For an architecture paper, the experimental bar is high: (a) comparison against ColPali, DocPruner, HPC-ColPali at matched storage budgets, (b) comparison against perceiver/resampler architectures, (c) visualization of learned anchor positions and receptive fields, (d) analysis of failure cases where K tokens are insufficient, (e) scaling behavior (does the approach improve with larger base VLMs?). |
| Reproducibility | 7/10 | The architecture is relatively simple and well-defined. Implementation requires a custom cross-attention layer with Gaussian bias, which is straightforward. The training procedure (contrastive + regularization + localization losses) is clear. A competent deep learning engineer could implement this. |
| Weakness Severity | 5/10 | The fundamental tension between compression and expressiveness is serious. For queries about small details (a specific number in a dense table, a date in fine print), 16-32 tokens may simply not have the capacity to encode sufficient information. ColPali's strength is that it preserves fine-grained patch-level information; this approach explicitly discards it. The Gaussian attention bias may also over-constrain the model, preventing anchors from attending to semantically relevant but spatially distant patches. |

**Overall Score: 6.0/10**
**Predicted Verdict: Borderline Accept**

### Strengths
- **S1:** Elegant architectural solution to the efficiency-localization tradeoff. Instead of post-hoc pruning (DocPruner, HPC-ColPali), this learns to compress from the start, which is principled and likely more effective.
- **S2:** Localization "for free" -- the anchor positions directly provide spatial grounding without any additional computation. This is a beautiful design where efficiency and localization are not in tension but are actually unified.
- **S3:** The connection to DETR's object queries and perceiver architectures, but with the novel addition of explicit spatial priors (Gaussian bias), positions this work at an interesting intersection of detection and retrieval research.

### Weaknesses
- **W1:** The K=16-32 bottleneck may be too aggressive for information-dense documents. A financial report with 50+ data points per page may simply require more than 32 tokens to adequately represent. The paper needs to show performance stratified by document information density.
- **W2:** The spatial regularization loss (uniform coverage) conflicts with the reality that document information is not uniformly distributed. Most content is in the main body, with margins largely empty. Forcing anchors to uniformly cover the page wastes tokens on low-information regions. An information-density-aware placement would be more principled.
- **W3:** Comparison against simple baselines is crucial but may be unflattering. Specifically, ColPali with random patch sampling (keep 32 random patches) or attention-weighted patch selection (keep 32 most attended patches) could achieve similar compression without the architectural complexity. If these naive baselines perform comparably, the spatial anchor contribution is diminished.

### Questions to Authors
- **Q1:** At K=16, what is the retrieval performance (nDCG@5 on ViDoRe) compared to full ColPali? If the gap is more than 5 points, the compression is too aggressive for practical use. Where is the sweet spot?
- **Q2:** Do the learned anchor positions converge to interpretable locations (e.g., header, body columns, figures)? Or are they uniformly distributed as the regularization loss encourages? Visualizations of learned (mu_k, sigma_k) across different document types would be highly informative.

### Missing References
- DETR (Carion et al., 2020) -- the acknowledged inspiration; should be discussed in depth.
- Perceiver (Jaegle et al., 2021) / Perceiver IO (Jaegle et al., 2022) -- the closest architectural relative; compression via learnable queries.
- Set Transformer (Lee et al., 2019) -- inducing points for attention compression; a relevant prior art.
- Q-Former (Li et al., 2023) -- BLIP-2's learnable queries for VLM compression; the most directly comparable architecture.
- Matryoshka Representation Learning (Kusupati et al., 2022) -- an alternative approach to multi-granularity efficiency.

### Suggested Experiments
1. **Compression-Quality Pareto Curve:** Sweep K in {4, 8, 16, 32, 64, 128, 256, 1024} and plot retrieval nDCG@5 and localization IoU@0.5 vs. K. Compare against DocPruner and HPC-ColPali at matched compression ratios. This is the single most important experiment: it reveals whether learned spatial anchors outperform post-hoc pruning at the same storage budget.
2. **Anchor Visualization and Specialization Analysis:** For a diverse set of documents, visualize the learned anchor positions and their attention distributions. Show that different anchors specialize for different document regions (e.g., one anchor for headers, one for tables, one for figures). If the anchors do not specialize, the spatial regularization may be counterproductive, and the paper should explore removing it.

---

## Idea 7: UnifiedRetGround -- A Single Autoregressive VLM for Retrieve-Ground-Answer

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Soundness | 5/10 | The core assumption -- that a 7B autoregressive model can reliably process 20 document images (20K+ visual tokens) in a single forward pass and produce accurate structured output -- is highly questionable. Current VLMs struggle with multi-image reasoning even at 4-5 images. The context length issue is acknowledged but not adequately addressed. Additionally, the structured output format (JSON with page index + bbox + answer) requires the model to solve three distinct tasks simultaneously, and errors in any component propagate to the final output. The phased training approach (multi-page QA then grounding then joint) is reasonable but the claim of "one forward pass" is misleading if a first-stage retriever (BiPali) is needed. |
| Novelty | 7/10 | Unifying retrieval, grounding, and generation into a single autoregressive model is conceptually novel. While individual components exist (generative retrieval in GENIUS, autoregressive grounding in Kosmos-2, multi-page QA in M3DocVQA), combining all three is new. The structured generation format for grounded retrieval is a practical contribution. |
| Significance | 6/10 | If it works, collapsing a 3-model pipeline into 1 model is significant for deployment simplicity. However, the first-stage retriever dependency means this is really a 2-model pipeline (BiPali + UnifiedRetGround), not a single model. The practical significance depends on whether the unified model actually outperforms the pipeline approach in accuracy, not just simplicity. Multi-model pipelines often outperform unified models on complex tasks. |
| Clarity | 7/10 | The input/output format, training phases, and inference pipeline are clearly described. The JSON-like output format is intuitive. The BiPali first-stage dependency is transparently stated. |
| Experimental Design | 5/10 | The description mentions using M3DocVQA and BBox-DocVQA but lacks specificity on evaluation protocol. Key missing elements: (a) comparison against the pipeline approach (ColPali + Florence-2 + Qwen2.5-VL) at matched compute budget, (b) scaling analysis of performance vs. number of input pages, (c) analysis of error compounding vs. pipeline approach, (d) ablation on each training phase, (e) evaluation on different document types. |
| Reproducibility | 6/10 | The approach requires fine-tuning a 7B VLM on multi-page inputs with bounding box annotations, which is resource-intensive (4 A100s). The training data exists but the specific data mixture, training hyperparameters, and structured output formatting details are crucial for reproduction. The BiPali first-stage adds another component to reproduce. |
| Weakness Severity | 3/10 | Multiple serious weaknesses: (1) The 20-page multi-image input is likely beyond the effective context window of current 7B models. At 1024 visual tokens per page, 20 pages = 20K tokens, leaving minimal budget for text. Quality will degrade significantly. (2) The model cannot do true retrieval -- it can only re-rank and ground among pre-filtered candidates. The title "retrieves, grounds, and answers" is therefore overclaiming. (3) The structured output format may lead to degenerate behaviors (always selecting page 1, generating trivial bounding boxes). (4) For large-scale deployment, this approach is actually LESS efficient than a pipeline because the 7B model must process all 20 page images, whereas a pipeline approach only processes the top-1 page for grounding and answering. |

**Overall Score: 4.5/10**
**Predicted Verdict: Reject**

### Strengths
- **S1:** Bold and ambitious vision of unifying the full pipeline. The idea of generating page index + bbox + answer in one autoregressive sequence is elegant and pushes the boundary of what VLMs can do.
- **S2:** Practical structured output format. The JSON-like generation with page selection, bounding box, and answer is a clean interface that could simplify downstream integration.
- **S3:** Leverages the strength of autoregressive models in joint reasoning -- the model can use information from the bounding box prediction to inform the answer generation, and vice versa, enabling consistency between grounding and answering.

### Weaknesses
- **W1:** Context length is a near-fatal limitation. Processing 20 full-resolution page images in a single forward pass exceeds the effective context of current 7B VLMs. The proposed mitigation (reduce to 5 pages or lower resolution to 512px) severely limits either the re-ranking power or the grounding precision, undermining the core value proposition.
- **W2:** Overclaiming -- this is a re-ranker/grounder, not a retriever. The system depends entirely on BiPali for initial retrieval. If BiPali fails to include the correct page in its top-20, UnifiedRetGround cannot recover. This is a fundamental limitation that the framing obscures.
- **W3:** Efficiency is actually worse than the pipeline. In a pipeline, ColPali retrieves the top page (fast), then Florence-2 grounds on 1 page (fast), then Qwen2.5-VL answers on 1 page (fast). In UnifiedRetGround, a 7B model must process 20 pages simultaneously. For a single query, the pipeline is likely faster. The unified approach only wins if deployment simplicity (fewer models) outweighs computational efficiency.

### Questions to Authors
- **Q1:** What is the accuracy of page selection when the number of input pages increases from 5 to 10 to 20? If page selection accuracy drops significantly at 20 pages (due to context length issues), the approach is fundamentally limited by VLM context windows.
- **Q2:** How does the end-to-end accuracy (correct page + IoU >= 0.5 + correct answer) compare to the pipeline approach (ColPali + Florence-2 + Qwen2.5-VL) on the same candidate pages? If the pipeline is more accurate, the unification provides simplicity at the cost of quality -- and reviewers will question whether the tradeoff is worthwhile.

### Missing References
- DocPedia (Feng et al., 2024) -- a generative document AI model that handles multiple document pages.
- mPLUG-DocOwl 2 (Ye et al., 2024) -- multi-page document understanding with a single VLM.
- TextMonkey (Liu et al., 2024) -- long-context document understanding that addresses the token limit problem.
- Many-Shot ICL (Agarwal et al., 2024) -- demonstrates how performance changes with more in-context examples; relevant for understanding the scaling behavior with multiple pages.

### Suggested Experiments
1. **Page Scaling Analysis:** Fix the query set and vary the number of candidate pages (2, 5, 10, 15, 20). Measure page selection accuracy, grounding IoU, and answer F1 at each level. This directly addresses the core feasibility question and is the minimum evidence needed to make this paper credible.
2. **Pipeline vs. Unified Comparison:** On the same candidate pages and queries, compare (a) UnifiedRetGround (single model), (b) ColPali re-ranking + Florence-2 grounding + Qwen2.5-VL answering (3-model pipeline), (c) ColPali re-ranking + Qwen2.5-VL grounding+answering (2-model pipeline). Report accuracy, latency, and memory for each. This is the experiment that determines whether unification is actually beneficial.

---

## Overall Ranking by Predicted Acceptance Likelihood

| Rank | Idea | Overall Score | Predicted Verdict | Best Venue Fit |
|------|------|---------------|-------------------|----------------|
| 1 | **Idea 1: ColPali-Ground** | 6.5/10 | Borderline Accept | ECCV 2026, CVPR 2027 |
| 2 | **Idea 6: Sparse Grounding Tokens** | 6.0/10 | Borderline Accept | NeurIPS 2026, ICLR 2027 |
| 3 | **Idea 2: GroundedViDoRe Benchmark** | 6.0/10 | Borderline Accept (Accept at D&B) | NeurIPS 2026 D&B Track |
| 4 | **Idea 3: Spatial Token Distillation** | 6.0/10 | Borderline Accept | EMNLP 2026, ACL 2026 |
| 5 | **Idea 5: Retrieval-Aware Pre-Training** | 5.5/10 | Borderline Reject | ICLR 2027 (if strong results) |
| 6 | **Idea 4: Hierarchical Multi-Granularity** | 5.0/10 | Borderline Reject | SIGIR 2026 |
| 7 | **Idea 7: UnifiedRetGround** | 4.5/10 | Reject | -- |

---

## Venue-Specific Predictions

### Likely to get into a top venue (with strong execution):
- **Idea 1 (ColPali-Ground):** Accept at ECCV or CVPR. The clear problem formulation, feasibility, and well-defined contribution make this a safe bet. Reviewers may dock points for limited novelty, but the practical utility and first-mover advantage on joint retrieval+grounding would carry it.
- **Idea 2 (GroundedViDoRe):** Accept at NeurIPS Datasets & Benchmarks track. Benchmark papers need to target the D&B track specifically; this would be borderline at a main track.
- **Idea 6 (Sparse Grounding Tokens):** Accept at NeurIPS or ICLR with strong empirical results. The architectural novelty is sufficient for these theory-friendly venues. The paper lives or dies on the compression-quality Pareto curve.

### Could get in with exceptional results:
- **Idea 3 (Spatial Token Distillation):** Accept at EMNLP or ACL. The NLP community values efficiency and distillation work. The OCR dependency is a weakness but not disqualifying. Needs very strong empirical comparison showing the student approaches the teacher.
- **Idea 5 (Retrieval-Aware Pre-Training):** Accept at ICLR 2027 if results are compelling. This is a high-risk/high-reward idea. If the three-loss pre-training demonstrably produces a superior foundation model, it is a strong paper. If the losses interfere and the model is mediocre, it is a clear reject.

### Unlikely to be accepted at a top venue:
- **Idea 4 (Hierarchical Multi-Granularity):** The fixed-grid limitation and lack of clear advantage over simpler baselines would draw strong criticism. Better suited for a workshop paper or a systems venue like SIGIR.
- **Idea 7 (UnifiedRetGround):** The context length limitation is near-fatal for the claimed contribution. The overclaiming (it is a re-ranker, not a retriever) would draw immediate reviewer criticism. Would need a fundamental advance in long-context VLMs to become viable.

---

## The Single Best Idea: Idea 6 -- Sparse Grounding Tokens

**Justification:**

From a reviewer's perspective, the single best idea is **Idea 6 (Sparse Grounding Tokens)**, for the following reasons:

1. **Novelty is the hardest criterion to satisfy, and Idea 6 scores highest here alongside Idea 5.** The learnable spatial anchors with Gaussian attention bias represent a genuinely new architectural primitive. Unlike Idea 5, which requires massive compute to validate, Idea 6 can be validated on a single GPU, making it more likely to produce a strong paper.

2. **It solves two problems simultaneously.** The efficiency problem (1024 vectors per page is too expensive) and the localization problem (where on the page is the answer?) are both addressed by a single architectural innovation. This kind of "two-for-one" contribution is highly valued by reviewers.

3. **The story is clean and compelling.** "We replace 1024 dumb patches with 16-32 smart spatial anchors that know where they are looking" is a one-sentence pitch that any reviewer can immediately grasp. Clean stories lead to better reviews.

4. **The ablation space is rich and well-defined.** K (number of anchors), the Gaussian vs. uniform attention, the spatial regularization, the localization loss -- each can be ablated independently. Papers with clear ablation structures tend to receive higher scores because reviewers can see that the authors understand each contribution.

5. **The risks are manageable.** Unlike Idea 5 (multi-loss pre-training at scale) or Idea 7 (context length limitations), the risks of Idea 6 (insufficient compression vs. quality) can be quantified early through a simple K-sweep experiment. If K=32 preserves 95%+ of ColPali's retrieval quality, the paper is strong. If not, the idea can be abandoned quickly.

**Caveat:** Idea 1 (ColPali-Ground) is the safest bet for acceptance due to its high feasibility and clear incremental contribution. If the goal is to maximize the probability of a single acceptance, Idea 1 is the pragmatic choice. But Idea 6 has the highest ceiling -- if the results are strong, it could be a spotlight paper at NeurIPS or ICLR. Idea 1 is unlikely to be more than a poster.

**Strategic Recommendation:** Execute Idea 1 and Idea 2 first (safe bets, 3-4 months, complement each other). Then invest in Idea 6 as the high-upside paper (3-5 months). If Idea 6's early ablations look promising, make it the flagship submission. If not, pivot to Idea 3 as a strong backup.
