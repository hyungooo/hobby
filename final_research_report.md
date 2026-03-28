# 최종 연구 아이디어 평가 보고서
## Image Retrieval with Spatial Localization: Beyond ColPali

**생성일:** 2026-03-28
**에이전트 팀:** 논문조사 x3, 아이디어생성 x1, 교수평가 x1, 리뷰어평가 x1

---

## Executive Summary

ColPali 이후 **"이미지를 검색하면서 동시에 해당 정보의 정확한 위치(bounding box)도 제공"**하는 시스템을 목표로, 40+ 최신 논문(2024-2026)을 분석하고 7개의 연구 아이디어를 생성한 뒤, 교수와 학회 리뷰어 관점에서 각각 독립 평가하였습니다.

### 핵심 발견: 현재 연구의 Gap

> **Retrieval과 Grounding은 여전히 분리된 파이프라인으로 존재한다.**
> ColPali는 페이지를 찾지만 "어디에" 정보가 있는지 모르고, Grounding 모델(Ferret, Florence-2)은 위치를 찾지만 대규모 검색을 하지 못한다. 유일한 브릿지인 Georgiou(2025)는 inference-time hack으로 IoU@0.5에서 59.7%에 불과하다.

---

## 최종 종합 랭킹

교수 가중 점수(50%)와 리뷰어 Overall Score(50%)를 결합한 종합 점수입니다.

| 순위 | 아이디어 | 교수 점수 | 리뷰어 점수 | **종합 점수** | 추천 등급 |
|:---:|---------|:---------:|:----------:|:-----------:|:--------:|
| **1** | **Idea 6: Sparse Grounding Tokens** | 8.05 | 6.0 | **7.03** | ⭐⭐⭐ 최우선 추천 |
| **2** | **Idea 3: Spatial Token Distillation** | 7.35 | 6.0 | **6.68** | ⭐⭐⭐ 강력 추천 |
| **3** | **Idea 2: GroundedViDoRe Benchmark** | 7.00 | 6.0 | **6.50** | ⭐⭐ 병행 추천 |
| **4** | **Idea 1: ColPali-Ground** | 6.25 | 6.5 | **6.38** | ⭐⭐ 안전한 선택 |
| **5** | **Idea 5: Retrieval-Aware Pre-Training** | 7.80 | 5.5 | **6.65** | ⭐ 장기 과제 |
| **6** | **Idea 4: Hierarchical Multi-Granularity** | 5.60 | 5.0 | **5.30** | 비추천 |
| **7** | **Idea 7: UnifiedRetGround** | 6.50 | 4.5 | **5.50** | 비추천 (시기상조) |

---

## Top 3 아이디어 상세 분석

### 1위: Sparse Grounding Tokens (Idea 6)
> *1024개 patch 대신 16-32개의 learnable spatial anchor로 검색과 위치를 동시에 해결*

**핵심 아이디어:**
ColPali는 페이지당 1024개 patch vector를 저장하지만, 대부분은 중복 정보. Learnable spatial anchor token에 Gaussian attention bias를 적용하여 16-32개의 "공간을 아는" 토큰으로 압축. 각 anchor의 학습된 위치(mu, sigma)가 곧 localization 정보가 됨.

| 평가 항목 | 교수 | 리뷰어 | 핵심 코멘트 |
|----------|:----:|:-----:|-----------|
| Novelty | 9 | 8 | "DETR의 object query를 retrieval에 적용한 새로운 architectural primitive" |
| Technical Depth | 8 | 7 | Gaussian attention + KL regularization + multi-task loss |
| Feasibility | 6 | 7 | 3-5개월, single GPU 가능 |
| Publication | 8 | 6 | NeurIPS/ICLR spotlight 가능성 |
| Impact | 8 | 7 | 32-64x 압축 + localization "for free" |

**교수 코멘트:**
> "K=1024에서 시작해서 하향 ablation하라. 성능이 유지되는 elbow point를 찾는 것이 곧 논문의 핵심 발견이 된다. 이 degradation curve가 논문 그 자체다."

**리뷰어 코멘트:**
> "Two-for-one contribution: efficiency와 localization을 하나의 아키텍처로 해결. Clean story, rich ablation space. Risks are manageable."

**핵심 실험:** K={4,8,16,32,64,128,256,1024} sweep → nDCG@5 vs IoU@0.5 Pareto curve

**타겟 학회:** NeurIPS 2026, ICLR 2027

---

### 2위: Spatial Token Distillation (Idea 3)
> *VLM의 공간 지식을 경량 text retriever로 증류하여 OCR+layout만으로 위치 정보 제공*

**핵심 아이디어:**
ColQwen2 teacher의 patch embedding을 IoU 기반 soft assignment matrix를 통해 OCR token embedding으로 distillation. Student는 vision encoder 없이도 공간적 localization이 가능한 text-only retriever가 됨. 10-50x 효율 향상.

| 평가 항목 | 교수 | 리뷰어 | 핵심 코멘트 |
|----------|:----:|:-----:|-----------|
| Novelty | 8 | 7 | "IoU-weighted spatial distillation은 genuinely novel" |
| Technical Depth | 7 | 7 | Geometric assignment matrix, confidence-aware loss |
| Feasibility | 7 | 7 | 4-5개월, teacher embedding은 한 번만 추출 |
| Publication | 7 | 6 | ICLR/EMNLP/ACL 가능 |
| Impact | 7 | 7 | 산업 배포에 즉시 적용 가능 |

**교수 코멘트:**
> "OCR 실패에 대비한 confidence-aware distillation을 추가하면 기술적 기여가 한 층 더해진다. VLM 추론 비용이 계속 하락해도, Pareto-optimal임을 증명하라."

**핵심 실험:** 문서 유형별(text-heavy/table-heavy/figure-heavy) teacher-student gap 분석

**타겟 학회:** ICLR 2027, ACL 2026, EMNLP 2026

---

### 3위: GroundedViDoRe Benchmark (Idea 2)
> *Retrieval + Localization을 동시에 평가하는 최초의 대규모 벤치마크*

**핵심 아이디어:**
50K 페이지, 5개 도메인, 3단계 난이도. 새로운 joint metric: **Grounded-Recall@K** (페이지 검색 + IoU 동시 달성 비율). 기존 시스템의 약점을 정량적으로 보여주는 baseline analysis 포함.

| 평가 항목 | 교수 | 리뷰어 | 핵심 코멘트 |
|----------|:----:|:-----:|-----------|
| Novelty | 7 | 5 | 벤치마크 자체의 novelty 기준은 다름 |
| Significance | 8 | 8 | "벤치마크를 정의하는 자가 연구 방향을 주도한다" |
| Feasibility | 9 | 9 | ~$5K 어노테이션 비용, 4-5개월 |
| Impact | 8 | 8 | 커뮤니티 채택 시 모든 후속 논문이 인용 |

**전략적 가치:** Idea 6, 3의 평가 인프라로 활용 + 독립 논문으로 NeurIPS D&B 제출

---

## 전략적 실행 로드맵

```
Month 1-2: GroundedViDoRe 벤치마크 구축 시작 + Sparse Grounding Tokens K-sweep 예비실험
           └─ K-sweep 결과로 Idea 6의 viability 조기 판단

Month 3-4: Idea 6 (Sparse Grounding Tokens) 본실험 + 벤치마크 완성
           └─ NeurIPS 2026 제출 준비

Month 5-6: Idea 3 (Spatial Token Distillation) 착수
           └─ Teacher embedding 추출 + distillation 실험

Month 7-8: Idea 3 완성 + GroundedViDoRe 독립 논문화
           └─ ICLR 2027 / NeurIPS D&B 제출

Month 9-12: (Optional) Idea 5 소규모 pilot → thesis paper로 확장
```

---

## 비추천 아이디어와 그 이유

### Idea 4: Hierarchical Multi-Granularity — 비추천
- **교수:** "Coarse-to-fine retrieval은 이미 well-worn path. Fixed grid는 문서 레이아웃과 근본적으로 misaligned."
- **리뷰어:** "ColPali + HNSW pre-filtering이라는 simple baseline을 이길 수 있는지 불분명."
- **종합:** 기술적 novelty 부족, 시스템 논문 성격 → top venue 어려움

### Idea 7: UnifiedRetGround — 비추천 (시기상조)
- **교수:** "Context length 문제가 minor detail이 아니라 central bottleneck. 5페이지로 제한하면 re-ranker일 뿐."
- **리뷰어:** "Overclaiming — retriever가 아니라 re-ranker. 파이프라인보다 오히려 느림."
- **종합:** 1-2년 후 long-context VLM 성숙 시 재검토

---

## 교수와 리뷰어의 관점 차이 분석

| 관점 | 교수 (지도 관점) | 리뷰어 (심사 관점) |
|------|---------------|-----------------|
| **Idea 5** | 2위 (7.80) — 높은 ceiling 평가 | 5위 (5.5) — compute 부족 우려 |
| **Idea 1** | 6위 (6.25) — "incremental" 우려 | 1위 (6.5) — 가장 안전한 accept |
| **Idea 6** | 1위 (8.05) — 창의성 최고 평가 | 2위 (6.0) — empirical risk 존재 |

> **시사점:** 교수는 novelty와 장기 성장 가능성을 중시하고, 리뷰어는 실현 가능성과 실험적 근거를 중시합니다. 최적 전략은 **교수가 높이 평가하는 창의적 아이디어(6, 3)를 리뷰어가 요구하는 수준의 실험으로 뒷받침**하는 것입니다.

---

## 참고: 조사된 주요 논문 목록

### VLM 기반 검색 (28편)
ColPali, ColQwen2/2.5, DSE, VisRAG, VDocRAG, Nemotron ColEmbed V2, HPC-ColPali, DocPruner, RegionRAG, ColParse, MM-Embed, GME, UniIR, GENIUS, Jina-Embeddings-v4, ModernVBERT, BBox-DocVQA, VISA, M3DocRAG, ViDoRAG, SERVAL, NL-DIR, Marten, etc.

### Visual Grounding & Localization (18편)
Grounding DINO 1.5, Grounded SAM, Florence-2, Ferret/v2, Kosmos-2, GLaMM, LISA, SpatialRGPT, Molmo/PixMo, Osprey, Qwen2.5-VL, LLaVA-Grounding, DocLLM, GEM, VDocRAG, GroundingGPT, etc.

### 문서 이해 & RAG (15편)
ColPali family, ViDoRe V1-V3, LayoutLLM, DocLLM, M3DocRAG, REAL-MM-RAG, MMDocBench, UniDoc-Bench, Pixel Poisoning, SERVAL, Nemotron, etc.

---

## 생성된 전체 파일 목록

| 파일 | 내용 |
|------|------|
| `research_ideas.md` | 7개 연구 아이디어 상세 기술서 |
| `professor_evaluation.md` | 교수 관점 7개 기준 평가 |
| `reviewer_evaluation.md` | 학회 리뷰어 관점 7개 기준 평가 |
| `final_research_report.md` | 이 최종 종합 보고서 |
