# 서베이: ColPali를 넘어서는 VLM 기반 이미지/문서 검색 (2024--2026)

**작성일: 2026-03-28**

본 서베이는 Vision-Language Model (VLM) 기반 문서/이미지 검색 시스템에 관한 최신 연구를 다루며, 검색과 공간적 grounding(문서나 이미지 내에서 정보가 어디에 위치하는지 파악하는 것)을 결합한 모델에 특별히 주목한다.

---

## 목차

1. [기반 Late-Interaction 검색](#1-기반-late-interaction-검색)
2. [Multi-Vector 검색을 위한 효율성 및 압축](#2-multi-vector-검색을-위한-효율성-및-압축)
3. [비전 기반 RAG 파이프라인](#3-비전-기반-rag-파이프라인)
4. [검색 + 공간적 Grounding](#4-검색--공간적-grounding)
5. [범용 멀티모달 검색](#5-범용-멀티모달-검색)
6. [Grounding 가능 VLM (보완적)](#6-grounding-가능-vlm-보완적)
7. [서베이 및 벤치마크](#7-서베이-및-벤치마크)
8. [갭 분석: 검색 + 위치 인식](#8-갭-분석-검색--위치-인식)

---

## 1. 기반 Late-Interaction 검색

### 1.1 ColPali: Efficient Document Retrieval with Vision Language Models

- **저자:** Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Celine Hudelot, Pierre Colombo
- **학회/연도:** ICLR 2025 (arXiv: 2024년 7월)
- **핵심 기여:** VLM에 late-interaction (ColBERT 방식) 스코어링을 적용하여 이미지에서 직접 문서 페이지를 검색하는 패러다임을 도입하며, OCR 파이프라인의 필요성을 제거한다. ViDoRe 벤치마크를 수립한다.
- **방법론:** PaliGemma-3B (SigLIP vision encoder + Gemma 2B LLM)를 사용하여 문서 페이지 이미지를 약 1,024개의 patch embedding으로 인코딩한다. 쿼리는 동일 모델의 텍스트 경로로 인코딩된다. 관련성은 쿼리 토큰 embedding과 문서 patch embedding 간의 MaxSim late interaction으로 계산된다.
- **한계점:** 페이지 수준 세분화만 가능하며, 하위 페이지 수준의 위치 파악이 불가능하다. 페이지당 약 1,024개의 128차원 벡터로 인해 저장 공간이 많이 필요하다. PaliGemma-3B의 시각적/언어적 능력에 제한된다.
- **검색 + 위치 인식:** ColPali의 patch별 유사도 점수는 암묵적으로 공간 정보를 인코딩한다. 원 논문은 검색 결정에 가장 크게 기여하는 이미지 patch를 보여주는 해석 가능성 히트맵을 시연했다. 그러나 bounding box나 명시적인 영역 좌표를 출력하지는 않는다.

### 1.2 ColQwen2: Late-Interaction Retrieval with Qwen2-VL

- **저자:** 동일 팀 (Illuin Tech / ViDoRe 협업)
- **학회/연도:** ColPali v1.2와 함께 공개 (2024년 후반)
- **핵심 기여:** PaliGemma를 Qwen2-VL-2B backbone으로 교체하여 ColPali 대비 nDCG@5에서 +5.3 향상을 달성한다. 동적 이미지 해상도를 지원하며 (종횡비 왜곡 없음) 최대 768개의 이미지 patch를 처리한다.
- **방법론:** ColPali와 동일한 late-interaction 프레임워크를 사용하되 더 강력한 VLM backbone을 적용한다. 가변 종횡비를 네이티브로 처리한다.
- **한계점:** 동일한 페이지 수준 세분화 한계가 존재한다. patch 수가 늘어남에 따라 메모리 요구량이 증가한다.
- **검색 + 위치 인식:** ColPali와 동일한 patch 수준 유사도를 통한 암묵적 공간 인식이 가능하나, 명시적 grounding 출력은 없다.

### 1.3 Nemotron ColEmbed V2 (NVIDIA)

- **저자:** Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Oliver Holworthy, Benedikt Schifferer, Zhiding Yu, Yauhen Babakhin, Radek Osmulski 외
- **학회/연도:** arXiv, 2026년 2월
- **핵심 기여:** ViDoRe V3 리더보드에서 최고 성능 달성 (8B 모델의 NDCG@10 = 63.42). NVIDIA Eagle 2 및 Qwen3-VL backbone 기반의 세 가지 모델 변형 (3B/4B/8B)을 공개한다.
- **방법론:** 클러스터 기반 샘플링, hard-negative 마이닝, 양방향 어텐션, late interaction, 모델 병합을 활용한다. 각기 다른 VLM backbone에 기반한 세 가지 변형이 있다.
- **한계점:** 대형 모델 크기 (3B--8B). 동일한 페이지 수준 검색 패러다임. multi-vector embedding의 저장 오버헤드가 지속된다.
- **검색 + 위치 인식:** late-interaction 메커니즘에서 patch 수준 공간 정보를 상속받지만, 명시적인 공간 출력을 생성하지는 않는다.

### 1.4 CausalEmbed: Auto-Regressive Multi-Vector Generation for Visual Document Embedding

- **저자:** Jiahao Huo, Yu Huang, Yibo Yan, Ye Pan, Yi Cao, Mingdong Ou, Philip S. Yu, Xuming Hu
- **학회/연도:** arXiv, 2026년 1월
- **핵심 기여:** multi-vector embedding 생성을 직접적인 patch 추출이 아닌 auto-regressive 생성 과정으로 재정의한다. 경쟁력 있는 성능을 유지하면서 토큰 수를 30--155배 줄인다.
- **방법론:** 사전 학습된 MLLM을 fine-tuning하여 컴팩트한 잠재 벡터를 순차적으로 합성하며, 대조 학습 중 반복적 마진 loss를 사용한다. 테스트 시 표현 크기를 유연하게 조절할 수 있다.
- **한계점:** auto-regressive 생성은 단일 패스 patch 추출에 비해 추론 지연을 추가한다. 새로운 패러다임으로 생태계 지원이 부족하다.
- **검색 + 위치 인식:** 생성된 벡터는 추상적인 잠재 표현으로, patch 수준 embedding이 갖는 직접적인 공간 대응 관계를 잃게 된다. 공간적 grounding을 위해서는 추가적인 매핑이 필요하다.

### 1.5 jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval

- **저자:** Jina AI 팀
- **학회/연도:** ACL MRL Workshop 2025 (arXiv: 2025년 6월)
- **핵심 기여:** 단일 backbone에서 텍스트, 이미지, 교차 모달 검색을 위한 단일 벡터 및 multi-vector (late-interaction) embedding을 모두 지원하는 38억 파라미터 모델. 시각적으로 풍부한 문서 검색을 위한 JVDR 벤치마크를 도입한다.
- **방법론:** Qwen2.5-VL-3B-Instruct를 기반으로 구축된다. 검색, 텍스트 매칭, 코드 작업을 위한 세 개의 태스크별 LoRA 어댑터 (각 60M)를 사용한다. 단일 벡터 모드는 2048차원 embedding을 생성하며 (MRL을 통해 128까지 축소 가능), multi-vector 모드는 late interaction을 위한 토큰당 128차원 embedding을 생성한다.
- **한계점:** 이중 모드는 아키텍처 복잡성을 추가한다. multi-vector 모드는 여전히 저장 오버헤드를 수반한다.
- **검색 + 위치 인식:** multi-vector 모드는 patch 수준 공간 대응을 유지한다. 명시적 grounding이나 bounding box 출력은 없다.

---

## 2. Multi-Vector 검색을 위한 효율성 및 압축

### 2.1 HPC-ColPali: Hierarchical Patch Compression for ColPali

- **저자:** (검색 결과에 완전히 명시되지 않음)
- **학회/연도:** SCITEPRESS 2025 / arXiv 2025년 6월
- **핵심 기여:** 세 가지 압축 전략을 통해 ColPali의 multi-vector embedding 저장 병목 현상을 해결한다.
- **방법론:** (1) K-Means 양자화로 patch embedding을 1바이트 중심점 인덱스로 압축 (최대 32배 저장 공간 감소); (2) 어텐션 기반 동적 프루닝으로 가장 핵심적인 상위 p%의 patch만 유지 (nDCG@10 손실 2% 미만으로 최대 60% 연산 감소); (3) 해밍 거리 검색을 위한 선택적 이진 인코딩.
- **한계점:** 손실 압축이며, 공격적인 프루닝은 성능을 저하시킨다. 양자화 클러스터가 의미적으로 구별되는 patch를 병합할 수 있다.
- **검색 + 위치 인식:** 어텐션 기반 프루닝은 어떤 patch가 가장 정보적인지를 드러내어 암묵적인 공간 신호를 제공한다. 프루닝된 patch는 공간 좌표를 유지한다.

### 2.2 DocPruner: Storage-Efficient Multi-Vector Visual Document Retrieval

- **저자:** (arXiv: 2509.23883)
- **학회/연도:** arXiv, 2025년 9월
- **핵심 기여:** VDR을 위한 적응형 patch 수준 embedding 프루닝을 사용하는 최초의 프레임워크. 검색 성능 저하를 최소화하면서 50--60%의 저장 공간 감소를 달성한다.
- **방법론:** 문서별 어텐션 점수 분포를 활용하여 프루닝을 위한 문서별 통계적 임계값을 계산한다. 정보 밀도와 문서 복잡성에 따라 프루닝 비율을 동적으로 조정한다.
- **한계점:** 프루닝이 문서별로 이루어져 배치 처리가 복잡해진다. 압축률이 60%를 넘으면 성능이 급격히 저하된다.
- **검색 + 위치 인식:** 유지된 patch는 공간적 위치를 보존하여, 프루닝된 집합에 대해 사후 공간 추론이 가능하다.

### 2.3 Sculpting the Vector Space: Prune-then-Merge Framework

- **저자:** (arXiv: 2602.19549)
- **학회/연도:** arXiv, 2026년 2월
- **핵심 기여:** 무손실에 가까운 압축 경계를 50--60% (DocPruner)에서 60--70%로 확장하고, 높은 압축률(80% 이상)에서의 급격한 성능 하락을 방지하는 2단계 접근법.
- **방법론:** 1단계 (Prune): 적응형 프루닝으로 정보가 적은 patch를 필터링한다. 2단계 (Merge): 계층적 병합으로 노이즈 유발 특성 희석 없이 의미적 내용을 요약하여 나머지 집합을 압축한다. ColQwen2.5, ColNomic, Jina-v4를 사용하여 29개의 VDR 데이터셋에서 평가한다.
- **한계점:** 2단계 파이프라인은 복잡성을 추가한다. 병합 단계에서 인접 patch 간의 세밀한 공간적 차이가 손실될 수 있다.
- **검색 + 위치 인식:** 프루닝 단계는 공간적으로 유의미한 patch를 유지하지만, 병합 단계에서 공간 정보가 집약되어 위치 파악 세분화가 감소할 수 있다.

### 2.4 ColParse: Beyond the Grid -- Layout-Informed Multi-Vector Retrieval

- **저자:** Yibo Yan, Mingdong Ou, Yi Cao, Xin Zou, Shuliang Liu, Jiahao Huo, Yu Huang, James Kwok, Xuming Hu
- **학회/연도:** arXiv, 2026년 3월
- **핵심 기여:** 균일한 patch 그리드를 문서 파싱 모델의 레이아웃 인식 하위 이미지 embedding으로 대체하여 95% 이상의 저장 공간 감소를 달성하면서 성능을 오히려 향상시킨다.
- **방법론:** 문서 파싱 모델을 사용하여 페이지를 의미적으로 유의미한 영역(문단, 표, 그림 등)으로 분할한다. 각 영역은 컴팩트한 하위 이미지 embedding으로 인코딩되고, 전역 페이지 수준 벡터와 결합되어 구조 인식 multi-vector 표현을 생성한다.
- **한계점:** 문서 파싱 모델의 품질에 의존한다. 인덱싱 파이프라인에 파싱 단계가 추가된다.
- **검색 + 위치 인식:** **높은 관련성.** 알려진 공간적 범위를 가진 의미적으로 유의미한 영역을 식별하기 위해 문서 파싱을 사용하므로, 표현 자체에 본질적으로 레이아웃/공간 정보가 포함된다. 검색 + 위치 인식 목표에 가장 가까운 논문 중 하나이다.

---

## 3. 비전 기반 RAG 파이프라인

### 3.1 VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents

- **저자:** Shi Yu, Chaoyue Tang 외 (OpenBMB)
- **학회/연도:** ICLR 2025 (arXiv: 2024년 10월)
- **핵심 기여:** 문서를 직접 이미지로 임베딩하여 검색하고, 검색된 페이지 이미지를 VLM 생성기에 입력하는 최초의 엔드투엔드 VLM 기반 RAG 파이프라인. 텍스트 기반 RAG 대비 20--40%의 엔드투엔드 성능 향상을 달성한다.
- **방법론:** VisRAG-Ret은 VLM을 사용하여 문서 페이지 스크린샷을 dense embedding으로 인코딩하여 검색한다. VisRAG-Gen은 생성형 VLM을 사용하여 검색된 페이지 이미지에서 답변을 생성하며, 모든 시각 정보(레이아웃, 그림, 표)를 보존한다.
- **한계점:** 페이지 수준 검색 세분화. 검색된 페이지 내 특정 영역을 식별하지 못한다. 다수의 페이지 이미지에 대한 VLM 기반 생성의 높은 연산 비용.
- **검색 + 위치 인식:** 페이지 수준에서 검색한다. 하위 페이지 수준 grounding은 없다.

### 3.2 VisRAG 2.0 (EVisRAG): Evidence-Guided Multi-Image Reasoning

- **저자:** OpenBMB 팀
- **학회/연도:** arXiv, 2025년 10월
- **핵심 기여:** VisRAG를 이미지별 근거 기록 및 집약으로 확장한다. 시각적 지각과 추론을 동시에 최적화하기 위한 RS-GRPO (Reward-Scoped Group Relative Policy Optimization)를 도입한다.
- **방법론:** 이미지를 검색한 후, 모델은 먼저 각 이미지를 관찰하고 이미지별 근거를 기록한 다음, 집약된 근거로부터 최종 답변을 도출한다. RS-GRPO는 세분화된 보상을 범위별 토큰에 연결한다.
- **한계점:** 여전히 페이지 수준 검색으로 동작한다. 근거는 영역 단위가 아닌 이미지 단위로 기록된다.
- **검색 + 위치 인식:** 근거 기록은 암묵적 grounding(어떤 페이지에 관련 정보가 있는지)을 제공하지만, 페이지 내 위치를 특정하지는 않는다.

### 3.3 VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents

- **저자:** Ryota Tanaka 외
- **학회/연도:** CVPR 2025 (arXiv: 2025년 4월)
- **핵심 기여:** 시각 정보를 dense 토큰 표현으로 압축하면서 텍스트 내용과 정렬하는 자기지도 pre-training 태스크를 제안하여 VLM을 검색에 적응시킨다. OpenDocVQA 벤치마크를 도입한다.
- **방법론:** VDocRetriever는 문서 이미지를 dense embedding으로 인코딩한다. VDocGenerator는 검색된 문서 이미지를 처리하여 답변을 생성한다. 새로운 pre-training은 라벨링된 데이터 없이 시각 토큰과 문서 텍스트를 정렬한다.
- **한계점:** 페이지 수준 검색. 자기지도 pre-training은 학습 오버헤드를 추가한다.
- **검색 + 위치 인식:** 전체 페이지를 검색한다. 공간적 grounding 출력은 없다.

### 3.4 ViDoRAG: Visual Document RAG via Dynamic Iterative Reasoning Agents

- **저자:** Alibaba NLP 팀
- **학회/연도:** EMNLP 2025 (arXiv: 2025년 2월)
- **핵심 기여:** 시각적 문서에 대한 복잡한 추론을 위한 반복적 탐색, 요약 및 성찰 기능을 갖춘 다중 에이전트 RAG 프레임워크. ViDoSeek 벤치마크를 도입한다. 기존 방법 대비 10% 이상 성능을 초과한다.
- **방법론:** 멀티모달 검색을 위한 GMM 기반 하이브리드 전략을 사용한다. 액터-크리틱 다중 에이전트 패러다임이 검색과 추론을 반복적으로 개선한다. 생성의 노이즈 견고성을 향상시킨다.
- **한계점:** 복잡한 다중 에이전트 아키텍처. 반복적 추론으로 인한 높은 지연 시간. 페이지 수준 검색.
- **검색 + 위치 인식:** 반복적 추론은 관련 정보를 식별하는 데 도움이 되지만, 문서 내 공간 좌표를 생성하지는 않는다.

### 3.5 RegionRAG: Region-Level Retrieval-Augmented Generation

- **저자:** (arXiv: 2510.27261)
- **학회/연도:** arXiv, 2025년 10월
- **핵심 기여:** **검색을 문서 수준에서 영역 수준으로 전환한다.** 페이지 수준 방법 대비 시각 토큰의 71.42%만 사용하면서 평균 R@1에서 +10.02% 향상을 달성한다.
- **방법론:** 학습 중, 라벨링된 bounding box 데이터와 라벨링되지 않은 데이터의 의사 라벨을 결합하는 하이브리드 감독 전략을 사용하여 쿼리 관련 patch를 식별한다. BFS를 통해 핵심 patch를 클러스터링하여 시각 영역 크롭을 생성한다. 이중 목표 대조 loss로 검색기를 학습한다.
- **한계점:** bounding box 주석이 (적어도 부분적으로) 필요하다. BFS 기반 영역 검출은 일관성 없는 영역 경계를 생성할 수 있다. 영역 품질은 patch 핵심도 추정에 의존한다.
- **검색 + 위치 인식:** **높은 관련성.** 영역 수준에서 직접 검색하므로, 각 검색 결과가 문서 페이지 내 공간적 영역에 대응한다. 검색 + 위치 인식과 가장 관련이 깊은 논문 중 하나이다.

---

## 4. 검색 + 공간적 Grounding (핵심 대상 영역)

### 4.1 Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation

- **저자:** Athos Georgiou
- **학회/연도:** arXiv, 2025년 12월 (2026년 1월 수정)
- **핵심 기여:** **가장 직접적으로 관련된 논문.** Late-interaction 멀티모달 검색(ColPali/ColQwen)과 OCR 기반 시스템을 연결하여 bounding box 좌표가 포함된 공간적으로 grounding된 검색 결과를 제공하는 하이브리드 아키텍처를 제안한다.
- **방법론:** ColPali/ColQwen 방식의 모델을 사용하여 patch 수준 유사도 점수를 생성한 다음, vision transformer patch 그리드와 OCR bounding box 간의 공식화된 좌표 매핑을 통해 이 점수를 OCR 추출 텍스트 영역에 전파한다. 관련성 전파를 위한 교차 지표(IoU 가중)를 도입한다. 추가 학습 없이 추론 시에 동작한다.
- **결과:** BBox-DocVQA에서 ColQwen3-4B에 백분위수-50 임계값 적용 시 IoU@0.5에서 59.7% 적중률 (IoU@0.25에서 84.4%)을 달성한다. 전체 OCR 영역 대비 컨텍스트 토큰을 28.8%, 전체 페이지 이미지 토큰 대비 52.3% 감소시킨다.
- **한계점:** 영역 추출을 위해 OCR 품질에 의존한다. OCR로 감지된 텍스트 영역에만 grounding이 가능하다(그림이나 차트와 같은 임의의 시각적 영역에는 적용 불가). IoU 임계값은 조정이 필요한 하이퍼파라미터이다.
- **검색 + 위치 인식:** **최고 관련성.** 이 논문은 검색 + 위치 인식 문제를 직접 다룬다. 관련 문서 페이지를 검색하고 해당 페이지 내에서 쿼리와 가장 관련된 특정 영역의 bounding box 좌표를 출력한다. "Snappy"라는 오픈소스 구현체가 있다.

### 4.2 EaGERS: Spatially Grounded Explanations in VLMs for Document VQA

- **저자:** (arXiv: 2507.12490)
- **학회/연도:** arXiv, 2025년 7월
- **핵심 기여:** 학습이 필요 없는 파이프라인으로, 자연어 설명을 생성하고 구성 가능한 그리드 위에서 멀티모달 embedding 유사도를 통해 공간적 하위 영역에 grounding한 다음, 마스킹된 이미지에 대해 모델을 재질의한다.
- **방법론:** (1) VLM이 이미지 + 질문으로부터 공간적 자연어 설명을 생성한다; (2) 이미지를 m x n 그리드로 분할한다; (3) BLIP, CLIP, ALIGN을 사용하여 설명과 각 하위 영역의 embedding을 추출한다; (4) 다수결 투표로 가장 관련 있는 영역을 선택한다; (5) 선택된 영역만 남기도록 이미지를 마스킹하여 최종 답변을 생성한다.
- **한계점:** 그리드 기반 분할은 거칠고 문서 레이아웃과 정렬되지 않을 수 있다. 다중 모델 파이프라인 (BLIP + CLIP + ALIGN + VLM 필요). 검색용으로 설계되지 않았으며 VQA에 초점을 맞춘다.
- **검색 + 위치 인식:** 단일 문서 페이지 내에서 공간적 grounding을 제공하지만 검색 시스템은 아니다. 검색 후 grounding 단계로 적용될 수 있는 잠재력이 있다.

### 4.3 BBox-DocVQA: Bounding-Box-Grounded Dataset for Document VQA

- **저자:** (arXiv: 2511.15090)
- **학회/연도:** arXiv, 2025년 11월
- **핵심 기여:** 모든 답변이 명시적 bounding box에 grounding된 대규모 벤치마크 (3.6K 문서, 32K QA 쌍). 최첨단 VLM(GPT, Qwen2.5-VL, InternVL)이 여전히 공간적 grounding에 어려움을 겪고 있음을 밝힌다.
- **방법론:** 자동화된 "Segment-Judge-Generate" 파이프라인: 영역 분할을 위한 분할 모델, 의미적 판단을 위한 VLM, QA 생성을 위한 고급 VLM, 이후 인간 검증. 단일/다중 영역 및 단일/다중 페이지 시나리오를 포괄한다.
- **한계점:** 데이터셋만 제공 (모델 아님). arXiv의 학술 논문에 한정된다. bounding box는 텍스트/표/그림 영역에 대한 것이며, 임의의 시각적 요소에 대한 것이 아니다.
- **검색 + 위치 인식:** 벤치마크로서 **높은 관련성**. 검색 + 위치 인식 시스템을 테스트하는 데 필요한 평가 프레임워크를 제공한다. BBox-DocVQA에서의 fine-tuning은 bounding box 위치 파악과 답변 생성을 모두 향상시킨다.

---

## 5. 범용 멀티모달 검색

### 5.1 MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs

- **저자:** NVIDIA 팀
- **학회/연도:** ICLR 2025 (arXiv: 2024년 11월)
- **핵심 기여:** 10개 데이터셋의 16개 검색 태스크에 걸쳐 bi-encoder 검색기로 fine-tuning된 MLLM. M-BEIR(멀티모달) 및 MTEB(텍스트 전용) 검색 벤치마크에서 동시에 최고 성능을 달성한다.
- **방법론:** MLLM을 dense bi-encoder 검색기로 fine-tuning한다. 모달리티 편향을 완화하기 위해 모달리티 인식 hard negative 마이닝을 제안한다. 프롬프팅을 통한 제로샷 MLLM 리랭킹도 탐구한다.
- **한계점:** 단일 벡터 embedding은 세밀한 patch 수준 상호작용이 부족하다. 공간적 grounding 없음. dense retrieval만 지원.
- **검색 + 위치 인식:** 전체 문서/이미지를 검색한다. 공간적 위치 파악 없음.

### 5.2 GME: Bridging Modalities -- Improving Universal Multimodal Retrieval by MLLMs

- **저자:** Zhang 외
- **학회/연도:** CVPR 2025
- **핵심 기여:** 대규모 융합 모달 학습 데이터를 구축하기 위한 학습 데이터 합성 파이프라인을 개발하고, 범용 멀티모달 검색을 위한 General Multimodal Embedder (GME)를 구축한다. UMRB 벤치마크를 구성한다.
- **방법론:** 기존 학습 세트의 모달리티 불균형을 해결하는 합성 멀티모달 학습 데이터를 활용한 MLLM 기반 dense 검색기. 텍스트, 이미지, 융합 모달리티에 걸친 다양한 쿼리-후보 쌍으로 학습한다.
- **한계점:** 단일 벡터 dense retrieval. late-interaction이나 공간 인식 없음. 합성 데이터 품질이 다를 수 있다.
- **검색 + 위치 인식:** 전체 항목을 검색한다. 공간적 위치 파악 없음.

### 5.3 UniIR: Training and Benchmarking Universal Multimodal Information Retrievers

- **저자:** TIGER Lab
- **학회/연도:** ECCV 2024
- **핵심 기여:** 8개의 서로 다른 검색 태스크를 위한 10개의 멀티모달 IR 데이터셋에서 공동 학습된 단일 검색 시스템. M-BEIR 벤치마크를 도입한다. 인스트럭션 튜닝이 교차 태스크 일반화의 핵심임을 입증한다.
- **방법론:** CLIP/BLIP2 backbone 기반의 인스트럭션 가이드 멀티모달 검색기. 사용자 인스트럭션이 원하는 검색 태스크를 지정한다. 다중 태스크 학습으로 새로운 태스크에 대한 제로샷 일반화가 가능하다.
- **한계점:** 구형 CLIP/BLIP2 backbone에 기반 (VLM이 아님). 단일 벡터 embedding. 문서 특화 기능(레이아웃 인식 등) 없음.
- **검색 + 위치 인식:** 전체 항목을 검색한다. 공간적 위치 파악 없음.

### 5.4 GENIUS: A Generative Framework for Universal Multimodal Search

- **저자:** Sung-Yeon Kim 외 (Amazon)
- **학회/연도:** CVPR 2025
- **핵심 기여:** embedding 기반 방법 대비 99% 이상의 저장 공간 감소를 달성하는 멀티모달 검색을 위한 생성적 검색 접근법. 모달리티 분리 의미 양자화를 도입한다.
- **방법론:** 잔차 양자화를 통해 멀티모달 데이터를 이산 ID로 변환한다 (첫 번째 레벨은 모달리티를 인코딩하고, 이후 레벨은 의미를 포착한다). auto-regressive 디코더가 쿼리 시 이 ID를 생성한다. 쿼리 증강은 쿼리와 대상 간을 보간한다.
- **한계점:** 생성적 검색 패러다임은 embedding 기반보다 성숙도가 낮다. 이산 ID는 세밀한 정보를 잃을 수 있다. 공간 인식 없음.
- **검색 + 위치 인식:** 전체 항목을 검색한다. 공간적 위치 파악 없음. 이산 ID 패러다임은 본질적으로 공간적 grounding과 상충한다.

### 5.5 Qwen3-VL-Embedding & Qwen3-VL-Reranker

- **저자:** Qwen 팀, Alibaba Group
- **학회/연도:** arXiv, 2026년 1월
- **핵심 기여:** 멀티모달 검색 및 랭킹을 위한 통합 프레임워크로, MMEB-V2에서 1위 달성 (8B 모델 점수: 77.8). 텍스트, 이미지, 스크린샷, 비디오를 처리한다.
- **방법론:** Qwen3-VL 파운데이션 모델 기반으로 구축된다. 검색을 위한 dense 단일 벡터 embedding과 후보 정제를 위한 별도의 리랭커 모델을 사용한다. JinaVDR 및 ViDoRe v3 데이터셋으로 학습된다.
- **한계점:** 단일 벡터 embedding (multi-vector/late-interaction 아님). 검색기와 리랭커가 별도 모델로 구성된다.
- **검색 + 위치 인식:** 전체 문서를 검색한다. 공간적 위치 파악 없음. 다만 기반 모델인 Qwen3-VL은 grounding 기능을 갖추고 있으나 embedding 인터페이스를 통해서는 노출되지 않는다.

---

## 6. Grounding 가능 VLM (보완적 -- 검색 네이티브 아님)

이 모델들은 공간적 grounding을 수행하지만 검색 시스템은 아니다. 검색 + grounding 파이프라인의 잠재적 구성 요소로서 관련이 있다.

### 6.1 SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models

- **저자:** An-Chieh Cheng, Hongxu Yin, Yang Fu 외 (UC San Diego, NVIDIA)
- **학회/연도:** NeurIPS 2024
- **핵심 기여:** 3D 장면 그래프 기반 데이터 큐레이션과 깊이 통합 플러그인 모듈을 통해 VLM의 공간 지각 능력을 향상시킨다. 사용자가 지정한 영역 간의 상대적 방향과 거리를 인식할 수 있다.
- **한계점:** 사용자가 지정한 영역 제안을 입력으로 요구한다. 검색 시스템이 아니다.
- **검색 + 위치 인식:** 강력한 공간 추론 능력을 갖추었으나 검색 시스템과 결합되어야 한다.

### 6.2 Groma: Localized Visual Tokenization for Grounding MLLMs

- **저자:** Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, Xiaojuan Qi (HKU, ByteDance)
- **학회/연도:** ECCV 2024
- **핵심 기여:** 영역 제안기를 통해 이미지를 관심 영역으로 분해하고 이를 영역 토큰으로 인코딩하며, 해당 토큰을 참조하여 텍스트 출력을 grounding한다. 명시적 좌표 회귀를 피한다.
- **한계점:** 검색 시스템이 아니다. 영역 제안기가 추론 오버헤드를 추가한다.
- **검색 + 위치 인식:** 뛰어난 grounding 기능 (영역 수준). 검색 후 grounding 모듈로 활용될 수 있다.

### 6.3 DocCogito: Layout Cognition and Step-Level Grounded Reasoning for Document Understanding

- **저자:** (arXiv: 2603.07494)
- **학회/연도:** arXiv, 2026년 3월
- **핵심 기여:** 전역 레이아웃 인지와 단계별 영역 grounding 추론을 결합한 OCR 없는 문서 이해 프레임워크. 6개 벤치마크 중 4개에서 최고 성능 달성.
- **방법론:** 경량 레이아웃 타워가 전역 레이아웃 사전 토큰을 생성한다. Visual-Semantic Chain (VSC)이 근거 영역과 정렬된 중간 추론을 감독한다. 점진적 학습: 레이아웃 지각 pre-training, VSC 가이드 콜드 스타트, 기각 샘플링, GRPO.
- **한계점:** 검색 시스템이 아니다. 문서 이해/QA에 초점을 맞춘다.
- **검색 + 위치 인식:** 문서 내에서 강력한 영역 grounding 추론 능력을 갖춘다. 엔드투엔드 검색 + grounding을 위해 검색 시스템과 결합될 수 있다.

### 6.4 LayoutLLM: Layout Instruction Tuning for Document Understanding

- **저자:** Luo 외 (Alibaba Research)
- **학회/연도:** CVPR 2024
- **핵심 기여:** 문서/영역/세그먼트 수준에서의 레이아웃 인식 pre-training과 질문 관련 영역에 집중하기 위한 LayoutCoT (Layout Chain-of-Thought).
- **한계점:** OCR + 레이아웃 입력이 필요하다. 검색 시스템이 아니다.
- **검색 + 위치 인식:** LayoutCoT는 영역 수준 집중을 가능하게 하여 검색 후 grounding에 관련이 있다.

---

## 7. 서베이 및 벤치마크

### 7.1 Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey

- **학회/연도:** AACL-IJCNLP 2025 (arXiv: 2025년 1월)
- **핵심 통찰:** VRD 검색에서 MLLM의 세 가지 역할을 체계화한다: (1) 모달리티 통합 캡셔너, (2) 멀티모달 임베더, (3) 엔드투엔드 표현기. 검색 세분화, 정보 충실도, 지연 시간, 리랭킹 및 grounding과의 호환성을 비교한다.

### 7.2 DSE: Unifying Multimodal Retrieval via Document Screenshot Embedding

- **저자:** Xueguang Ma 외
- **학회/연도:** EMNLP 2024 (arXiv: 2024년 6월)
- **핵심 기여:** VLM (Phi-3-vision, 4B)으로 문서 스크린샷을 직접 인코딩하면 효과적인 dense retrieval embedding을 생성할 수 있음을 초기에 입증한다. Wiki-SS에서 BM25를 17점 상회한다.
- **한계점:** 단일 벡터 embedding (late interaction 없음). 픽셀 포이즈닝 공격에 취약하다 (2025년 1월 후속 연구에서 밝혀짐).

### 7.3 FLMR / PreFLMR: Fine-Grained Late-Interaction Multi-Modal Retrieval

- **저자:** Weizhe Lin, Jingbiao Mei, Jinghong Chen, Bill Byrne
- **학회/연도:** FLMR: NeurIPS 2023; PreFLMR: ACL 2024
- **핵심 기여:** ColPali 방식 접근법의 선행 연구. 토큰 수준의 시각적 및 텍스트 특성을 교차 모달리티 late interaction이 포함된 다차원 embedding에 통합하여 RA-VQA에 활용한다. PreFLMR은 범용 멀티모달 검색기를 위한 M2KR 벤치마크를 도입한다.

---

## 8. 갭 분석: 검색 + 위치 인식

### 현재 최신 기술 현황

**검색**(올바른 문서/페이지 찾기)과 **위치 인식**(해당 문서/페이지 내에서 답변이 어디에 있는지 파악)을 결합하는 목표는 서베이된 논문들에 의해 다양한 수준으로 다루어지고 있다:

| 논문 | 검색 | 하위 페이지 위치 파악 | Bounding Box | 학습 불필요 |
|-------|-----------|----------------------|----------------|---------------|
| ColPali/ColQwen2 | 가능 (페이지 수준) | 암묵적 (히트맵) | 불가 | 해당 없음 |
| Patch-to-Region (Georgiou) | 가능 (페이지 수준) | 가능 (OCR 영역) | 가능 | 가능 |
| RegionRAG | 가능 (영역 수준) | 가능 (patch 클러스터) | 부분적 | 불가 |
| ColParse | 가능 (영역 수준) | 가능 (파싱 영역) | 암묵적 | 불가 |
| EaGERS | 불가 (VQA 전용) | 가능 (그리드 셀) | 불가 | 가능 |
| BBox-DocVQA | 벤치마크 전용 | 벤치마크 전용 | 가능 | 해당 없음 |
| DocCogito | 불가 (이해) | 가능 (VSC 영역) | 가능 | 불가 |
| Groma | 불가 (이해) | 가능 (영역 토큰) | 가능 | 불가 |

### 핵심 갭

현재 단일 시스템으로 다음을 모두 제공하는 것은 존재하지 않는다:
1. 문서 코퍼스에 대한 대규모 검색 (수백만 개 중에서 올바른 페이지 찾기)
2. 검색된 페이지 내에서 bounding box 좌표를 통한 세밀한 영역 위치 파악
3. OCR 없이 엔드투엔드 학습

**가장 근접한 접근법들:**

- **Patch-to-Region Relevance Propagation** (Georgiou, 2025): ColPali/ColQwen 검색과 OCR 기반 영역 grounding을 추론 시에 결합한다. 현재 가장 완전한 솔루션이나 OCR에 의존한다.
- **RegionRAG** (2025): patch 클러스터링을 통해 검색을 영역 수준으로 전환하지만, 거친 BFS 기반 영역 검출을 사용하며 학습 중 bounding box 감독이 필요하다.
- **ColParse** (2026): 암묵적 영역 인식과 95% 이상의 저장 공간 감소를 가진 레이아웃 인식 검색을 위해 문서 파싱을 사용하지만, 명시적으로 bounding box를 출력하지는 않는다.

### 검색 + 위치 인식을 위한 권장 아키텍처

본 서베이를 바탕으로 가장 유망한 접근법은 다음을 결합하는 것이다:
1. **검색 단계:** 페이지 수준 검색을 위한 ColQwen2/Nemotron ColEmbed V2
2. **위치 파악 단계:** patch 유사도를 bounding box가 있는 OCR 영역에 매핑하기 위한 Patch-to-Region Relevance Propagation (Georgiou)
3. **또는** 구조적 인식을 가진 영역 수준 검색을 위한 ColParse 방식 레이아웃 파싱
4. **선택적 grounding 정제:** 검색된 영역에 대한 단계별 grounding 추론을 위한 DocCogito 또는 Groma

---

## 출처

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
