# CORTIS

**C**ontinuous financing **O**pportunity **R**eassessment & **T**ailored **I**ntelligent **S**ervice

청년 금융 생애주기 이벤트 예측 기반 부채 포트폴리오 재설계 서비스

**"대출은 한 번의 이벤트가 아니라, 인생 전체에 걸쳐 계속 재조정되어야 하는 대상이다."**

---

## 프로젝트 소개

CORTIS는 청년 대출자의 거래내역 변화를 스스로 관찰해 다음 생애주기 이벤트(취업·이직·독립·결혼 등)를 미리 예측하고, 그 이벤트가 실제로 발생했을 때 필요한 자금을 어떤 정책·대출 상품으로 조달하는 게 최선인지까지 계산해서 제안하는 AI 에이전트 서비스입니다.

기존 은행 서비스가 "지금 이 순간의 스냅샷"(신용점수, 자산유형 등 정적 지표 하나)에 반응하거나 사용자가 먼저 조회해야 작동하는 반면, CORTIS는 최근 거래 추세를 스스로 관찰해 여러 신호를 종합하고, 일이 벌어지기 전에 먼저 준비할 수 있게 합니다.

## 문제 정의

- **소득과 상환 구조의 미스매치**: 5대 은행 20대 이하 가계대출 연체율이 전 연령대 중 최고(0.44%, 2026.3말 기준)이며, 인터넷은행에서는 격차가 더 두드러집니다(토스뱅크 20대 신용대출 연체율 2.5%). 플랫폼노동·프리랜서·단기계약 등으로 소득 발생 시기가 일정하지 않은데 상환일은 고정돼 있다는 구조적 미스매치가 원인입니다.
- **정책금융의 역설**: 청년도약계좌 가입자의 평균 신용점수(876.2점)가 전체 청년 평균(814.1점)보다 62점 높습니다 — 정작 지원이 필요한 취약 청년일수록 정보 탐색 여력이 없어 정책상품에서 소외되는 구조입니다.
- **기존 서비스의 한계**: KB를 포함한 기존 뱅킹 앱은 신용점수·자산유형 같은 하나의 정적 신호에만 반응하거나, 사용자가 먼저 조회해야 작동합니다. "이번 달 무엇을 언제 얼마나 갚아야 하는가", "다음 이벤트에 어떤 자금이 필요한가" 같은 실행 판단까지는 내려주지 않습니다.

## 전체 아키텍처

CORTIS는 두 개 레이어가 순환 구조로 연결됩니다. **C가 메인 AI 추론 엔진**이고, **A는 C가 감지한 이벤트를 받아 정책·대출 자격을 판정하는 서포팅 레이어**입니다. (B: 상환관리 단기대응 서비스는 이번 제출 범위에서 팀 결정으로 제외했습니다 — 별도 인프라가 필요해 시간상 우선순위에서 밀림)

```
                    ┌─────────────────────────────────────────┐
                    │   C. 생애주기 이벤트 예측 (메인 엔진)      │
                    │                                           │
  거래내역 ──▶ Feature Extractor ──▶ State Builder              │
                    │        │                                  │
                    │        ▼                                  │
                    │  History / State / Transaction            │
                    │  3분리 임베딩 검색 (adaptive 가중합)        │
                    │        │                                  │
                    │        ▼                                  │
                    │  LLM Reasoning (Rerank + Critic)           │
                    │        │                                  │
                    │        ▼                                  │
                    │  Scenario Tree (depth-2)                   │
                    │        │                                  │
                    │        ▼                                  │
                    │  Strategy Generator (규칙기반 후보 생성)     │──▶  ┌───────────────────┐
                    │        │                                  │     │  A. 정책/대출 매칭   │
                    │        ▼                                  │◀──  │  (자격 판정 레이어)  │
                    │  Portfolio Optimizer (목적함수 4단계)        │     └───────────────────┘
                    │        │                                  │
                    │        ▼                                  │
                    │  LLM 설명 (숫자 검증가드 포함)                │
                    └─────────────────────────────────────────┘
                             │ 이벤트 확정 시 순환
                             ▼
                    (다시 History에 반영, 다음 예측 시작)
```

### C. 생애주기 이벤트 예측 — 단계별 설명

1. **Transaction Feature Extractor**: 최근 3개월 vs 이전 3개월 거래 추세(소득/지출/저축/**대출상환액** 증가율)와 현금흐름 변동성을 규칙기반으로 추출. 스냅샷이 아니라 **추세**를 본다는 게 핵심 — 청년 연체 문제의 원인이 "소득이 낮아서"가 아니라 "소득 발생이 불안정해서"라는 문제의식과 설계가 일치합니다.
2. **State Builder**: DB 프로필(나이/고용형태/신용점수/DSR 등) + 위 거래 feature를 하나의 State로 결합.
3. **History / State / Transaction 3분리 임베딩 검색**: 합성 코호트 300개와 유사도를 계산해 top-k를 검색합니다. 3개 공간을 하나로 합치지 않는 이유는, State만 강조하면 "이벤트 순서에 따른 조건부 추론"이라는 핵심 차별점이 희석되기 때문입니다. **가중치는 히스토리 길이(콜드스타트 여부)뿐 아니라, History 검색 자체가 특정 코호트에 집중되는지(엔트로피 기반)까지 반영해 3단계로 adaptive하게 조정**됩니다.
4. **LLM Reasoning (Rerank + Critic)**: 검색된 근거를 바탕으로 다음 이벤트의 확률·예상 시점·예상 필요자금을 추론합니다. **예상 필요자금은 LLM이 지어내지 않고, 검색된 코호트들의 실제 집계값을 그대로 사용**합니다 — 이 파이프라인 전체의 핵심 방어 논리입니다. Critic이 유저의 현재 상황과 논리적으로 모순되는 후보(예: 이미 월세 거주 중인데 "독립(월세)" 예측)를 걸러냅니다.
5. **Scenario Tree**: 다음 이벤트 하나가 아니라, depth-2까지 분기한 여러 시나리오(예: "독립→결혼", "이직→독립")를 확률과 함께 제시합니다.
6. **Strategy Generator**: 예측된 이벤트에 필요한 자금을 어떻게 조달할지 후보를 규칙기반으로 나열합니다(정책대출/일반대출/현금충당). 기존 대출이 있으면 대환/조기상환/유지 후보를, 없으면 신규 조달 후보를 생성합니다. **LLM은 후보를 만들지 않습니다** — A파트가 이미 자격 판정을 마친 상품만 후보로 들어옵니다.
7. **Portfolio Optimizer**: 후보들을 숫자로 평가해 최선을 고릅니다. 목적함수는 4단계 계층 구조입니다.
   - 1단계(Hard Constraint): 정책 자격(A파트가 이미 필터링) · DSR 상한 · 최소 유동성 유지
   - 2단계(Primary): 유동성 위험 분류에 따라 저위험=총비용(NPV) 최소화, 고위험=월상환액 최소화
   - 3단계(Secondary): 스트레스 금리 적용 시 DSR(worst-case)
   - 4단계(Tie-break): 정책형 상품 우선
8. **LLM 설명**: 최종 선택된 전략과 근거를 "언제/얼마나/효과/비교/다음 행동" 5항목으로 자연어 설명합니다. 출력에 포함된 모든 숫자가 실제 계산값에서 나왔는지 코드가 재검증합니다(검증가드).
9. **순환**: 예측했던 이벤트가 실제로 확정되면 히스토리에 반영되고, 다음 예측이 다시 시작됩니다.

### A. 정책/대출 매칭 레이어

- C가 이벤트를 감지·예측할 때마다 자동으로 재호출되어, 신규 자격 발생/상실 여부를 판정합니다.
- 정책 데이터는 두 소스로 구성됩니다: 수작업 큐레이션 정책(15건) + 온통청년 오픈API 연동(인증키 정식 발급 완료, API 서버 접근 이슈로 실행 검증은 불가 — 아래 [알려진 한계](#알려진-한계) 참고).
- 동일한 자격판정 로직을 Strategy Generator의 후보 필터링에도 재사용합니다.

## 핵심 차별점

| 구분 | 기존 뱅킹 서비스 | CORTIS |
|---|---|---|
| 판단 신호 | 신용점수 등 정적 지표 하나 | 소득/지출/저축/대출상환 추세 + 이벤트 히스토리 종합 |
| 반응 시점 | 문제 발생 후 사후 알림 | 이벤트 발생 전 사전 준비 |
| 추천 근거 | 설명 어려움 | 검색된 코호트 근거 + 규칙기반 후보 + 명시적 목적함수로 전부 설명 가능 |
| 숫자의 출처 | - | LLM이 숫자를 생성하지 않고, 전부 코드가 계산 후 검증가드로 재확인 |

## Tech Stack

| 분류 | 기술 |
|---|---|
| DB (Source of Truth) | MySQL — 유저 프로필, 대출 DB, 정책 DB, 이벤트 로그, 임베딩 벡터 |
| 임베딩 | 오프라인 해시 기반 임베딩(64차원) — 데이터 규모상 전용 벡터DB(ChromaDB 등) 불필요 |
| 유사도 검색 | numpy 코사인 유사도 (History/State/Transaction 3분리, adaptive 가중합) |
| 추론 | Anthropic Claude API — JSON-schema 강제 출력, 숫자 검증가드 포함 |
| 백엔드 | FastAPI |
| 계산 엔진 | 순수 Python 결정론적 함수 (NPV/DSR/스트레스테스트 — LLM 미개입) |

## 프로젝트 구조

```
CORTIS/
├── backend/                       # FastAPI 서버 + 데이터 계층
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── db/
│   │   ├── schema.sql                 # MySQL DDL
│   │   ├── models.py                  # SQLAlchemy ORM
│   │   ├── database.py
│   │   └── seed/                      # 정책·대출상품·데모유저·코호트 적재 스크립트
│   ├── matcher_a/                 # A: 규칙기반 이벤트 감지 + 정책/대출 자격 판정
│   │   ├── detector.py
│   │   └── policy_matcher.py
│   ├── routers/                   # API 엔드포인트 (users/events/predictions/policies/loans/cohorts)
│   └── integrations/              # 온통청년 오픈API 연동
│
├── pipeline/                      # C: 임베딩·검색·LLM 추론·전략생성·포트폴리오
│   ├── tx_features.py                 # Transaction Feature Extractor
│   ├── state_builder.py               # State Builder
│   ├── embedding.py                    # 오프라인 해시 임베딩
│   ├── similarity.py                   # History/State/Tx 3분리 검색 (adaptive 가중치)
│   ├── reasoning.py                    # LLM Reasoning (Rerank/Critic, confidence 계산)
│   ├── scenario_tree.py                # Scenario Tree
│   ├── strategy_generator.py           # Strategy Generator (신규 자금조달 후보)
│   ├── portfolio.py                    # Portfolio Optimizer (목적함수 4단계)
│   ├── portfolio_summary.py            # LLM 설명 + 숫자 검증가드
│   ├── agent_loop.py                   # Agentic 순환 로직 (C→A 트리거 연결)
│   └── backend_client.py               # 백엔드 API 클라이언트
│
├── data/
│   └── cohort_sequences_300.py    # 합성 코호트 300개 (generate_cohorts.py 산출물)
│
├── generate_cohorts.py             # 코호트 300개 생성 스크립트
├── demo_full_flow.py                # 실제 백엔드 연동 엔드투엔드 데모
├── demo_ab_contrast.py              # 오프라인 임베딩 대조 데모 (히스토리 다른 두 유저 비교)
└── doc/                             # 기획서, 팀 문서
```

## 실행 방법

### 1. 준비

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

copy backend\.env.example backend\.env    # macOS/Linux: cp
# backend/.env 를 열어 MYSQL_PASSWORD, ANTHROPIC_API_KEY 를 채운다.
# YOUTHCENTER_API_KEY(온통청년 오픈API 인증키)도 있으면 넣는다 — 없어도 시드는 정상 진행된다.
```

### 2. 코호트 생성 + DB 시드

```bash
python generate_cohorts.py            # data/cohort_sequences_300.py 생성
python -m backend.db.seed.run_all     # 스키마 적용 + 카탈로그 + 온통청년API(선택) + 데모유저 + 코호트 순으로 적재
```

시드 파일(코호트 생성 로직, 데모 유저 정의 등)이 바뀌면 반드시 `run_all`을 다시 돌려야 로컬 DB에 반영됩니다.

### 3. 서버 실행

```bash
uvicorn backend.main:app --reload
```

Swagger UI: http://localhost:8000/docs · 상태 확인: http://localhost:8000/health

### 4. 데모

```bash
python demo_full_flow.py       # 실제 DB 유저로 이벤트 확정 → 예측 → 정책매칭까지 엔드투엔드
python demo_ab_contrast.py     # 히스토리만 다른 두 유저가 다르게 예측되는지 오프라인으로 비교
```

## 알려진 한계

- **코호트 300개는 실거래 시뮬레이션이 아니라 합성 검색 코퍼스**입니다 — 유사 코호트 검색 + LLM 조건부 추론이라는 메커니즘을 검증하기 위한 MVP이며, 실제 KB 데이터 연동 시 데이터 소스 교체만으로 같은 파이프라인을 그대로 쓸 수 있도록 설계했습니다.
- **온통청년 오픈API**: 인증키는 정식 발급받았으나(2026.7.24 승인), API 서버 접근 이슈로 실제 호출 검증은 하지 못했습니다. 코드(`backend/integrations/`)는 공식 문서 기준으로 완성돼 있고, 데모는 수작업 큐레이션 정책 15건으로 진행합니다. 실 서비스 전환 시 이 부분만 API 응답으로 교체하면 됩니다.
- **B(상환관리 단기대응 서비스)**: 별도 인프라가 필요해 이번 제출 범위에서는 팀 결정으로 제외했습니다.

## 팀 정보

**팀명: 영크크**

| 이름 | 역할 | 담당 영역 |
|---|---|---|
| 박나림 | AI/ML 파이프라인 (Developer 2) | Transaction/State 임베딩, 유사도 검색 로직, LLM 추론·Critic·Scenario Tree·Strategy Generator·Portfolio Optimizer, Agentic 순환 로직 |
| 황재령 | 데이터/백엔드 (Developer 1) | MySQL 스키마 설계, 합성 코호트 생성 기반 마련, FastAPI 서버, A파트(정책/대출 매칭) 로직, 온통청년 API 연동 |
| 이채은 | 기획/자료 | 문제 정의, 시장 조사, 기술설명서(PPT) |

---

*제출 마감: 2026.8.3(월) 16:00 / 제출처: kb-aichallenge.com*
*본선: 2026.9.2(수) 이화여대 ECC*
