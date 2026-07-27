# CORTIS
Continuous financing Opportunity Reassessment & Tailored Intelligent Service

변동 소득 청년 대출자를 위한 AI 자율 상환 페이스메이커 & 생애주기 기반 부채 포트폴리오 재설계 서비스

**"정해진 날짜가 아니라, 내 상황에 맞춰 갚는다."**

---

## 프로젝트 소개

CORTIS는 KB의 대출·계좌 데이터를 기반으로 청년 대출자의 상환 시점과 부채 포트폴리오를 지속적으로 관찰·재설계하는 AI 에이전트 서비스입니다.

고정된 상환 스케줄이 플랫폼노동·프리랜서 등 변동 소득 청년의 현실과 어긋나면서 발생하는 연체를, 사용자가 요청하지 않아도 거래내역 변화를 스스로 감지하고 최적의 상환·정책 조합을 다시 계산해 제안하는 방식으로 방어합니다. 목표는 단순 알림형 뱅킹 앱이 아니라, KB 내부에서 PB 고객에게만 제공되던 수준의 자문을 청년 대출자에게도 확장하는 것입니다.

## 프로젝트 배경 및 문제 정의

**왜 청년 대출자는 반복적으로 연체에 빠지는가?**

- **소득과 상환 구조의 미스매치**: 5대 은행 20대 이하 가계대출 연체율이 전 연령대 중 최고(0.44%, 2026.3말 기준)이며, 인터넷은행에서는 격차가 더 두드러집니다(토스뱅크 20대 연체율 2.5%). 원인은 상환 의지가 아니라, 플랫폼노동·프리랜서·단기계약 등으로 소득 발생 시기가 일정하지 않은데 상환일은 고정돼 있다는 구조적 미스매치입니다.
- **기존 서비스의 한계**: KB를 포함한 기존 뱅킹 앱은 "연체 임박"을 사후에 알려주거나, 신용점수·자산유형 같은 하나의 정적 신호에만 반응합니다. 정작 "이번 달 무엇을 언제 얼마나 갚아야 하는가"라는 실행 판단까지는 내려주지 않습니다.
- **제도적 공백**: 정책금융은 오히려 신용도가 높은 청년이 더 잘 활용하는 역설 구조이며, 상환 행동을 신용점수 개선으로 연결하는 메커니즘도 없습니다.

## CORTIS의 해결책

CORTIS는 시장이나 상품을 추천하는 것이 아니라, **사용자의 거래내역 변화를 스스로 관찰해 상환 판단을 대신 내려주는 에이전트**를 지향합니다. 하나의 정적 신호에 반응하는 기존 서비스들과 달리, 여러 신호를 종합해 생애주기 변화를 예측하고 그 결과를 정책 매칭·상환 계획 재설계로 즉시 연결합니다.

## 주요 기능 (Key Features)

CORTIS는 3개 모듈이 순환 구조로 연결되며, **C가 메인 AI 추론 엔진**, A·B는 C의 산출물을 받아 실행하는 서포팅 레이어입니다.

```
   C. 생애주기 이벤트 예측 (메인 엔진)
   감지(규칙기반) → 유사 코호트 검색 → LLM 추론
              │ 이벤트 확정 시 트리거
              ▼
   A. 정책 매칭 레이어 (규칙기반 위주)
   C의 산출물을 받아 정책 자격 재판단·매칭
              │
              ▼
   B. 상환 관리 (후순위)
   MAD 기반 유동성 버퍼 + LLM 넛지 문구
```

### 1. C. 생애주기 이벤트 예측 (메인 AI 추론 엔진)

- **감지**: 규칙 기반 — 신규 이체처 발생 / 이체 중단 등 패턴 감지
- **다음 이벤트 예측**: 합성 코호트 약 300개 이벤트 시퀀스를 사전 임베딩해두고, 사용자의 확정된 이벤트 히스토리로 유사 코호트를 검색 → 그 결과를 근거로 LLM이 다음 이벤트를 예측
- 같은 이벤트라도 개인 히스토리가 다르면 예측 결과가 달라지는 **조건부 추론**이 핵심 차별점

### 2. A. 정책 매칭 레이어

- C가 감지한 생애주기 이벤트를 입력으로 받아, 신규 자격 발생/상실 정책을 규칙 기반으로 재탐색
- RAG 기반 정책 DB 매칭은 보조 기능으로 축소

### 3. B. 상환 관리

- 후순위 기능, 볼륨 축소
- MAD(중위절대편차) 기반 유동성 버퍼 계산 + LLM 기반 넛지 문구 생성

> 아래 기능은 목표 제품 기준이며, 실제 구현 범위는 개발 진행 상황에 맞춰 조정됩니다.

## 데이터 전략 및 기술 아키텍처

**데이터 파이프라인**

- KB 실제 API 연동 대신 **합성/시뮬레이션 데이터**를 사용해 대출·계좌·거래내역을 재현
- PPT/코드에 실제 KB API 연동 시 확장 가능성을 명시

**Tech Stack**

| 분류 | 기술 |
| --- | --- |
| DB (Source of Truth) | MySQL — 유저 프로필, 대출 DB, 정책 DB, 이벤트 로그, 임베딩 벡터 |
| 유사도 검색 | numpy 코사인 유사도 (경량, 데이터 규모상 전용 벡터DB 불필요) |
| 오케스트레이션 | LangChain |
| 추론 | LLM (provider-agnostic) |
| 백엔드 | FastAPI |
| 통계 | MAD 기반 유동성 버퍼 산출 |

## 프로젝트 구조

```
cortis/
├── backend/                  # FastAPI 서버 + 데이터 계층 (개발자1)
│   ├── main.py                   # 앱 엔트리포인트
│   ├── config.py                 # .env 기반 설정
│   ├── schemas.py                # API 요청/응답 스키마
│   ├── embedding_compat.py       # 임베딩 호환 레이어 (C파트 구현 우선 사용)
│   ├── db/
│   │   ├── schema.sql                # MySQL DDL (원본)
│   │   ├── models.py                 # SQLAlchemy ORM
│   │   ├── database.py               # 엔진/세션
│   │   └── seed/                     # 정책·대출상품·데모유저·코호트 적재
│   ├── repositories/             # cohort_repo(C엔진 연동), event_repo
│   ├── matcher_a/                # A: 규칙기반 이벤트 감지 + 정책 자격 매칭
│   ├── routers/                  # API 엔드포인트
│   └── tests/test_smoke.py       # MySQL 없이 SQLite로 도는 배선 검증
│
├── pipeline/                 # C: 임베딩·유사 코호트 검색·LLM 추론 (개발자2)
├── data/                     # 합성 코호트 300개 시퀀스
└── doc/                      # 기획서(PPT), 팀 문서
```

## 실행 및 데모

### 1. 준비

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

copy backend\.env.example backend\.env    # macOS/Linux: cp
# backend/.env 를 열어 MYSQL_PASSWORD 만 본인 로컬 값으로 채운다
```

### 2. DB 생성 + 시드 (한 번만)

```bash
python -m backend.db.seed.run_all
```

`cortis` 데이터베이스 생성 → 테이블 9개 → 정책 15건·대출상품 6건·데모 유저 A/B·코호트 300건(임베딩 포함)까지 한 번에 적재된다.

### 3. 서버 실행

```bash
uvicorn backend.main:app --reload
```

Swagger UI: http://localhost:8000/docs · 상태 확인: http://localhost:8000/health

> MySQL 설치 전이라도 `python -m backend.tests.test_smoke` 로 감지→확정→정책매칭 배선을 SQLite에서 검증할 수 있다.
> 배포된 데모 url 필요해지면 rds나 planetScale 같은 걸 붙이면 됨 (스키마도 지금 그대로)

## 모듈 간 연동 규약

C엔진(개발자2)이 백엔드(개발자1)와 주고받는 지점은 아래 네 개다.

| 방향 | 엔드포인트 | 용도 |
| --- | --- | --- |
| 읽기 | `GET /cohorts` | 코호트 시퀀스 + 임베딩 벡터. 응답을 그대로 `CohortIndex.load_from_mysql_rows()` 에 전달 |
| 읽기 | `GET /users/{id}/history` | 확정 이벤트 히스토리 + 콜드스타트 여부 + LLM 프롬프트용 `user_context` |
| 쓰기 | `POST /users/{id}/predictions` | 추론 결과 저장 (근거 코호트 top-k까지 함께 보관) |
| 쓰기 | `POST /users/{id}/policy-match` | 예측 이벤트를 A파트에 넘겨 정책 자격 재판단 |

같은 프로세스에서 돌릴 때는 HTTP 없이 `backend.repositories.cohort_repo.load_cohort_rows(db)` 를 직접 호출해도 형태가 동일하다.

**Agentic 순환 고리** — 이벤트 확정이 트리거다.

```
POST /users/{id}/detect          거래내역 스캔 → 이벤트 후보(detected)
POST /events/{event_id}/confirm  사용자 확인 → confirmed 승격
GET  /users/{id}/history         갱신된 히스토리
   → C엔진 재검색·재예측
POST /users/{id}/predictions     예측 결과 저장
POST /users/{id}/policy-match    A파트 정책 재탐색
```

**임베딩 정합성**: 코호트 적재 때 쓴 임베딩과 검색 때 쓰는 임베딩이 같아야 유사도가 의미를 갖는다. `cohort_sequences.embedding_model` 에 모델명을 함께 저장하고, 조회 시 같은 모델 벡터만 로드한다. `EMBEDDING_BACKEND` 를 바꾸면 `python -m backend.db.seed.run_all --no-drop` 로 재적재할 것.

## 기대 효과 및 학술적 의의

- **진단에서 실행 판단으로의 전환**: 기존 신용관리 리포트·DSR계산기가 "지금 부채 수준이 적정한가"를 진단하는 정적 스냅샷이라면, CORTIS는 매일 갱신되는 거래내역을 근거로 "이번 달 무엇을 언제 얼마나 갚을지" 실행 단위까지 판단합니다.
- **사회적 가치**: 청년 대출 연체를 사후 대응이 아닌 조기 관리 대상으로 전환하고, KB 내부에서만 제공되던 PB급 자문을 청년 대출자에게 확장합니다.

## 팀 정보

**팀명: 영크크**

| 이름 | 역할 | 담당 영역 |
| --- | --- | --- |
| 황재령 | 데이터/백엔드 | MySQL 스키마 설계, 합성 코호트 시퀀스 생성, FastAPI 서버, A 파트(정책 매칭) 로직 |
| 박나림 | AI/ML 파이프라인 | 임베딩 파이프라인, numpy 유사도 검색 로직, LLM 추론 프롬프트 설계, Agentic 순환 로직, 콜드스타트 처리, 데모 시나리오 데이터 준비 |
| 이채은 | 기획/자료 | 문제 정의, 시장 조사, 기술설명서(PPT) |

---

*제출 마감: 2026.8.3(월) 16:00 / 제출처: kb-aichallenge.com*
*최종수정일: 2026.07.26*
