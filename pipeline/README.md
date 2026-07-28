# C파트(pipeline) 사용 가이드

## 데모 스크립트 3개, 언제 뭘 쓰는지

| 스크립트 | 용도 | 서버/DB 필요 여부 |
|---|---|---|
| `demo_ab_contrast.py` | 로직만 빠르게 검증 (오프라인 300개 합성 데이터) | 필요 없음 |
| `demo_real_backend.py` | 재령이 실제 MySQL 코호트로 검색+예측만 확인 | 서버 필요, A파트 호출은 안 함 |
| `demo_full_flow.py` | **발표용 최종 데모** — 이벤트 등록→예측 저장→A파트 정책매칭까지 실제 API로 전체 실행 | 서버 필요 |

## 발표 당일 실행 순서

1. 터미널 1: `uvicorn backend.main:app --reload` (서버 켜두고 그대로 둠)
2. 터미널 2:
   ```
   python demo_full_flow.py U_A 취업
   ```
   (다른 시나리오 보여주려면 `U_B`, `이직` 등으로 인자만 바꿔서 재실행)

## 환경변수 (.env) 우선순위

- 최상위 `.env`: `ANTHROPIC_API_KEY`, `LLM_BACKEND` (LLM 추론용)
- `backend/.env`: `MYSQL_*`, `EMBEDDING_BACKEND`, `COHORT_TOP_K`, `COLD_START_THRESHOLD`

**중요**: `COHORT_TOP_K`, `COLD_START_THRESHOLD`, `EMBEDDING_BACKEND`는 재령이 백엔드(시드 스크립트)와 반드시 같은 값을 써야 한다. 값이 다르면:
- top_k가 다르면 A/B 두 데모에서 "근거 건수"가 다르게 나와서 비교가 안 맞음
- 콜드스타트 threshold가 다르면 어떤 유저는 여기선 low인데 백엔드 기준으론 아닌 것처럼 보일 수 있음
- 임베딩 방식이 다르면 유사도 자체가 무의미해짐 (README의 "임베딩 정합성" 항목 참고)

## 발표 전 체크리스트

- [ ] `backend/.env`의 `EMBEDDING_BACKEND`가 시드 때 쓴 것과 같은지 확인 (지금은 `offline`)
- [ ] 발표용 최종 데모는 `LLM_BACKEND=anthropic`으로 돌려서 실제 근거 문장이 나오게 할 것 (mock으로 하면 "[MOCK]" 문구가 그대로 보임)
- [ ] `python demo_full_flow.py U_A 취업` 최소 1회 리허설 — API 응답 속도 확인 (발표 중 몇 초 걸릴 수 있음을 감안한 멘트 준비)
- [ ] 테스트하면서 쌓인 불필요한 이벤트/예측 데이터는 발표 전에 정리하거나, 혹은 "이것도 실제 누적 데이터"라는 걸 자연스럽게 설명에 포함
