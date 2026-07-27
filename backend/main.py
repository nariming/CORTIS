"""
CORTIS 백엔드 엔트리포인트.

    uvicorn backend.main:app --reload
    Swagger UI: http://localhost:8000/docs

담당: 개발자1 (데이터/백엔드) — MySQL 스키마, 시드, API, A파트 정책 매칭
연동: 개발자2 (C엔진) 는 /cohorts, /users/{id}/history 를 읽고 /users/{id}/predictions 에 결과를 쓴다.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend import config
from backend.db.database import engine
from backend.routers import cohorts, events, policies, predictions, users

app = FastAPI(
    title="CORTIS API",
    description=(
        "변동 소득 청년 대출자를 위한 생애주기 이벤트 예측 기반 금융 에이전트.\n\n"
        "- **events**: 거래내역 규칙기반 감지 → 사용자 확정 (C엔진 입력 생성)\n"
        "- **cohorts**: 합성 코호트 300건 + 임베딩 벡터 (C엔진 검색 인덱스 소스)\n"
        "- **predictions**: C엔진 추론 결과 저장 (근거 코호트까지 함께 보관)\n"
        "- **policies**: A파트 — 예측 이벤트 기반 정책 자격 재판단"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 데모 단계라 전체 허용. 배포 시 프런트 도메인으로 좁힐 것
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(events.router)
app.include_router(cohorts.router)
app.include_router(policies.router)
app.include_router(predictions.router)


@app.get("/health", tags=["system"], summary="DB 연결 상태 확인")
def health():
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT VERSION()")).scalar()
            cohort_count = conn.execute(text("SELECT COUNT(*) FROM cohort_sequences")).scalar()
        return {
            "status": "ok",
            "mysql_version": version,
            "database": config.MYSQL_DB,
            "cohort_rows": cohort_count,
            "embedding_backend": config.EMBEDDING_BACKEND,
        }
    except Exception as exc:  # DB가 안 떠 있을 때 원인을 바로 보여준다
        return {"status": "error", "detail": str(exc)}
