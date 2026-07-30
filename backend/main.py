"""
CORTIS 백엔드 엔트리포인트.

    uvicorn backend.main:app --reload
    Swagger UI: http://localhost:8000/docs

담당: 개발자1 (데이터/백엔드) — MySQL 스키마, 시드, API, A파트 정책 매칭
연동: 개발자2 (C엔진) 는 /cohorts, /users/{id}/history 를 읽고 /users/{id}/predictions 에 결과를 쓴다.
"""

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend import config
from backend.db.database import engine
from backend.routers import cohorts, events, loans, policies, predictions, users
from backend.security import verify_api_key

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="CORTIS API",
    description=(
        "변동 소득 청년 대출자를 위한 생애주기 이벤트 예측 기반 금융 에이전트.\n\n"
        "- **events**: 거래내역 규칙기반 감지 → 사용자 확정 (C엔진 입력 생성)\n"
        "- **cohorts**: 합성 코호트 300건 + 임베딩 벡터 (C엔진 검색 인덱스 소스)\n"
        "- **predictions**: C엔진 추론 결과 저장 (근거 코호트까지 함께 보관)\n"
        "- **policies**: A파트 — 예측 이벤트 기반 정책 자격 재판단\n\n"
        "인증: 모든 엔드포인트는 `X-API-Key` 헤더가 필요합니다 (backend/security.py 참고)."
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

if not config.API_KEY:
    logger.warning(
        "[보안 경고] API_KEY가 .env에 설정되지 않아 API 키 인증이 꺼진 상태입니다. "
        "로컬 개발용으로만 이렇게 두고, 데모/발표 전 .env에 API_KEY를 설정하세요."
    )

_api_key_dep = [Depends(verify_api_key)]

app.include_router(users.router, dependencies=_api_key_dep)
app.include_router(events.router, dependencies=_api_key_dep)
app.include_router(cohorts.router, dependencies=_api_key_dep)
app.include_router(policies.router, dependencies=_api_key_dep)
app.include_router(loans.router, dependencies=_api_key_dep)
app.include_router(predictions.router, dependencies=_api_key_dep)


@app.get("/health", tags=["system"], summary="DB 연결 상태 확인 (인증 불필요)")
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