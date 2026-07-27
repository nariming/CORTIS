"""
코호트 API — 개발자2(C엔진)가 쓰는 연동 지점.

GET /cohorts 응답을 그대로 CohortIndex.load_from_mysql_rows() 에 넣으면 검색 인덱스가 완성된다.
같은 프로세스 안에서 돌 때는 HTTP를 거치지 말고
backend.repositories.cohort_repo.load_cohort_rows(db) 를 직접 호출해도 된다 (동일 형태).
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend import config
from backend.db.database import get_db
from backend.repositories import cohort_repo
from backend.schemas import CohortRowOut, CohortStatsOut

router = APIRouter(prefix="/cohorts", tags=["cohorts (C엔진 연동)"])


@router.get("", response_model=List[CohortRowOut], summary="코호트 시퀀스 + 임베딩 벡터 전체 로드")
def list_cohorts(
    embedding_model: Optional[str] = Query(
        None, description="지정하지 않으면 현재 설정된 임베딩 모델의 벡터만 반환"
    ),
    db: Session = Depends(get_db),
):
    return cohort_repo.load_cohort_rows(db, embedding_model=embedding_model)


@router.get("/stats", response_model=CohortStatsOut, summary="코호트 통계 (population prior)")
def cohort_stats(db: Session = Depends(get_db)):
    """전체 next_event 분포.

    top-k 5건 집계만 보면 표본이 작아 흔들리므로, "전체 300건 중 결혼은 원래 몇 %인가"를
    같이 보여줘야 '이 유저에게 유독 독립이 몰렸다'는 주장이 성립한다.
    """
    return CohortStatsOut(
        total=cohort_repo.count_cohorts(db),
        embedding_model=f"offline-hash-{config.EMBEDDING_DIM}"
        if config.EMBEDDING_BACKEND == "offline"
        else "text-embedding-3-small",
        next_event_distribution=cohort_repo.next_event_distribution(db),
    )
