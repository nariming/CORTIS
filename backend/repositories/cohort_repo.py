"""
코호트 조회 레포지토리 — 개발자1(MySQL) ↔ 개발자2(C엔진) 의 계약 지점.

C파트 pipeline/similarity.py 의 CohortIndex.load_from_mysql_rows() 가 기대하는 형태가
  [{"history": [...], "embedding_vector": [...], "next_event": "..."}]
이므로, 이 함수 반환값을 그대로 넘기면 배선이 끝난다.

    from backend.repositories.cohort_repo import load_cohort_rows
    index.load_from_mysql_rows(load_cohort_rows(db))
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from backend import config
from backend.db.models import CohortSequence


def load_cohort_rows(db: Session, embedding_model: Optional[str] = None) -> List[dict]:
    """CohortIndex.load_from_mysql_rows() 에 그대로 넘길 수 있는 dict 리스트.

    embedding_model 을 지정하면 그 모델로 만든 벡터만 골라온다.
    (offline 해시 벡터와 OpenAI 벡터가 섞이면 코사인 유사도가 무의미해지므로 기본값으로 현재 설정 모델만 로드)
    """
    query = db.query(CohortSequence)
    if embedding_model is None and config.EMBEDDING_BACKEND == "offline":
        embedding_model = f"offline-hash-{config.EMBEDDING_DIM}"
    if embedding_model:
        query = query.filter(CohortSequence.embedding_model == embedding_model)

    return [
        {
            "cohort_id": c.cohort_id,
            "history": c.history_json,
            "event_history_text": c.event_history_text,
            "next_event": c.next_event,
            "embedding_vector": c.embedding_vector,
        }
        for c in query.order_by(CohortSequence.cohort_id).all()
    ]


def count_cohorts(db: Session) -> int:
    return db.query(CohortSequence).count()


def next_event_distribution(db: Session) -> dict:
    """전체 코호트의 next_event 분포 (population prior).

    검색 top-k 집계만으로는 표본이 5건뿐이라 흔들릴 수 있어서,
    C엔진이 "전체 분포 대비 이 유저의 top-k가 얼마나 치우쳐 있는지"를 볼 때 쓰라고 함께 제공한다.
    """
    rows = db.query(CohortSequence.next_event).all()
    dist: dict = {}
    for (event,) in rows:
        dist[event] = dist.get(event, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))
