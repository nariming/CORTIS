"""
확정 이벤트 히스토리 레포지토리.

C엔진의 입력은 결국 "이 유저의 확정된 이벤트 순서 리스트"이고, 그 원천이 life_events 테이블이다.
offset_month / prev_gap_month 계산도 여기서 책임진다 (기획 캡처의 저장 형태 그대로).
"""

from datetime import date
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.db.models import LifeEvent


def confirmed_history(db: Session, user_id: str) -> List[str]:
    """C엔진에 넘길 확정 히스토리 (이벤트명만, 시간순)."""
    rows = (
        db.query(LifeEvent)
        .filter(LifeEvent.user_id == user_id, LifeEvent.status == "confirmed")
        .order_by(LifeEvent.occurred_at, LifeEvent.event_id)
        .all()
    )
    return [r.event_type for r in rows]


def event_timeline(db: Session, user_id: str, status: Optional[str] = None) -> List[LifeEvent]:
    query = db.query(LifeEvent).filter(LifeEvent.user_id == user_id)
    if status:
        query = query.filter(LifeEvent.status == status)
    return query.order_by(LifeEvent.occurred_at, LifeEvent.event_id).all()


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def add_event(
    db: Session,
    user_id: str,
    event_type: str,
    occurred_at: date,
    status: str = "detected",
    detected_by: str = "rule",
    confidence: Optional[float] = None,
    evidence: Optional[dict] = None,
) -> LifeEvent:
    """이벤트 1건 추가. offset_month / prev_gap_month 를 자동 계산해서 채운다."""
    existing = event_timeline(db, user_id, status="confirmed")

    if existing:
        first_date = existing[0].occurred_at
        offset_month = _months_between(first_date, occurred_at)
        prev = max(existing, key=lambda e: e.occurred_at)
        prev_gap = _months_between(prev.occurred_at, occurred_at)
    else:
        offset_month = 0
        prev_gap = None

    event = LifeEvent(
        user_id=user_id,
        event_type=event_type,
        occurred_at=occurred_at,
        offset_month=max(offset_month, 0),
        prev_gap_month=prev_gap if prev_gap is None else max(prev_gap, 0),
        status=status,
        detected_by=detected_by,
        confidence=confidence,
        evidence=evidence,
    )
    db.add(event)
    db.flush()
    return event


def confirm_event(db: Session, event_id: int) -> Optional[LifeEvent]:
    """사용자 확인을 거쳐 detected → confirmed 로 승격.

    이 시점이 agentic 순환의 트리거다 (C엔진 재검색·재예측 → A/B 재호출).
    """
    event = db.get(LifeEvent, event_id)
    if event is None:
        return None
    event.status = "confirmed"
    event.detected_by = "user"
    db.flush()
    return event


def reject_event(db: Session, event_id: int) -> Optional[LifeEvent]:
    event = db.get(LifeEvent, event_id)
    if event is None:
        return None
    event.status = "rejected"
    db.flush()
    return event
