"""
생애주기 이벤트 API — C엔진의 입력을 만들고 관리하는 부분.

흐름:
    POST /users/{id}/detect          거래내역 스캔 → 후보를 detected 상태로 저장
    POST /events/{event_id}/confirm  사용자 확인 → confirmed 승격  ← agentic 순환의 트리거
    GET  /users/{id}/history         C엔진이 예측 직전에 읽어갈 확정 히스토리 + 유저 컨텍스트
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import config
from backend.db.database import get_db
from backend.db.models import Transaction
from backend.matcher_a.detector import detect_all
from backend.repositories import event_repo
from backend.routers.users import get_user_or_404
from backend.schemas import (
    DetectResponse,
    EventCandidateOut,
    EventCreateIn,
    HistoryOut,
    LifeEventOut,
)
from pipeline.state_builder import build_user_state

router = APIRouter(tags=["events"])


def build_user_context(user, db: Session) -> str:
    """LLM 프롬프트의 '유저 현재 상황' 자리에 그대로 들어갈 한 문단.

    C파트 reasoning.build_user_prompt(user_context=...) 에 넘길 값을 백엔드가 만들어 준다.
    (프롬프트에 넣을 사실은 DB가 원천이어야 하므로 여기서 조립하는 게 맞다)

    state_builder.build_user_state()가 계산한 값(DSR, 거래 트렌드)을 그대로 이어붙인다 —
    프로필/대출 요약과 거래 트렌드가 서로 다른 두 함수에서 각각 조립되면 값이 어긋날 수 있어,
    State Builder 하나를 두 곳(State Embedding, 이 함수)이 공유하도록 통일했다.
    """
    state = build_user_state(user, txs=list(user.transactions), as_of=None)

    loans = user.loans
    loan_desc = (
        ", ".join(f"{ln.product_id} 잔액 {ln.balance:,}원(월 {ln.monthly_payment:,}원)" for ln in loans)
        or "보유 대출 없음"
    )
    dsr_desc = f"DSR {state.dsr_pct:.1f}%" if state.dsr_pct is not None else "대출 없음"

    return (
        f"{user.age}세 {user.employment_type}, 월평균 소득 {user.monthly_income_avg:,}원 "
        f"(소득 변동성 {float(user.income_volatility):.2f}), 거주형태 {user.housing_type}, "
        f"혼인상태 {user.marital_status}. 보유 대출: {loan_desc} ({dsr_desc}). "
        f"최근 거래 추세: {state.tx_features.to_sentence()}"
    )


@router.get("/users/{user_id}/events", response_model=List[LifeEventOut], summary="이벤트 타임라인")
def list_events(user_id: str, status: str = None, db: Session = Depends(get_db)):
    get_user_or_404(db, user_id)
    return event_repo.event_timeline(db, user_id, status=status)


@router.get("/users/{user_id}/history", response_model=HistoryOut, summary="C엔진 입력: 확정 히스토리")
def get_history(user_id: str, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    history = event_repo.confirmed_history(db, user_id)

    is_cold = len(history) < config.COLD_START_THRESHOLD
    note = (
        f"확정 이벤트가 {len(history)}건뿐이라 검색 근거가 빈약합니다. "
        "예측 결과에 '초기 데이터 부족' 신뢰도 표시가 필요합니다."
        if is_cold
        else None
    )
    return HistoryOut(
        user_id=user_id,
        confirmed_history=history,
        is_cold_start=is_cold,
        cold_start_note=note,
        user_context=build_user_context(user, db),
    )


@router.post("/users/{user_id}/detect", response_model=DetectResponse, summary="거래내역 규칙기반 이벤트 감지")
def detect_events(user_id: str, persist: bool = True, db: Session = Depends(get_db)):
    """거래내역을 스캔해 이벤트 후보를 뽑는다.

    persist=True 면 후보를 life_events 에 status='detected' 로 저장하고 event_id 를 함께 돌려준다.
    프런트는 그 event_id로 확인 질문("취업하셨나요?")의 예/아니오를 confirm/reject 로 보내면 된다.
    """
    get_user_or_404(db, user_id)

    txs = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.tx_date)
        .all()
    )
    confirmed_rows = event_repo.event_timeline(db, user_id, status="confirmed")
    candidates = detect_all(txs, [(e.event_type, e.occurred_at) for e in confirmed_rows])

    out: List[EventCandidateOut] = []
    for cand in candidates:
        event_id = None
        if persist:
            saved = event_repo.add_event(
                db,
                user_id=user_id,
                event_type=cand.event_type,
                occurred_at=cand.occurred_at,
                status="detected",
                detected_by="rule",
                confidence=cand.confidence,
                evidence={"tx_ids": cand.evidence_tx_ids, "reason": cand.reason},
            )
            event_id = saved.event_id
        out.append(EventCandidateOut(**cand.to_dict(), event_id=event_id))

    if persist:
        db.commit()

    return DetectResponse(
        user_id=user_id,
        scanned_transactions=len(txs),
        confirmed_history=[e.event_type for e in confirmed_rows],
        candidates=out,
    )


@router.post("/events/{event_id}/confirm", response_model=LifeEventOut, summary="이벤트 확정 (재예측 트리거)")
def confirm_event(event_id: int, db: Session = Depends(get_db)):
    """detected → confirmed 승격.

    여기서 히스토리가 바뀌므로, 호출한 쪽은 이어서
    GET /users/{id}/history → C엔진 재예측 → POST /users/{id}/predictions → POST /users/{id}/policy-match
    순서로 순환 고리를 돌리면 된다.
    """
    event = event_repo.confirm_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"이벤트를 찾을 수 없습니다: {event_id}")
    db.commit()
    db.refresh(event)
    return event


@router.post("/events/{event_id}/reject", response_model=LifeEventOut, summary="이벤트 기각 (오탐 처리)")
def reject_event(event_id: int, db: Session = Depends(get_db)):
    event = event_repo.reject_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"이벤트를 찾을 수 없습니다: {event_id}")
    db.commit()
    db.refresh(event)
    return event


@router.post("/users/{user_id}/events", response_model=LifeEventOut, summary="이벤트 수동 추가 (데모/테스트용)")
def create_event(user_id: str, payload: EventCreateIn, db: Session = Depends(get_db)):
    get_user_or_404(db, user_id)
    event = event_repo.add_event(
        db,
        user_id=user_id,
        event_type=payload.event_type,
        occurred_at=payload.occurred_at,
        status=payload.status,
        detected_by="user",
        confidence=payload.confidence,
    )
    db.commit()
    db.refresh(event)
    return event