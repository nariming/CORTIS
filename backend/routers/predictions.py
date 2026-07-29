"""
C엔진 예측 결과 저장/조회.

C파트가 LLM 추론을 끝낸 뒤 여기에 POST 하면, 근거로 쓴 코호트 top-k까지 통째로 남는다.
"이 숫자 어디서 나왔냐"는 질문에 DB 레코드로 답할 수 있게 하는 게 목적이고,
trigger_event_id 로 "어떤 이벤트가 이 재예측을 유발했는지" 인과가 추적된다.
"""

from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Prediction
from backend.routers.users import get_user_or_404
from backend.schemas import PredictionIn, PredictionOut

router = APIRouter(tags=["predictions (C엔진 결과)"])


@router.post("/users/{user_id}/predictions", response_model=PredictionOut, summary="예측 결과 저장")
def save_prediction(user_id: str, payload: PredictionIn, db: Session = Depends(get_db)):
    get_user_or_404(db, user_id)
    record = Prediction(
        user_id=user_id,
        trigger_event_id=payload.trigger_event_id,
        input_history_json=payload.input_history,
        predictions_json=payload.predictions,
        confidence_level=payload.confidence_level,
        confidence_note=payload.confidence_note,
        matched_cohorts_json=payload.matched_cohorts,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get(
    "/users/{user_id}/predictions",
    response_model=List[PredictionOut],
    summary="예측 이력 (최신순) — 재예측이 쌓이는 게 agentic 순환의 증거",
)
def list_predictions(
    user_id: str, limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)
):
    get_user_or_404(db, user_id)
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc(), Prediction.prediction_id.desc())
        .limit(limit)
        .all()
    )
