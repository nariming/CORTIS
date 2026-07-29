"""
A파트 API — 정책 자격 매칭.

핵심 엔드포인트는 POST /users/{id}/policy-match 로, C엔진이 예측한 이벤트 목록을 받아
"그 이벤트 때문에 새로 자격이 생기는 정책"을 결정론적으로 돌려준다.
데모의 대비 장면(A는 전월세자금대출, B는 신혼부부 정책)이 바로 이 응답으로 그려진다.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Policy, PolicyMatchResult
from backend.matcher_a import policy_matcher
from backend.routers.users import get_user_or_404
from backend.schemas import (
    EventPolicyGroupOut,
    PolicyMatchIn,
    PolicyMatchOut,
    PolicyOut,
)

router = APIRouter(tags=["policies (A파트)"])


@router.get("/policies", response_model=List[PolicyOut], summary="정책 DB 전체 조회")
def list_policies(
    category: Optional[str] = None,
    trigger_event: Optional[str] = Query(None, description="이 이벤트를 트리거로 갖는 정책만 필터"),
    db: Session = Depends(get_db),
):
    query = db.query(Policy)
    if category:
        query = query.filter(Policy.category == category)
    policies = query.order_by(Policy.priority, Policy.policy_id).all()
    if trigger_event:
        policies = [p for p in policies if trigger_event in (p.trigger_events or [])]
    return policies


@router.get(
    "/users/{user_id}/policies",
    response_model=List[PolicyMatchOut],
    summary="현재 자격이 있는 정책 (이벤트 무관, 기본 화면용)",
)
def current_policies(user_id: str, category: Optional[str] = None, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    return [m.to_dict() for m in policy_matcher.current_eligibility(db, user, category)]


@router.post(
    "/users/{user_id}/policy-match",
    response_model=List[EventPolicyGroupOut],
    summary="C엔진 예측 이벤트를 받아 정책 자격 재판단",
)
def match_policies(user_id: str, payload: PolicyMatchIn, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    groups = policy_matcher.match_for_events(
        db,
        user,
        payload.event_types,
        include_ineligible=payload.include_ineligible,
        prospective=payload.prospective,
    )

    if payload.persist:
        for group in groups:
            for policy in group["policies"]:
                db.add(
                    PolicyMatchResult(
                        user_id=user_id,
                        prediction_id=payload.prediction_id,
                        policy_id=policy["policy_id"],
                        basis_event=group["basis_event"],
                        status=policy["status"],
                        reason=policy["reason"][:500],
                    )
                )
        db.commit()

    return groups
