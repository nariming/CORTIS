"""
A파트 API — KB 대출상품 자격 매칭 (policies.py의 LoanProduct 버전).

POST /users/{id}/loan-match 가 핵심: C엔진이 예측한 이벤트를 받아 자격이 되는
KB 대출상품을 돌려준다. min_rate/max_rate를 그대로 포함하므로, 나림이가 유저의
기존 대출(user_loans.interest_rate)과 비교해 "갈아타면 절감되는 금액"을 계산하는
입력으로 바로 쓸 수 있다.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import LoanProduct
from backend.matcher_a import policy_matcher
from backend.routers.users import get_user_or_404
from backend.schemas import EventLoanGroupOut, LoanMatchIn, LoanProductMatchOut, LoanProductOut

router = APIRouter(tags=["loans (A파트)"])


@router.get("/loan-products", response_model=List[LoanProductOut], summary="KB 대출상품 카탈로그 전체 조회")
def list_loan_products(
    product_type: Optional[str] = None,
    trigger_event: Optional[str] = Query(None, description="이 이벤트를 트리거로 갖는 상품만 필터"),
    db: Session = Depends(get_db),
):
    query = db.query(LoanProduct)
    if product_type:
        query = query.filter(LoanProduct.product_type == product_type)
    products = query.order_by(LoanProduct.min_rate, LoanProduct.product_id).all()
    if trigger_event:
        products = [p for p in products if trigger_event in (p.trigger_events or [])]
    return products


@router.get(
    "/users/{user_id}/loan-products",
    response_model=List[LoanProductMatchOut],
    summary="현재 자격이 되는 KB 대출상품 (이벤트 무관, 기본 화면용)",
)
def current_loan_products(user_id: str, product_type: Optional[str] = None, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    return [m.to_dict() for m in policy_matcher.current_loan_eligibility(db, user, product_type)]


@router.post(
    "/users/{user_id}/loan-match",
    response_model=List[EventLoanGroupOut],
    summary="C엔진 예측 이벤트를 받아 KB 대출상품 자격 재판단",
)
def match_loan_products(user_id: str, payload: LoanMatchIn, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    return policy_matcher.match_for_loan_events(
        db,
        user,
        payload.event_types,
        include_ineligible=payload.include_ineligible,
        prospective=payload.prospective,
    )
