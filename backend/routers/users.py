"""유저 프로필 / 대출 / 거래내역 조회."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Transaction, User
from backend.repositories.event_repo import confirmed_history
from backend.schemas import TransactionOut, UserDetailOut, UserOut

router = APIRouter(prefix="/users", tags=["users"])


def get_user_or_404(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"유저를 찾을 수 없습니다: {user_id}")
    return user


@router.get("", response_model=List[UserOut], summary="전체 유저 목록 (데모용)")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.user_id).all()


@router.get("/{user_id}", response_model=UserDetailOut, summary="유저 상세 (프로필+대출+확정 히스토리)")
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = get_user_or_404(db, user_id)
    detail = UserDetailOut.model_validate(user)
    detail.age = user.age
    detail.annual_income = user.annual_income
    detail.confirmed_history = confirmed_history(db, user_id)
    return detail


@router.get("/{user_id}/transactions", response_model=List[TransactionOut], summary="거래내역 조회")
def get_transactions(
    user_id: str,
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    get_user_or_404(db, user_id)
    return (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.tx_date.desc(), Transaction.tx_id.desc())
        .limit(limit)
        .all()
    )
