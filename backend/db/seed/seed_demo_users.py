"""
데모용 가상 유저 A/B 시드.

기획서의 핵심 데모 장면을 데이터로 재현한다:
  "같은 '취업' 이벤트를 히스토리가 다른 두 유저에게 적용하면 예측이 갈린다"

  - 유저 A (대학생 → 졸업 → [취업])      : 아직 부모동거. 코호트상 다음은 '독립' 계열이 유력
  - 유저 B (취업 → 이직 → 독립(월세) → 퇴직 → [취업]) : 이미 독립함. 코호트상 다음은 '결혼' 계열이 유력

두 유저 모두 거래내역에 '급여 개시' 패턴을 심어 두었으므로,
POST /users/{id}/detect 를 호출하면 규칙기반으로 '취업'이 후보로 잡히고,
확정(confirm)하는 순간 C엔진 재예측이 트리거된다 — 이게 데모의 시작점이다.

profile(employment_type, monthly_income_avg)은 '최근 6개월 거래내역에서 파생된 현재 상태'이고,
life_events는 '의미적으로 확정된 사건'이다. 둘은 별개 축이므로 값이 어긋나 보여도 정상이다.
"""

from datetime import date, timedelta
from typing import List

from sqlalchemy.orm import Session

from backend.db.models import LifeEvent, Transaction, User, UserLoan
from backend.repositories.event_repo import add_event

# 시드 기준일 — 재현 가능하도록 고정값 사용 (실행 시점에 따라 데이터가 흔들리면 데모가 깨진다)
TODAY = date(2026, 7, 1)


def _months_ago(n: int) -> date:
    y, m = TODAY.year, TODAY.month - n
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, 25)


# 유저 B가 독립(월세)한 시점 = 첫 월세 이체일. 확정 이벤트·대출 실행일·거래내역이 모두 이 날짜를 공유해야
# "이미 확정된 독립을 감지기가 또 물어보는" 모순이 생기지 않는다.
B_RENT_START = _months_ago(11) + timedelta(days=1)

DEMO_USERS = [
    dict(
        user_id="U_A",
        name="김하늘",
        birth_year=2001,
        employment_type="정규직",
        monthly_income_avg=2_400_000,
        income_volatility=0.080,
        marital_status="미혼",
        housing_type="부모동거",
        region_code="11",
        credit_score=752,
    ),
    dict(
        user_id="U_B",
        name="박지훈",
        birth_year=1995,
        employment_type="계약직",
        monthly_income_avg=3_100_000,
        income_volatility=0.340,          # 공백기가 있어 변동성이 크다 → B파트 버퍼 계산의 입력
        marital_status="미혼",
        housing_type="월세",
        region_code="11",
        credit_score=688,
    ),
]

DEMO_LOANS = [
    dict(
        loan_id="LN_A_1",
        user_id="U_A",
        product_id="KB-STUDENT-01",
        principal=12_000_000,
        balance=8_400_000,
        interest_rate=2.30,
        monthly_payment=180_000,
        due_day=15,
        started_at=date(2022, 3, 2),
        maturity_at=date(2029, 3, 2),
        status="정상",
    ),
    dict(
        loan_id="LN_B_1",
        user_id="U_B",
        product_id="KB-MONTHLY-01",
        principal=30_000_000,
        balance=30_000_000,
        interest_rate=2.80,
        monthly_payment=70_000,
        due_day=25,
        started_at=B_RENT_START,
        maturity_at=date(B_RENT_START.year + 2, B_RENT_START.month, B_RENT_START.day),
        status="정상",
    ),
    dict(
        loan_id="LN_B_2",
        user_id="U_B",
        product_id="KB-CREDIT-02",
        principal=15_000_000,
        balance=11_200_000,
        interest_rate=8.40,
        monthly_payment=420_000,
        due_day=10,
        started_at=date(2023, 9, 8),
        maturity_at=date(2027, 9, 8),
        status="정상",
    ),
]

# 확정 히스토리 — 데모의 '대비'를 만드는 부분
CONFIRMED_HISTORY = {
    "U_A": [("대학생", date(2020, 3, 2)), ("졸업", date(2026, 2, 20))],
    # 독립(월세) 시점을 아래 월세 거래내역 시작(11개월 전)과 맞춰 둔다.
    # 안 맞추면 감지기가 이미 확정된 독립(월세)를 새 이벤트로 또 물어본다.
    "U_B": [
        ("취업", date(2019, 9, 1)),
        ("이직", date(2022, 4, 1)),
        ("독립(월세)", B_RENT_START),
        ("퇴직", date(2025, 11, 30)),
    ],
}


def _tx(user_id: str, d: date, amount: int, counterparty: str, category: str, memo=None) -> Transaction:
    return Transaction(
        user_id=user_id, tx_date=d, amount=amount, counterparty=counterparty,
        category=category, memo=memo,
    )


def _build_transactions() -> List[Transaction]:
    txs: List[Transaction] = []

    # --- 유저 A: 12개월간 부모 용돈 → 최근 3개월 급여 개시 (취업 감지 대상) ---
    for i in range(12, 3, -1):
        txs.append(_tx("U_A", _months_ago(i), 400_000, "김성호", "기타", "부모님 용돈"))
        txs.append(_tx("U_A", _months_ago(i) + timedelta(days=2), -180_000, "KB학자금대출", "대출상환", None))
    for i in range(3, 0, -1):
        txs.append(_tx("U_A", _months_ago(i), 2_400_000, "(주)테크스타트", "급여", "급여"))
        txs.append(_tx("U_A", _months_ago(i) + timedelta(days=2), -180_000, "KB학자금대출", "대출상환", None))
        txs.append(_tx("U_A", _months_ago(i) + timedelta(days=4), -320_000, "삼성카드", "카드", None))

    # --- 유저 B: 이전 직장 급여 → 공백기 → 최근 3개월 새 급여 (재취업 감지 대상) ---
    for i in range(12, 8, -1):
        txs.append(_tx("U_B", _months_ago(i), 3_050_000, "(주)한빛물류", "급여", "급여"))
    for i in range(11, 0, -1):        # 월세는 공백기에도 계속 나감 → 연체 위험 신호
        txs.append(_tx("U_B", _months_ago(i) + timedelta(days=1), -650_000, "이정미", "월세", "월세 이체"))
    for i in range(12, 0, -1):
        txs.append(_tx("U_B", _months_ago(i) + timedelta(days=3), -420_000, "KB신용대출", "대출상환", None))
    for i in range(8, 3, -1):         # 공백기: 급여 없음, 소액 프리랜서 수입만
        txs.append(_tx("U_B", _months_ago(i) + timedelta(days=5), 620_000, "크몽", "기타", "프리랜스 정산"))
    for i in range(3, 0, -1):
        txs.append(_tx("U_B", _months_ago(i), 3_100_000, "(주)디자인랩", "급여", "급여"))

    return txs


def seed_demo_users(db: Session) -> dict:
    """데모 유저를 깨끗하게 재생성한다 (반복 실행 가능하도록 기존 데이터 삭제 후 재삽입)."""
    user_ids = [u["user_id"] for u in DEMO_USERS]

    db.query(Transaction).filter(Transaction.user_id.in_(user_ids)).delete(synchronize_session=False)
    db.query(LifeEvent).filter(LifeEvent.user_id.in_(user_ids)).delete(synchronize_session=False)
    db.query(UserLoan).filter(UserLoan.user_id.in_(user_ids)).delete(synchronize_session=False)
    db.flush()

    for row in DEMO_USERS:
        db.merge(User(**row))
    db.flush()

    for row in DEMO_LOANS:
        db.merge(UserLoan(**row))

    for tx in _build_transactions():
        db.add(tx)

    event_count = 0
    for user_id, history in CONFIRMED_HISTORY.items():
        for event_type, occurred_at in history:
            add_event(
                db,
                user_id=user_id,
                event_type=event_type,
                occurred_at=occurred_at,
                status="confirmed",
                detected_by="seed",
                confidence=1.0,
            )
            event_count += 1

    db.flush()
    return {"users": len(DEMO_USERS), "loans": len(DEMO_LOANS), "confirmed_events": event_count}
