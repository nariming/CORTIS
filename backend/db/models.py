"""
SQLAlchemy ORM 모델. schema.sql 의 DDL과 1:1로 대응한다.

DDL 자체는 schema.sql 이 원본이고(주석/COMMENT가 기술설명서에 그대로 쓰임),
이 파일은 파이썬에서 읽고 쓰기 위한 매핑이다. 컬럼을 바꿀 땐 두 파일을 같이 고칠 것.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base

# MySQL은 BIGINT AUTO_INCREMENT를 지원하지만 SQLite는 INTEGER PK만 자동증가시킨다.
# 팀원 PC에 MySQL이 없어도 backend/tests/test_smoke.py 를 SQLite로 돌릴 수 있도록 variant를 건다.
# (운영 DDL은 schema.sql 의 BIGINT 그대로)
AutoBigInt = BigInteger().with_variant(Integer, "sqlite")

EMPLOYMENT_TYPES = ("정규직", "계약직", "프리랜서", "플랫폼노동", "자영업", "무직", "학생")
HOUSING_TYPES = ("부모동거", "월세", "전세", "자가", "기숙사")
MARITAL_TYPES = ("미혼", "기혼")


class User(Base):
    __tablename__ = "users"

    user_id = Column(String(32), primary_key=True)
    name = Column(String(50), nullable=False)
    birth_year = Column(SmallInteger, nullable=False)
    employment_type = Column(Enum(*EMPLOYMENT_TYPES), nullable=False)
    monthly_income_avg = Column(Integer, nullable=False, default=0)
    income_volatility = Column(Numeric(5, 3), nullable=False, default=0)
    marital_status = Column(Enum(*MARITAL_TYPES), nullable=False, default="미혼")
    housing_type = Column(Enum(*HOUSING_TYPES), nullable=False, default="부모동거")
    region_code = Column(String(10), nullable=False, default="11")
    credit_score = Column(SmallInteger, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    loans = relationship("UserLoan", back_populates="user", cascade="all, delete-orphan")
    events = relationship("LifeEvent", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

    @property
    def age(self) -> int:
        """만 나이 근사 (정책 자격 판정용). 생년만 있으므로 연도 차이로 계산."""
        return datetime.now().year - self.birth_year

    @property
    def annual_income(self) -> int:
        return self.monthly_income_avg * 12


class LoanProduct(Base):
    """policies와 동일한 자격요건 컬럼 규약 + trigger_events.

    match_for_loan_event() 가 이 규약(속성명 동일)에 의존해 Policy와 같은
    _eligibility_check() 로직을 공유한다. 필드를 바꿀 땐 Policy와 짝을 맞출 것.
    """

    __tablename__ = "loan_products"

    product_id = Column(String(32), primary_key=True)
    product_name = Column(String(100), nullable=False)
    product_type = Column(
        Enum("전월세자금", "신용대출", "학자금", "사업자", "마이너스통장"), nullable=False
    )
    min_rate = Column(Numeric(4, 2), nullable=False)
    max_rate = Column(Numeric(4, 2), nullable=False)
    max_amount = Column(BigInteger, nullable=False)
    target_desc = Column(String(255), nullable=True)

    min_age = Column(SmallInteger, nullable=True)
    max_age = Column(SmallInteger, nullable=True)
    max_annual_income = Column(BigInteger, nullable=True)
    allowed_employment = Column(JSON, nullable=True)
    allowed_housing = Column(JSON, nullable=True)
    allowed_marital = Column(JSON, nullable=True)
    region_code = Column(String(10), nullable=True)

    trigger_events = Column(JSON, nullable=False)


class UserLoan(Base):
    __tablename__ = "user_loans"

    loan_id = Column(String(32), primary_key=True)
    user_id = Column(String(32), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    product_id = Column(String(32), ForeignKey("loan_products.product_id"), nullable=False)
    principal = Column(BigInteger, nullable=False)
    balance = Column(BigInteger, nullable=False)
    interest_rate = Column(Numeric(4, 2), nullable=False)
    monthly_payment = Column(Integer, nullable=False)
    due_day = Column(SmallInteger, nullable=False)
    started_at = Column(Date, nullable=False)
    maturity_at = Column(Date, nullable=False)
    status = Column(Enum("정상", "연체", "완제"), nullable=False, default="정상")

    user = relationship("User", back_populates="loans")
    product = relationship("LoanProduct")


class Policy(Base):
    """정책 자격요건을 정형 컬럼으로 보관 → A파트가 SQL/파이썬 비교만으로 결정론적 판정."""

    __tablename__ = "policies"

    policy_id = Column(String(32), primary_key=True)
    policy_name = Column(String(150), nullable=False)
    provider = Column(String(50), nullable=False)
    category = Column(Enum("주거", "고용", "창업", "결혼출산", "금융", "교육"), nullable=False)
    benefit_summary = Column(String(500), nullable=False)
    apply_url = Column(String(255), nullable=True)

    min_age = Column(SmallInteger, nullable=True)
    max_age = Column(SmallInteger, nullable=True)
    max_annual_income = Column(BigInteger, nullable=True)
    allowed_employment = Column(JSON, nullable=True)
    allowed_housing = Column(JSON, nullable=True)
    allowed_marital = Column(JSON, nullable=True)
    region_code = Column(String(10), nullable=True)

    trigger_events = Column(JSON, nullable=False)
    priority = Column(SmallInteger, nullable=False, default=5)

    benefit_amount_krw = Column(BigInteger, nullable=True)
    benefit_period_month = Column(SmallInteger, nullable=True)
    benefit_rate_pct = Column(Numeric(4, 2), nullable=True)
    source = Column(Enum("manual", "youthcenter_api"), nullable=False, default="manual")
    external_policy_no = Column(String(30), nullable=True)


class Transaction(Base):
    __tablename__ = "transactions"

    tx_id = Column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    tx_date = Column(Date, nullable=False)
    amount = Column(BigInteger, nullable=False)
    counterparty = Column(String(100), nullable=False)
    category = Column(String(30), nullable=False)
    memo = Column(String(255), nullable=True)

    user = relationship("User", back_populates="transactions")


class LifeEvent(Base):
    """확정된 생애주기 이벤트. C엔진 입력(확정 히스토리)의 원천."""

    __tablename__ = "life_events"

    event_id = Column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String(30), nullable=False)
    occurred_at = Column(Date, nullable=False)
    offset_month = Column(SmallInteger, nullable=False, default=0)
    prev_gap_month = Column(SmallInteger, nullable=True)
    status = Column(Enum("detected", "confirmed", "rejected"), nullable=False, default="detected")
    detected_by = Column(Enum("rule", "user", "seed"), nullable=False, default="rule")
    confidence = Column(Numeric(3, 2), nullable=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    user = relationship("User", back_populates="events")


class CohortSequence(Base):
    """합성 코호트 300개 + 사전 임베딩. C파트 CohortIndex 가 그대로 읽어간다."""

    __tablename__ = "cohort_sequences"

    cohort_id = Column(Integer, primary_key=True, autoincrement=True)
    history_json = Column(JSON, nullable=False)
    event_history_text = Column(String(500), nullable=False)
    next_event = Column(String(30), nullable=False)
    history_length = Column(SmallInteger, nullable=False)
    embedding_vector = Column(JSON, nullable=False)
    embedding_model = Column(String(50), nullable=False)
    embedding_dim = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class Prediction(Base):
    """C엔진 예측 결과 로그. trigger_event_id 로 agentic 순환의 인과관계를 추적한다."""

    __tablename__ = "predictions"

    prediction_id = Column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    trigger_event_id = Column(
        BigInteger, ForeignKey("life_events.event_id", ondelete="SET NULL"), nullable=True
    )
    input_history_json = Column(JSON, nullable=False)
    predictions_json = Column(JSON, nullable=False)
    confidence_level = Column(Enum("high", "medium", "low"), nullable=False)
    confidence_note = Column(String(500), nullable=True)
    matched_cohorts_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class PolicyMatchResult(Base):
    """A파트 산출물: 예측/확정 이벤트로 인해 자격이 새로 생기거나 사라진 정책."""

    __tablename__ = "policy_match_results"

    match_id = Column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    prediction_id = Column(
        BigInteger, ForeignKey("predictions.prediction_id", ondelete="SET NULL"), nullable=True
    )
    policy_id = Column(String(32), ForeignKey("policies.policy_id"), nullable=False)
    basis_event = Column(String(30), nullable=False)
    status = Column(
        Enum("newly_eligible", "eligible", "lost", "not_eligible"), nullable=False
    )
    reason = Column(String(500), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    policy = relationship("Policy")
