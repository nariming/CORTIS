"""API 요청/응답 스키마 (pydantic v2)."""

from datetime import date, datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------- 유저
class LoanOut(ORMModel):
    loan_id: str
    product_id: str
    principal: int
    balance: int
    interest_rate: float
    rate_type: str
    monthly_payment: int
    due_day: int
    started_at: date
    maturity_at: date
    status: str


class UserOut(ORMModel):
    user_id: str
    name: str
    birth_year: int
    employment_type: str
    monthly_income_avg: int
    liquid_assets_krw: int = Field(description="여유자금/예금 잔액(원) — 조기상환 여력 판단용")
    income_volatility: float
    marital_status: str
    housing_type: str
    region_code: str
    credit_score: Optional[int] = None


class UserDetailOut(UserOut):
    age: int
    annual_income: int
    loans: List[LoanOut] = []
    confirmed_history: List[str] = []


class TransactionOut(ORMModel):
    tx_id: int
    tx_date: date
    amount: int
    counterparty: str
    category: str
    memo: Optional[str] = None


# --------------------------------------------------------------------- 이벤트
class LifeEventOut(ORMModel):
    event_id: int
    user_id: str
    event_type: str
    occurred_at: date
    offset_month: int
    prev_gap_month: Optional[int] = None
    status: str
    detected_by: str
    confidence: Optional[float] = None
    evidence: Optional[Any] = None
    created_at: datetime


class EventCandidateOut(BaseModel):
    event_type: str
    occurred_at: date
    confidence: float
    reason: str
    evidence_tx_ids: List[int] = []
    event_id: Optional[int] = Field(
        None, description="detected 상태로 저장된 life_events.event_id — 확정 API에 그대로 넘기면 됨"
    )


class DetectResponse(BaseModel):
    user_id: str
    scanned_transactions: int
    confirmed_history: List[str]
    candidates: List[EventCandidateOut]


class EventCreateIn(BaseModel):
    event_type: str
    occurred_at: date
    status: str = Field("confirmed", pattern="^(detected|confirmed)$")
    confidence: Optional[float] = None


class HistoryOut(BaseModel):
    """C엔진이 예측 직전에 읽어가는 최소 페이로드."""

    user_id: str
    confirmed_history: List[str]
    is_cold_start: bool
    cold_start_note: Optional[str] = None
    user_context: str = Field(description="LLM 프롬프트에 그대로 넣을 유저 현재 상황 요약")


# --------------------------------------------------------------------- 코호트
class CohortRowOut(BaseModel):
    """CohortIndex.load_from_mysql_rows() 가 기대하는 형태 그대로."""

    cohort_id: int
    history: List[str]
    event_history_text: str
    next_event: str
    embedding_vector: List[float]


class CohortStatsOut(BaseModel):
    total: int
    embedding_model: str
    next_event_distribution: dict


# --------------------------------------------------------------------- 정책 (A파트)
class PolicyOut(ORMModel):
    policy_id: str
    policy_name: str
    provider: str
    category: str
    benefit_summary: str
    apply_url: Optional[str] = None
    trigger_events: List[str]
    priority: int
    benefit_amount_krw: Optional[int] = Field(None, description="1회성/총 정액 지원금(원)")
    benefit_period_month: Optional[int] = Field(None, description="정액 지원 지속 개월수")
    benefit_rate_pct: Optional[float] = Field(None, description="금리 인하/적용 금리(%p)")
    source: str = Field(description="manual | youthcenter_api")


class PolicyMatchOut(BaseModel):
    policy_id: str
    policy_name: str
    provider: str
    category: str
    benefit_summary: str
    apply_url: Optional[str] = None
    status: str
    reason: str
    assumption: Optional[str] = Field(
        None, description="예측 모드에서 가정한 상태 변화 (예: '독립(월세)' 발생 가정)"
    )
    passed: List[str] = []
    failed: List[str] = []


class EventPolicyGroupOut(BaseModel):
    basis_event: str
    matched_count: int
    policies: List[PolicyMatchOut]


# --------------------------------------------------------------------- 대출상품 (A파트, LoanProduct)
class LoanProductOut(ORMModel):
    product_id: str
    product_name: str
    product_type: str
    rate_type: str
    min_rate: float
    max_rate: float
    max_amount: int
    target_desc: Optional[str] = None
    trigger_events: List[str]


class LoanProductMatchOut(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    rate_type: str
    min_rate: float
    max_rate: float
    max_amount: int
    target_desc: Optional[str] = None
    status: str
    reason: str
    assumption: Optional[str] = None
    passed: List[str] = []
    failed: List[str] = []


class EventLoanGroupOut(BaseModel):
    basis_event: str
    matched_count: int
    loan_products: List[LoanProductMatchOut]


class LoanMatchIn(BaseModel):
    """C엔진 예측 결과를 받아 KB 대출상품 자격을 재판단하는 요청 (PolicyMatchIn과 동일 규약)."""

    event_types: List[str] = Field(description="예측/확정된 이벤트 목록", min_length=1)
    include_ineligible: bool = False
    prospective: bool = True


class PolicyMatchIn(BaseModel):
    """C엔진 예측 결과를 받아 A파트를 돌리는 요청."""

    event_types: List[str] = Field(description="예측/확정된 이벤트 목록", min_length=1)
    prediction_id: Optional[int] = None
    include_ineligible: bool = False
    prospective: bool = Field(
        True,
        description=(
            "True = 아직 안 일어난 예측 이벤트 기준 사전 안내 (이벤트 발생 후 프로필을 가정). "
            "False = 이미 확정된 이벤트 기준 현재 프로필 그대로 판정."
        ),
    )
    persist: bool = Field(True, description="매칭 결과를 policy_match_results 에 기록할지")


# --------------------------------------------------------------------- 예측 (C엔진 결과 저장)
class PredictionIn(BaseModel):
    trigger_event_id: Optional[int] = None
    input_history: List[str]
    predictions: List[dict]
    confidence_level: str = Field(pattern="^(high|medium|low)$")
    confidence_note: Optional[str] = None
    matched_cohorts: Optional[List[dict]] = None


class PredictionOut(ORMModel):
    prediction_id: int
    user_id: str
    trigger_event_id: Optional[int] = None
    input_history_json: List[str]
    predictions_json: List[dict]
    confidence_level: str
    confidence_note: Optional[str] = None
    matched_cohorts_json: Optional[List[dict]] = None
    created_at: datetime
