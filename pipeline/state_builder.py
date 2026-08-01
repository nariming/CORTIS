"""
User State Builder — 유저의 DB 프로필/대출/거래 feature를 하나의 State로 결합.

지금까지 C엔진(embedding/reasoning)은 확정된 이벤트 히스토리(history)만 보고 검색·판단해 왔다.
이 모듈은 "이 유저가 지금 어떤 재무 상태에 있는가"를 별도 축(State)으로 구성해,
State Embedding과 LLM user_context 양쪽에 동일한 재료를 공급하는 단일 창구 역할을 한다.

설계 원칙
  - State에 담기는 값은 전부 DB 컬럼이거나 tx_features.py/portfolio.py의 결정론적 계산 결과다.
    LLM은 이 값을 서술/해석할 수 있어도 값 자체를 생성하지 않는다.
  - DSR은 별도로 재계산하지 않고 portfolio.py의 기존 평가 로직(evaluate_combo)을 그대로
    재사용한다. 모든 대출을 KEEP으로 두고 평가하면 그 결과가 곧 "현재 DSR"이므로, 포트폴리오
    결정 로직과 State 조회 로직 사이에 서로 다른 DSR 계산식이 생기는 것을 방지한다.
  - 성별은 State 스키마에서 제외했다. 현재 users 테이블에 해당 컬럼이 없고, 이를 자격요건으로
    요구하는 정책 유형도 확인되지 않았기 때문이다.
  - 임대차 보증금/월세의 정확한 금액은 users 테이블에 전용 컬럼이 없어(housing_type만 보유)
    이번 버전에는 포함하지 않았다. 거래내역에서 월세 이체 금액을 추정하는 방식으로 보완할 수
    있으며, 그 경우에도 값의 출처는 거래내역 집계여야 한다(LLM 추정 금지 원칙과 동일선상).
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from backend.db.models import Transaction, User, UserLoan
from pipeline.portfolio import ExistingLoan, LoanAction, UserFinancialProfile, evaluate_combo
from pipeline.tx_features import TransactionFeatures, extract_transaction_features


@dataclass
class UserState:
    """State Embedding 문장 및 LLM user_context에 공급되는 유저 재무 상태 스냅샷."""

    user_id: str
    age: int
    employment_type: str
    monthly_income: int
    housing_type: str
    marital_status: str
    credit_score: Optional[int]
    liquid_assets_krw: int
    loan_count: int
    loan_balance_total: int
    dsr_pct: Optional[float]  # None = 보유 대출이 없어 계산 대상이 아님(0%와는 다른 의미)
    tx_features: TransactionFeatures

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "employment_type": self.employment_type,
            "monthly_income": self.monthly_income,
            "housing_type": self.housing_type,
            "marital_status": self.marital_status,
            "credit_score": self.credit_score,
            "liquid_assets_krw": self.liquid_assets_krw,
            "loan_count": self.loan_count,
            "loan_balance_total": self.loan_balance_total,
            "dsr_pct": self.dsr_pct,
            "tx_features": self.tx_features.to_dict(),
        }

    def to_sentence(self) -> str:
        """State Embedding 대상 문장. history 문장과는 독립된 임베딩 공간에 들어간다."""
        credit_part = f"신용점수 {self.credit_score}" if self.credit_score is not None else "신용점수 미확인"
        dsr_part = f"DSR {self.dsr_pct:.1f}%" if self.dsr_pct is not None else "보유 대출 없음"
        return (
            f"{self.age}세 {self.employment_type}, 월소득 {self.monthly_income:,}원, "
            f"거주형태 {self.housing_type}, 혼인상태 {self.marital_status}, {credit_part}, "
            f"{dsr_part}, 여유자금 {self.liquid_assets_krw:,}원. "
            f"{self.tx_features.to_sentence()}"
        )


def _existing_loans_from_orm(loans: List[UserLoan], today: Optional[date] = None) -> List[ExistingLoan]:
    """UserLoan(ORM) -> portfolio.ExistingLoan 변환.

    pipeline/backend_client.py의 build_existing_loans_from_user_detail()이 HTTP 응답(dict)을
    같은 방식으로 변환하는 것과 동일한 규칙(정상 상태만 포함, remaining_months = 만기 - 기준일)을
    ORM 객체에 적용한 것이다. 두 경로(HTTP 응답 경유 vs 백엔드 프로세스 내 직접 ORM 접근)가
    서로 다른 DSR/NPV 결과를 내지 않도록 계산 규칙을 통일했다.
    """
    today = today or date.today()
    result: List[ExistingLoan] = []
    for loan in loans:
        if loan.status != "정상":
            continue
        remaining_months = max(
            (loan.maturity_at.year - today.year) * 12 + (loan.maturity_at.month - today.month), 1
        )
        result.append(
            ExistingLoan(
                loan_id=loan.loan_id,
                product_id=loan.product_id,
                balance=loan.balance,
                interest_rate=float(loan.interest_rate),
                monthly_payment=loan.monthly_payment,
                remaining_months=remaining_months,
                rate_type=loan.rate_type,
            )
        )
    return result


def _current_dsr_pct(user: User, loans: List[ExistingLoan]) -> Optional[float]:
    """현재 DSR = 보유 대출 전부를 KEEP으로 두고 평가한 조합의 DSR.

    portfolio.evaluate_combo()를 그대로 호출해 포트폴리오 결정 로직과 별개의 DSR 공식이
    생기는 것을 막는다. 대출이 없으면 DSR 자체가 정의되지 않으므로 0.0이 아니라 None을 반환한다.
    """
    if not loans:
        return None
    profile = UserFinancialProfile(
        user_id=user.user_id,
        annual_income=user.annual_income,
        income_volatility=float(user.income_volatility),
        employment_type=user.employment_type,
        region_code=user.region_code,
        liquid_assets_krw=user.liquid_assets_krw,
    )
    keep_combo = [LoanAction(loan_id=ln.loan_id, action="KEEP") for ln in loans]
    metrics = evaluate_combo(keep_combo, loans, profile)
    return metrics.dsr_pct


def build_user_state(
    user: User,
    txs: Optional[List[Transaction]] = None,
    as_of: Optional[date] = None,
) -> UserState:
    """State Embedding/LLM user_context에 공급할 UserState를 조립하는 단일 진입점.

    txs를 별도로 넘기지 않으면 user.transactions(ORM relationship)를 그대로 사용한다.
    호출부(예: events.py)가 이미 같은 요청 안에서 거래내역을 조회해 둔 경우, 그 결과를
    그대로 주입해 중복 쿼리를 피할 수 있도록 매개변수로 열어 두었다.

    as_of는 이 함수에서 한 번만 확정해 대출 잔여기간 계산과 거래 트렌드 계산 양쪽에
    동일하게 전달한다. 두 계산이 각자 다른 기본값(예: 대출 쪽은 실행 시점의 실제 날짜,
    거래 쪽은 거래내역의 마지막 날짜)을 쓰면 "최근 N개월"이 서로 다른 구간을 가리키게 되어
    같은 유저의 상태를 놓고 서로 모순된 값이 나올 수 있다 — 이를 방지하기 위한 단일화다.
    """
    resolved_txs = txs if txs is not None else list(user.transactions)
    resolved_as_of = as_of or (
        max((t.tx_date for t in resolved_txs), default=date.today())
    )

    existing_loans = _existing_loans_from_orm(list(user.loans), today=resolved_as_of)
    tx_features = extract_transaction_features(resolved_txs, as_of=resolved_as_of)

    return UserState(
        user_id=user.user_id,
        age=user.age,
        employment_type=user.employment_type,
        monthly_income=user.monthly_income_avg,
        housing_type=user.housing_type,
        marital_status=user.marital_status,
        credit_score=user.credit_score,
        liquid_assets_krw=user.liquid_assets_krw,
        loan_count=len(existing_loans),
        loan_balance_total=sum(ln.balance for ln in existing_loans),
        dsr_pct=_current_dsr_pct(user, existing_loans),
        tx_features=tx_features,
    )