"""
Strategy Generator — "기존 대출이 없는데 새로 자금이 필요한 상황"의 후보 전략 생성.

문제의식 (설계 검토 및 실제 코드 확인 결과)
  agent_loop.py의 _run_portfolio_decision()은 지금 이렇게 시작한다:

      loans = backend_client.build_existing_loans_from_user_detail(user_detail)
      if not loans:
          return  # 보유 대출 없으면 포트폴리오 재설계 대상 없음

  즉 "기존 대출을 어떻게 조정할지"(KEEP/PREPAY/REFINANCE)는 portfolio.py가 이미 잘 계산하지만,
  "기존 대출이 없는데 예측된 이벤트(독립 등) 때문에 처음으로 대출이 필요해지는 상황"은
  포트폴리오 결정 단계 자체가 스킵되어 아무 전략도 제시하지 않는다.

  기획서 원문에도 있던 "추가 대출 필요성 자동 검토"가 사실 이 구멍이다.

역할 분담 원칙 (portfolio.py와 동일하게 계승)
  - 후보는 LLM이 아니라 전수탐색으로 뽑는다 (WAIT / CASH_ONLY / 자격 되는 정책·일반대출 상품 전부).
  - 숫자(NPV, 월상환액)는 이 파일이 계산하지 않고, portfolio.py의 기존 순수 계산 함수
    (_npv_cost, _new_monthly_payment)를 그대로 재사용한다 - 계산 로직이 두 곳에서 따로
    구현되면 정합성이 깨진다는 기존 원칙(backend_client.py 등에서 이미 여러 번 명시된 원칙)을
    그대로 따른다.
  - cash_need_krw가 없으면(코호트 집계 부족 등) 후보 자체를 만들지 않는다 - "근거 없는 숫자를
    지어내지 않는다"는 파이프라인 전체의 핵심 방어 논리를 여기서도 그대로 지킨다.

신규 대출 NPV 계산 시 중도상환수수료가 0이 되는 이유
  portfolio._npv_cost()의 REFINANCE 분기는 fee = balance * 관행수수료율 * (3 - loan_age_years)/3
  로 "기존 대출을 대체하는 데 드는 중도상환수수료"를 계산한다. 그런데 지금 다루는 건 대체할
  기존 대출이 아예 없는 "신규 실행"이라 수수료 개념 자체가 없다. loan_age_years=3.0을 넘겨서
  fee가 자동으로 0이 되게 하는 방식으로 처리한다(새 파라미터를 추가하지 않고 기존 공식의
  "3년 지나면 수수료 면제"라는 의미를 그대로 활용 - 방어 논리상 새 개념을 안 만드는 쪽이 낫다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .portfolio import (
    ExistingLoan,
    RefinanceCandidate,
    LoanAction,
    UserFinancialProfile,
    _npv_cost,
    _new_monthly_payment,
)

NewLoanActionType = Literal["POLICY_LOAN", "GENERAL_LOAN", "CASH_ONLY", "WAIT"]

# 상품 카탈로그(RefinanceCandidate)에 만기 정보가 없어, 신규 대출 상환기간을 알 수 없다.
# 정책 전월세자금대출류의 통상 만기(2년 단위 갱신, 보통 최장 8~10년 분할)를 참고해 보수적으로
# 60개월(5년)을 기본값으로 둔다 - 발표 시 "상품별 실제 만기 데이터 연동 전 근사치"로 명시할 것.
DEFAULT_NEW_LOAN_TERM_MONTHS = 60


@dataclass
class NewLoanCandidate:
    """기존 대출이 없는 상태에서 cash_need를 충당하기 위한 전략 후보 하나."""

    action: NewLoanActionType
    target_product: Optional[RefinanceCandidate] = None  # POLICY_LOAN/GENERAL_LOAN일 때만
    cash_need_krw: int = 0
    label: str = ""  # 사람이 읽을 짧은 설명 (LLM 설명 단계에서 그대로 활용 가능)


@dataclass
class NewLoanMetrics:
    candidate: NewLoanCandidate
    monthly_payment: int
    npv_total_cost: float
    dsr_pct: float
    feasible: bool
    infeasible_reason: Optional[str] = None


def generate_new_loan_candidates(
    event_name: str,
    cash_need_krw: Optional[int],
    liquid_assets_krw: Optional[int],
    policy_candidates: List[RefinanceCandidate],
    general_loan_candidates: List[RefinanceCandidate],
    event_confirmed: bool = True,
) -> List[NewLoanCandidate]:
    """실행 가능한 후보를 전수탐색으로 나열한다 (LLM 호출 없음).

    policy_candidates/general_loan_candidates는 이미 A파트 자격판정(_check_eligibility)을
    통과한 상품만 들어있다고 가정한다 (backend_client.request_policy_match /
    request_loan_match가 이미 그렇게 필터링해서 반환함) - 여기서 자격을 다시 판정하지 않는다.

    cash_need_krw가 None이면(코호트 집계 부족 등) 빈 리스트를 반환한다 - 근거 없이 후보 자체를
    만들지 않는다.

    event_confirmed 처리 (실제 테스트로 발견한 문제 반영)
      WAIT은 "이벤트가 아직 확정 안 됐으니 대기"라는 의미인데, event_confirmed=True(이미
      확정된 이벤트)일 때 이 후보를 넣으면 두 가지 문제가 생긴다:
        1) 이미 벌어진 일에 "대기하라"는 건 논리적으로 말이 안 됨
        2) WAIT은 이자비용이 없어 NPV가 항상 0원이라, 실제 자금 조달이 필요한 상황에서도
           "아무것도 안 하는 것"이 숫자상 항상 최선으로 나와 추천 기능 자체가 무력화됨
      그래서 event_confirmed=True면 WAIT을 아예 후보에서 뺀다. CASH_ONLY(유동자산으로 충당)는
      확정 이벤트에서도 여전히 유효한 "0원 선택지"라서 그대로 둔다 - 이건 실제로 돈이 있어서
      대출이 필요 없다는 뜻이지 "아무것도 안 한다"는 뜻이 아니기 때문.
    """
    if cash_need_krw is None or cash_need_krw <= 0:
        return []

    candidates: List[NewLoanCandidate] = []

    if not event_confirmed:
        candidates.append(
            NewLoanCandidate(
                action="WAIT",
                cash_need_krw=cash_need_krw,
                label=f"'{event_name}' 아직 확정 전 - 확정 시점까지 대기 후 재계산",
            )
        )

    if liquid_assets_krw is not None and liquid_assets_krw >= cash_need_krw:
        candidates.append(
            NewLoanCandidate(
                action="CASH_ONLY",
                cash_need_krw=cash_need_krw,
                label=f"보유 유동자산({liquid_assets_krw:,}원)으로 대출 없이 충당",
            )
        )

    for product in policy_candidates:
        candidates.append(
            NewLoanCandidate(
                action="POLICY_LOAN",
                target_product=product,
                cash_need_krw=cash_need_krw,
                label=f"{product.product_name} 신규 신청",
            )
        )

    for product in general_loan_candidates:
        candidates.append(
            NewLoanCandidate(
                action="GENERAL_LOAN",
                target_product=product,
                cash_need_krw=cash_need_krw,
                label=f"{product.product_name} 신규 신청",
            )
        )

    return candidates


def _as_virtual_loan(candidate: NewLoanCandidate, term_months: int) -> ExistingLoan:
    """신규 대출 후보를 portfolio.py의 기존 계산 함수가 받아들이는 ExistingLoan 형태로 변환.

    balance=cash_need_krw로 두면 '이만큼을 새로 빌린다'는 의미가 되고, interest_rate/
    monthly_payment는 아직 실행 전이라 의미가 없어 0으로 둔다(실제 계산은 target_product의
    금리를 쓰는 REFINANCE 액션 쪽에서 이뤄짐).
    """
    return ExistingLoan(
        loan_id=f"NEW_{candidate.action}",
        product_id=candidate.target_product.product_id if candidate.target_product else "NONE",
        balance=candidate.cash_need_krw,
        interest_rate=0.0,
        monthly_payment=0,
        remaining_months=term_months,
        rate_type=candidate.target_product.rate_type if candidate.target_product else "변동",
    )


def evaluate_new_loan_candidate(
    candidate: NewLoanCandidate,
    profile: UserFinancialProfile,
    term_months: int = DEFAULT_NEW_LOAN_TERM_MONTHS,
    dsr_cap_pct: float = 40.0,
) -> NewLoanMetrics:
    """후보 하나를 평가 (월상환액/NPV/DSR). portfolio.py의 기존 계산 함수를 그대로 재사용."""

    if candidate.action == "WAIT":
        return NewLoanMetrics(
            candidate=candidate, monthly_payment=0, npv_total_cost=0.0, dsr_pct=0.0, feasible=True,
        )

    if candidate.action == "CASH_ONLY":
        # 대출이 아니므로 이자비용 자체가 없다 - NPV/DSR 모두 0.
        return NewLoanMetrics(
            candidate=candidate, monthly_payment=0, npv_total_cost=0.0, dsr_pct=0.0, feasible=True,
        )

    # POLICY_LOAN / GENERAL_LOAN: 실제 신규 대출 실행
    virtual_loan = _as_virtual_loan(candidate, term_months)
    action = LoanAction(loan_id=virtual_loan.loan_id, action="REFINANCE", target_product=candidate.target_product)

    monthly_payment = _new_monthly_payment(virtual_loan, action)
    # loan_age_years=3.0 -> 중도상환수수료 공식이 자동으로 0이 됨 (대체할 기존 대출이 없으므로).
    npv_cost = _npv_cost(virtual_loan, action, loan_age_years=3.0)

    dsr_pct = (monthly_payment * 12 / profile.annual_income) * 100 if profile.annual_income else 0.0
    feasible = dsr_pct <= dsr_cap_pct
    reason = None if feasible else f"이 상품으로 실행 시 DSR {dsr_pct:.1f}%가 상한 {dsr_cap_pct}% 초과"

    return NewLoanMetrics(
        candidate=candidate,
        monthly_payment=monthly_payment,
        npv_total_cost=round(npv_cost),
        dsr_pct=round(dsr_pct, 1),
        feasible=feasible,
        infeasible_reason=reason,
    )


def select_best_new_loan_strategy(
    candidates: List[NewLoanCandidate],
    profile: UserFinancialProfile,
    term_months: int = DEFAULT_NEW_LOAN_TERM_MONTHS,
    dsr_cap_pct: float = 40.0,
) -> tuple:
    """전체 파이프라인: 후보 평가 -> 제약필터(DSR) -> NPV 오름차순 정렬.

    portfolio.select_best_portfolio()와 동일한 계층형(lexicographic) 원칙: 가중합이 아니라
    "제약 통과 -> 비용 낮은 순" 우선순위 정렬. CASH_ONLY는 NPV 0이라 대출 실행보다 항상
    우선순위가 높게 나오는데, 이건 의도된 동작이다 - 유동자산으로 충당 가능하면 그게 실제로
    빚을 안 지는 최선의 선택이기 때문이다. WAIT은 event_confirmed=True일 때 애초에
    generate_new_loan_candidates()가 후보로 넣지 않으므로(이미 확정된 이벤트에 "대기하라"는
    답은 나오지 않음), 확정 이벤트에서는 이 함수가 "아무것도 안 하는 선택지"와 "실제 자금
    조달 방법"을 잘못 저울질하는 일이 없다.

    반환: (선택된 1위 NewLoanMetrics, 비교용 2위 NewLoanMetrics | None, 탈락 사유 리스트)
    """
    evaluated = [evaluate_new_loan_candidate(c, profile, term_months, dsr_cap_pct) for c in candidates]

    feasible = [m for m in evaluated if m.feasible]
    infeasible = [m for m in evaluated if not m.feasible]

    feasible.sort(key=lambda m: (m.npv_total_cost, m.monthly_payment))

    if not feasible:
        return None, None, [m.infeasible_reason for m in infeasible]

    best = feasible[0]
    second = feasible[1] if len(feasible) > 1 else None
    return best, second, [m.infeasible_reason for m in infeasible]