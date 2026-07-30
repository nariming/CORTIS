"""
포트폴리오 결정 레이어: C(이벤트 예측) + A(정책/대출상품 자격 매칭) 산출물을 받아
"어떤 대출상품을 어떤 구조로 언제 구성할지" 하나의 타겟 포트폴리오를 도출한다.

핵심 원칙 (팀 논의 결론):
  - 숫자(NPV, DSR)는 전부 이 파일의 순수 함수가 계산한다. LLM은 절대 숫자를 만들지 않는다.
  - 가중합을 쓰지 않는다. 대신 "제약조건 필터 -> 우선순위 규칙"의 계층형(lexicographic) 구조.
  - 전략은 LLM이 제안하지 않는다. 유한한 액션 공간을 전수탐색(enumerate)해서 뽑는다.

TODO(개발자1 PR #5 merge 후 반영): LoanProduct/UserLoan에 rate_type(고정/변동/혼합형/주기형)
필드가 추가되면 stress_test()의 "전부 변동금리로 가정" 부분을 실제 분기로 교체할 것.
그 전까지는 보수적으로 전부 변동금리 취급 (스트레스 금리 100% 적용) - 안전한 쪽으로 근사.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Literal


# ---------------------------------------------------------------------------
# 0. 데이터 구조 (A/backend API 응답을 이 형태로 매핑해서 사용)
# ---------------------------------------------------------------------------

@dataclass
class UserFinancialProfile:
    """User 테이블 + 파생 필드. backend GET /users/{user_id} 응답에서 매핑."""
    user_id: str
    annual_income: int              # KRW
    income_volatility: float        # User.income_volatility 그대로
    employment_type: str            # "정규직" / "프리랜서" / "계약직" / "무직" 등
    region_code: str                # "11" = 서울(수도권), 그 외 = 비수도권으로 근사
    # TODO(개발자1 필드 추가 대기 중): User.liquid_assets_krw 필드가 생기면 여기 채우기.
    # None인 동안은 PREPAY 액션을 아예 후보에서 제외한다 (현금 있는지 모르는 채로
    # "전액 조기상환이 항상 제일 이득"이라는 잘못된 결론이 나오는 걸 막기 위함).
    liquid_assets_krw: Optional[int] = None


@dataclass
class ExistingLoan:
    """UserLoan 한 건. GET /users/{user_id} 의 loans[] 에서 매핑."""
    loan_id: str
    product_id: str
    balance: int                    # 현재 잔액 (KRW)
    interest_rate: float            # 현재 적용금리 (%), UserLoan.interest_rate
    monthly_payment: int            # 현재 월 상환액 (KRW)
    remaining_months: int           # maturity_at 기준 잔여 개월수 (호출부에서 계산해 채움)


@dataclass
class RefinanceCandidate:
    """대환 후보 상품. A의 POST /users/{user_id}/loan-match 응답에서 매핑."""
    product_id: str
    product_name: str
    min_rate: float
    max_rate: float
    max_amount: int

    def assumed_rate(self) -> float:
        """실제 심사 전이라 확정 금리를 모르므로, 보수적으로 min_rate(최저금리)를 가정.
        결과 문구에도 반드시 '최저금리 가정'이라고 명시해야 한다 (근거 없는 낙관 방지)."""
        return self.min_rate


# ---------------------------------------------------------------------------
# 1. 액션 정의 (Stage A: WHAT)
# ---------------------------------------------------------------------------

ActionType = Literal["KEEP", "PREPAY", "REFINANCE"]


@dataclass
class LoanAction:
    loan_id: str
    action: ActionType
    target_product: Optional[RefinanceCandidate] = None  # REFINANCE일 때만
    prepay_amount: Optional[int] = None  # PREPAY일 때만. None이면 전액상환 의미.


def build_action_candidates(
    loan: ExistingLoan,
    refinance_options: List[RefinanceCandidate],
    liquid_assets_krw: Optional[int] = None,
) -> List[LoanAction]:
    """대출 한 건에 대해 가능한 액션 리스트 생성 (유지/조기상환/대환 후보들).

    liquid_assets_krw가 None이면 PREPAY 자체를 후보에서 뺀다 (개발자1이 필드 추가 대기 중).
    값이 있으면, 여유자금과 대출잔액 중 작은 쪽만큼만 조기상환 가능하도록 상한을 건다
    (여유자금 < 대출잔액이면 '일부상환'으로 자동 조정).
    """
    actions: List[LoanAction] = [
        LoanAction(loan_id=loan.loan_id, action="KEEP"),
    ]
    if liquid_assets_krw is not None and liquid_assets_krw > 0:
        prepay_amount = min(liquid_assets_krw, loan.balance)
        actions.append(LoanAction(loan_id=loan.loan_id, action="PREPAY", prepay_amount=prepay_amount))
    for candidate in refinance_options:
        actions.append(LoanAction(loan_id=loan.loan_id, action="REFINANCE", target_product=candidate))
    return actions


def generate_portfolio_combinations(
    loans: List[ExistingLoan],
    refinance_map: dict,  # loan_id -> List[RefinanceCandidate]
    liquid_assets_krw: Optional[int] = None,
) -> List[List[LoanAction]]:
    """대출별 액션 리스트의 카테시안 곱 = 포트폴리오 후보 전체.

    대출 개수가 적어(보통 1~3개) 완전탐색이 부담 없다. 최적화 알고리즘 불필요.
    """
    per_loan_actions = [
        build_action_candidates(loan, refinance_map.get(loan.loan_id, []), liquid_assets_krw)
        for loan in loans
    ]
    combinations = list(itertools.product(*per_loan_actions))
    return [list(combo) for combo in combinations]


# ---------------------------------------------------------------------------
# 2. 계산 엔진 (NPV / DSR / 스트레스 테스트) - 전부 결정론적
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMetrics:
    combo: List[LoanAction]
    monthly_payment_total: int      # 조합 적용 후 월 총 상환액
    dsr_pct: float                  # 조합 적용 후 DSR (%)
    npv_total_cost: float           # 조합의 총비용 NPV (이자, 원 단위)
    npv_savings_vs_keep: float      # '전부 유지' 대비 절감액 (양수 = 절감)
    dsr_stress_pct: float           # 스트레스 금리 적용 시 DSR (%, 변동금리 100% 가정)
    feasible: bool = True
    infeasible_reason: Optional[str] = None


def _new_monthly_payment(loan: ExistingLoan, action: LoanAction) -> int:
    """액션 적용 후 이 대출의 월 상환액. 원리금 상환구조 세부 반영 없이
    '남은 원금을 남은 개월수로 균등분할 + 이자' 근사 (조기상환수수료 등 부대비용은 NPV에서 별도 반영)."""
    if action.action == "PREPAY":
        prepay = action.prepay_amount or loan.balance
        if prepay >= loan.balance:
            return 0  # 전액상환 -> 월상환액 소멸
        # 일부상환: 남은 원금 기준으로 기존 금리 그대로 재분할 (근사)
        remaining_balance = loan.balance - prepay
        monthly_rate = loan.interest_rate / 100 / 12
        n = max(loan.remaining_months, 1)
        if monthly_rate == 0:
            return round(remaining_balance / n)
        payment = remaining_balance * monthly_rate / (1 - (1 + monthly_rate) ** (-n))
        return round(payment)
    if action.action == "KEEP":
        return loan.monthly_payment
    if action.action == "REFINANCE" and action.target_product:
        rate = action.target_product.assumed_rate()
        monthly_rate = rate / 100 / 12
        n = max(loan.remaining_months, 1)
        principal = loan.balance
        if monthly_rate == 0:
            return round(principal / n)
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n))
        return round(payment)
    return loan.monthly_payment


def _npv_cost(loan: ExistingLoan, action: LoanAction, discount_rate_annual: float = 0.03,
              prepay_fee_rate_convention: float = 0.014, loan_age_years: float = 0.5) -> float:
    """대출 한 건, 액션 하나의 NPV 총비용(이자+수수료, 현재가치 할인).

    prepay_fee_rate_convention: 국내 대출 중도상환수수료 통상 관행값(3년 체감형, 초기 약 1.4%).
    정확한 상품별 스케줄 데이터가 없어 관행값으로 근사 - 발표 시 '업계 통상 관행 기준 근사'로 명시할 것.
    loan_age_years: 대출 실행 후 경과년수. 데모 페르소나 데이터에서 started_at 기준 계산해 채워야 함.
    """
    n = max(loan.remaining_months, 1)
    monthly_discount = discount_rate_annual / 12

    if action.action == "KEEP":
        total = 0.0
        balance = loan.balance
        monthly_rate = loan.interest_rate / 100 / 12
        for t in range(1, n + 1):
            interest = balance * monthly_rate
            principal_paid = min(loan.monthly_payment - interest, balance)
            total += interest / ((1 + monthly_discount) ** t)
            balance -= principal_paid
            if balance <= 0:
                break
        return total

    if action.action == "PREPAY":
        prepay = action.prepay_amount or loan.balance
        fee = prepay * prepay_fee_rate_convention * max(0, (3 - loan_age_years) / 3)
        if prepay >= loan.balance:
            return fee  # 전액상환: 이후 이자 0원, 수수료만 즉시 비용으로 발생
        # 일부상환: 남은 잔액에 대해 기존 금리로 계속 이자 발생 (근사)
        remaining_balance = loan.balance - prepay
        monthly_rate = loan.interest_rate / 100 / 12
        n = max(loan.remaining_months, 1)
        new_payment = _new_monthly_payment(loan, action)
        total = fee
        balance = remaining_balance
        for t in range(1, n + 1):
            interest = balance * monthly_rate
            principal_paid = min(new_payment - interest, balance)
            total += interest / ((1 + monthly_discount) ** t)
            balance -= principal_paid
            if balance <= 0:
                break
        return total

    if action.action == "REFINANCE" and action.target_product:
        fee = loan.balance * prepay_fee_rate_convention * max(0, (3 - loan_age_years) / 3)
        rate = action.target_product.assumed_rate()
        monthly_rate = rate / 100 / 12
        new_payment = _new_monthly_payment(loan, action)
        total = fee
        balance = loan.balance
        for t in range(1, n + 1):
            interest = balance * monthly_rate
            principal_paid = min(new_payment - interest, balance)
            total += interest / ((1 + monthly_discount) ** t)
            balance -= principal_paid
            if balance <= 0:
                break
        return total

    return 0.0


def _stress_addon_pct(profile: UserFinancialProfile) -> float:
    """스트레스 DSR 가산금리 근사 (2025.7 3단계 기준, 변동금리 100% 적용 가정).
    수도권(서울=region_code '11') 주담대 3.0%p, 그 외(비수도권/신용대출) 0.75%p.
    TODO: rate_type 필드 들어오면 고정금리는 0 처리하도록 분기 추가."""
    return 3.0 if profile.region_code == "11" else 0.75


def evaluate_combo(
    combo: List[LoanAction],
    loans: List[ExistingLoan],
    profile: UserFinancialProfile,
    dsr_cap_pct: float = 40.0,
) -> PortfolioMetrics:
    """포트폴리오 후보 하나를 평가 (월상환액/DSR/NPV/스트레스DSR 전부 계산)."""
    loan_by_id = {l.loan_id: l for l in loans}

    monthly_total = 0
    npv_total = 0.0
    for action in combo:
        loan = loan_by_id[action.loan_id]
        monthly_total += _new_monthly_payment(loan, action)
        npv_total += _npv_cost(loan, action)

    dsr = (monthly_total * 12 / profile.annual_income) * 100 if profile.annual_income else 0.0

    stress_addon = _stress_addon_pct(profile)
    # 근사: 스트레스 상황에서 월상환액이 (신규금리+가산금리)/신규금리 배율만큼 커진다고 가정
    stressed_monthly = 0
    for action in combo:
        loan = loan_by_id[action.loan_id]
        if action.action == "REFINANCE" and action.target_product:
            base_rate = action.target_product.assumed_rate()
            stressed_rate = base_rate + stress_addon
            monthly_rate = stressed_rate / 100 / 12
            n = max(loan.remaining_months, 1)
            principal = loan.balance
            payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n)) if monthly_rate else principal / n
            stressed_monthly += round(payment)
        else:
            stressed_monthly += _new_monthly_payment(loan, action)
    dsr_stress = (stressed_monthly * 12 / profile.annual_income) * 100 if profile.annual_income else 0.0

    feasible = dsr <= dsr_cap_pct
    reason = None if feasible else f"DSR {dsr:.1f}%가 상한 {dsr_cap_pct}% 초과"

    return PortfolioMetrics(
        combo=combo,
        monthly_payment_total=monthly_total,
        dsr_pct=round(dsr, 1),
        npv_total_cost=round(npv_total),
        npv_savings_vs_keep=0.0,  # select_best_portfolio에서 KEEP 기준 대비로 채움
        dsr_stress_pct=round(dsr_stress, 1),
        feasible=feasible,
        infeasible_reason=reason,
    )


# ---------------------------------------------------------------------------
# 3. 유동성 위험 분류 + 우선순위(lexicographic) 선택
# ---------------------------------------------------------------------------

HIGH_VOLATILITY_THRESHOLD = 0.3  # income_volatility 값의 임계치 (데이터 분포 보고 조정 필요)
UNSTABLE_EMPLOYMENT = {"프리랜서", "계약직", "무직", "일용직"}


def classify_liquidity_risk(profile: UserFinancialProfile) -> Literal["고위험", "저위험"]:
    if profile.income_volatility > HIGH_VOLATILITY_THRESHOLD or profile.employment_type in UNSTABLE_EMPLOYMENT:
        return "고위험"
    return "저위험"


def select_best_portfolio(
    loans: List[ExistingLoan],
    refinance_map: dict,
    profile: UserFinancialProfile,
    dsr_cap_pct: float = 40.0,
) -> tuple:
    """전체 파이프라인: 조합 생성 -> 평가 -> 제약필터 -> 우선순위 정렬.

    반환: (선택된 1위 PortfolioMetrics, 비교용 2위 PortfolioMetrics | None, 탈락 사유 리스트)
    """
    combos = generate_portfolio_combinations(loans, refinance_map, profile.liquid_assets_krw)
    evaluated = [evaluate_combo(c, loans, profile, dsr_cap_pct) for c in combos]

    keep_only = next((m for m in evaluated if all(a.action == "KEEP" for a in m.combo)), None)
    baseline_npv = keep_only.npv_total_cost if keep_only else 0.0
    for m in evaluated:
        m.npv_savings_vs_keep = round(baseline_npv - m.npv_total_cost)

    feasible = [m for m in evaluated if m.feasible]
    infeasible = [m for m in evaluated if not m.feasible]

    risk = classify_liquidity_risk(profile)
    if risk == "고위험":
        feasible.sort(key=lambda m: (m.monthly_payment_total, m.npv_total_cost))
    else:
        feasible.sort(key=lambda m: (m.npv_total_cost, m.monthly_payment_total))

    if not feasible:
        return None, None, [m.infeasible_reason for m in infeasible]

    best = feasible[0]
    second = feasible[1] if len(feasible) > 1 else None
    return best, second, [m.infeasible_reason for m in infeasible]


# ---------------------------------------------------------------------------
# 4. eligibility_timeline (자격의 시간축) - A의 '현재 자격'과 대비되는 C의 차별점
# ---------------------------------------------------------------------------

YOUTH_AGE_CAP = 34


@dataclass
class EligibilityTimelineEntry:
    product_or_policy_name: str
    change_type: Literal["자격상실 예정", "신규자격 예상"]
    basis: str            # 근거 (연령상한 / 예측이벤트 등)
    certainty: Literal["확정", "추정"]  # 나이 기반은 확정, 이벤트 예측 기반은 추정


def compute_age_based_timeline(birth_year: int, product_name: str, age_cap: int = YOUTH_AGE_CAP) -> Optional[EligibilityTimelineEntry]:
    """나이 기반 자격상실은 확정적으로 계산 가능 (생년월일은 추론이 아니라 확정정보)."""
    from datetime import datetime
    current_age = datetime.now().year - birth_year
    if current_age >= age_cap:
        return None  # 이미 상실
    years_left = age_cap - current_age
    return EligibilityTimelineEntry(
        product_or_policy_name=product_name,
        change_type="자격상실 예정",
        basis=f"만 {age_cap}세 연령상한까지 약 {years_left}년 남음",
        certainty="확정",
    )


def compute_event_based_timeline(predicted_event: str, evidence_count: int,
                                   policy_or_product_name: str) -> EligibilityTimelineEntry:
    """이벤트 예측 기반 자격변화는 '추정'으로만 표기 (확정 아님 - 반드시 사용자 확인 필요).

    NOTE: 현재 C엔진(reasoning.py)의 PredictionResult는 '언제'에 대한 구체적 시점을
    반환하지 않는다 (evidence_count 기반 confidence만 있음). 그래서 여기서도 정확한
    날짜를 만들어내지 않고 '확정 시' 조건부로만 표기한다 - 근거 없는 날짜를 지어내는 것을 방지.
    """
    return EligibilityTimelineEntry(
        product_or_policy_name=policy_or_product_name,
        change_type="신규자격 예상",
        basis=f"'{predicted_event}' 이벤트 예측 (유사 코호트 {evidence_count}건 근거) 확정 시 자격 발생",
        certainty="추정",
    )


# ---------------------------------------------------------------------------
# 5. 감시조건 (재최적화 트리거) - agent_loop.py의 기존 순환 구조에 얹어서 사용
# ---------------------------------------------------------------------------

def build_watch_conditions(
    best: PortfolioMetrics,
    event_name: str,
    event_confirmed: bool,
    dsr_cap_pct: float = 40.0,
) -> List[str]:
    """추천 실행 후에도 '한 번 추천하고 끝'이 아니라, 상황이 바뀌면 재계산해야 하는
    조건들을 명시적으로 반환한다. agent_loop.py의 콜백이 이 조건들을 등록해두었다가,
    조건 충족 시 select_best_portfolio()를 재호출하는 식으로 쓰면 된다.
    """
    conditions = []
    if not event_confirmed:
        conditions.append(f"'{event_name}' 이벤트가 실제로 확정되면 재계산 필요 (현재는 예측 기반 잠정 추천)")
    if any(a.action == "REFINANCE" for a in best.combo):
        margin = dsr_cap_pct - best.dsr_stress_pct
        if margin < 5.0:
            conditions.append(
                f"스트레스 DSR({best.dsr_stress_pct}%)이 상한({dsr_cap_pct}%)에 근접(여유 {margin:.1f}%p) "
                f"- 금리 추가 상승 시 재검토 필요"
            )
    conditions.append("다음 생애주기 이벤트가 새로 확정될 때마다 포트폴리오 재계산")
    return conditions