"""
A파트 ②: 정책 자격 매칭 (규칙기반).

C엔진이 "다음에 독립(월세)가 유력함"이라고 예측하면, 그 이벤트를 트리거로
"그럼 지금 새로 자격이 생기는 정책이 뭔가"를 결정론적으로 판정한다.

RAG를 안 쓰는 이유
  정책 자격요건(나이/소득/거주형태/지역)은 원래 정형 데이터라서, 벡터 검색으로 근사하면
  오히려 "자격 없는데 있다고 말하는" 사고가 난다. 자격 판정은 SQL/비교연산으로 결정론적으로 하고,
  AI다움은 C엔진(무슨 이벤트가 올지 예측)이 담당하는 게 역할 분담상 맞다.
  (RAG 기반 정책 문서 검색은 '설명 보강' 용도의 보조 기능으로만 남긴다)
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from sqlalchemy.orm import Session

from backend.db.models import LoanProduct, Policy, User

# 이벤트가 실제로 일어나면 프로필의 어떤 값이 바뀌는가.
#
# 이게 이 서비스의 핵심 동작이다. C엔진이 "다음은 독립(월세)가 유력"이라고 예측했을 때,
# 현재 프로필(부모동거)로 자격을 따지면 월세 지원 정책은 전부 '자격 없음'으로 떨어진다.
# 하지만 사용자가 알고 싶은 건 "독립하면 뭘 받을 수 있나"이므로,
# 예측 모드에서는 이벤트가 일어난 뒤의 프로필을 가정하고 판정한다 (= 사전 안내).
EVENT_IMPLIED_PROFILE = {
    "독립(월세)": {"housing_type": "월세"},
    "독립(전세)": {"housing_type": "전세"},
    "내집마련": {"housing_type": "자가"},
    "결혼": {"marital_status": "기혼"},
    "퇴직": {"employment_type": "무직", "monthly_income_avg": 0},
    "창업": {"employment_type": "자영업"},
    "대학생": {"employment_type": "학생"},
}


class ProjectedUser:
    """이벤트 발생 후를 가정한 유저 뷰 (DB는 건드리지 않는 읽기 전용 오버레이)."""

    def __init__(self, user: User, overrides: dict):
        self._user = user
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._user, name)

    @property
    def annual_income(self) -> int:
        return self.monthly_income_avg * 12


@dataclass
class PolicyMatch:
    policy: Policy
    status: str                      # newly_eligible | eligible | not_eligible
    passed: List[str]                # 충족한 조건 설명
    failed: List[str]                # 불충족 조건 설명
    assumption: Optional[str] = None  # 예측 모드에서 어떤 상태 변화를 가정했는지

    @property
    def reason(self) -> str:
        core = (
            "미충족: " + ", ".join(self.failed)
            if self.failed
            else ("충족: " + ", ".join(self.passed) if self.passed else "별도 자격요건 없음")
        )
        return f"[{self.assumption}] {core}" if self.assumption else core

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy.policy_id,
            "policy_name": self.policy.policy_name,
            "provider": self.policy.provider,
            "category": self.policy.category,
            "benefit_summary": self.policy.benefit_summary,
            "apply_url": self.policy.apply_url,
            "status": self.status,
            "reason": self.reason,
            "assumption": self.assumption,
            "passed": self.passed,
            "failed": self.failed,
        }


def _check_eligibility(user: User, item: Union[Policy, LoanProduct]) -> tuple:
    """정형 자격요건(min_age~region_code)을 하나씩 비교해 (passed, failed) 를 돌려준다.

    Policy와 LoanProduct는 이 여섯 컬럼 이름이 완전히 동일하므로(schema.sql 참고)
    이 함수 하나로 둘 다 판정한다. NULL 컬럼은 '조건 없음'으로 통과 처리.
    """
    passed: List[str] = []
    failed: List[str] = []
    age = user.age

    if item.min_age is not None or item.max_age is not None:
        lo = item.min_age if item.min_age is not None else 0
        hi = item.max_age if item.max_age is not None else 200
        (passed if lo <= age <= hi else failed).append(f"연령 {age}세 (요건 {lo}~{hi}세)")

    if item.max_annual_income is not None:
        income = user.annual_income
        ok = income <= item.max_annual_income
        (passed if ok else failed).append(
            f"연소득 {income:,}원 (상한 {item.max_annual_income:,}원)"
        )

    if item.allowed_employment:
        ok = user.employment_type in item.allowed_employment
        (passed if ok else failed).append(
            f"고용형태 {user.employment_type} (허용 {'/'.join(item.allowed_employment)})"
        )

    if item.allowed_housing:
        ok = user.housing_type in item.allowed_housing
        (passed if ok else failed).append(
            f"거주형태 {user.housing_type} (허용 {'/'.join(item.allowed_housing)})"
        )

    if item.allowed_marital:
        ok = user.marital_status in item.allowed_marital
        (passed if ok else failed).append(
            f"혼인상태 {user.marital_status} (허용 {'/'.join(item.allowed_marital)})"
        )

    if item.region_code:
        ok = user.region_code == item.region_code
        (passed if ok else failed).append(f"거주지역 코드 {user.region_code} (요건 {item.region_code})")

    return passed, failed


def _check(user: User, policy: Policy) -> PolicyMatch:
    passed, failed = _check_eligibility(user, policy)
    return PolicyMatch(
        policy=policy,
        status="not_eligible" if failed else "eligible",
        passed=passed,
        failed=failed,
    )


def match_for_event(
    db: Session,
    user: User,
    event_type: str,
    include_ineligible: bool = False,
    prospective: bool = True,
) -> List[PolicyMatch]:
    """특정 이벤트를 트리거로 자격을 재판단한다.

    trigger_events 에 해당 이벤트가 들어 있는 정책만 후보로 좁힌 뒤 조건을 검사하므로,
    "왜 이 정책이 지금 뜨는가"에 대한 답이 '이 이벤트가 트리거라서'로 명확하다.

    prospective=True (기본): 아직 일어나지 않은 예측 이벤트에 대한 사전 안내 모드.
      EVENT_IMPLIED_PROFILE 로 이벤트 발생 후 프로필을 가정해 판정한다.
    prospective=False: 이미 확정된 이벤트에 대해 현재 프로필 그대로 판정 (사후 확인용).
    """
    candidates = [p for p in db.query(Policy).all() if event_type in (p.trigger_events or [])]

    overrides = EVENT_IMPLIED_PROFILE.get(event_type, {}) if prospective else {}
    subject = ProjectedUser(user, overrides) if overrides else user

    results = [_check(subject, p) for p in candidates]
    for r in results:
        # 이벤트로 새로 촉발된 자격이므로, 통과한 건 newly_eligible 로 표시해서 UI에서 강조
        if r.status == "eligible":
            r.status = "newly_eligible"
        if overrides:
            assumed = ", ".join(f"{k}={v}" for k, v in overrides.items())
            r.assumption = f"'{event_type}' 발생 가정 ({assumed})"

    if not include_ineligible:
        results = [r for r in results if r.status != "not_eligible"]

    results.sort(key=lambda r: (r.policy.priority, r.policy.policy_name))
    return results


def match_for_events(
    db: Session,
    user: User,
    event_types: List[str],
    include_ineligible: bool = False,
    prospective: bool = True,
) -> List[dict]:
    """C엔진 예측 결과(여러 후보 이벤트)를 한 번에 받아 처리하는 진입점.

    반환 형태를 이벤트별로 묶어서, 데모 화면에서
    "독립(월세) 예측 → 전월세자금대출 정책 미리 안내" 흐름을 그대로 그릴 수 있게 한다.
    """
    out = []
    for event_type in event_types:
        matches = match_for_event(db, user, event_type, include_ineligible, prospective)
        out.append(
            {
                "basis_event": event_type,
                "matched_count": len(matches),
                "policies": [m.to_dict() for m in matches],
            }
        )
    return out


def current_eligibility(db: Session, user: User, category: Optional[str] = None) -> List[PolicyMatch]:
    """이벤트와 무관하게 '지금 이 유저가 받을 수 있는 정책' 전수 조회 (기본 화면용)."""
    query = db.query(Policy)
    if category:
        query = query.filter(Policy.category == category)
    results = [_check(user, p) for p in query.all()]
    results = [r for r in results if r.status == "eligible"]
    results.sort(key=lambda r: (r.policy.priority, r.policy.policy_name))
    return results


# =========================================================================
# LoanProduct 매칭 — KB 자체 대출상품 버전.
#
# Policy와 판정 로직(_check_eligibility)은 공유하되, 결과 형태는 다르다.
# Policy에는 없는 min_rate/max_rate/max_amount를 그대로 넘겨줘야
# B파트가 유저의 기존 대출(user_loans.interest_rate)과 비교해서
# "이 상품으로 갈아타면 절감되는 금액"을 계산할 수 있다.
# priority 컬럼이 LoanProduct엔 없어서 정렬은 이름순으로 한다.
# =========================================================================


@dataclass
class LoanProductMatch:
    product: LoanProduct
    status: str                       # newly_eligible | eligible | not_eligible
    passed: List[str]
    failed: List[str]
    assumption: Optional[str] = None

    @property
    def reason(self) -> str:
        core = (
            "미충족: " + ", ".join(self.failed)
            if self.failed
            else ("충족: " + ", ".join(self.passed) if self.passed else "별도 자격요건 없음")
        )
        return f"[{self.assumption}] {core}" if self.assumption else core

    def to_dict(self) -> dict:
        return {
            "product_id": self.product.product_id,
            "product_name": self.product.product_name,
            "product_type": self.product.product_type,
            "min_rate": float(self.product.min_rate),
            "max_rate": float(self.product.max_rate),
            "max_amount": self.product.max_amount,
            "target_desc": self.product.target_desc,
            "status": self.status,
            "reason": self.reason,
            "assumption": self.assumption,
            "passed": self.passed,
            "failed": self.failed,
        }


def _check_loan(user: User, product: LoanProduct) -> LoanProductMatch:
    passed, failed = _check_eligibility(user, product)
    return LoanProductMatch(
        product=product,
        status="not_eligible" if failed else "eligible",
        passed=passed,
        failed=failed,
    )


def match_for_loan_event(
    db: Session,
    user: User,
    event_type: str,
    include_ineligible: bool = False,
    prospective: bool = True,
) -> List[LoanProductMatch]:
    """match_for_event()의 LoanProduct 버전. 이벤트 트리거 → 자격 있는 KB 대출상품.

    prospective=True면 Policy와 동일하게 EVENT_IMPLIED_PROFILE로 이벤트 발생 후
    프로필을 가정한다 (예: 독립(월세) 예측 시 월세 대출상품이 자격 판정에 걸리도록).
    """
    candidates = [
        p for p in db.query(LoanProduct).all() if event_type in (p.trigger_events or [])
    ]

    overrides = EVENT_IMPLIED_PROFILE.get(event_type, {}) if prospective else {}
    subject = ProjectedUser(user, overrides) if overrides else user

    results = [_check_loan(subject, p) for p in candidates]
    for r in results:
        if r.status == "eligible":
            r.status = "newly_eligible"
        if overrides:
            assumed = ", ".join(f"{k}={v}" for k, v in overrides.items())
            r.assumption = f"'{event_type}' 발생 가정 ({assumed})"

    if not include_ineligible:
        results = [r for r in results if r.status != "not_eligible"]

    results.sort(key=lambda r: (r.product.min_rate, r.product.product_name))
    return results


def match_for_loan_events(
    db: Session,
    user: User,
    event_types: List[str],
    include_ineligible: bool = False,
    prospective: bool = True,
) -> List[dict]:
    """match_for_events()의 LoanProduct 버전. C엔진 예측 결과를 그대로 받는 진입점."""
    out = []
    for event_type in event_types:
        matches = match_for_loan_event(db, user, event_type, include_ineligible, prospective)
        out.append(
            {
                "basis_event": event_type,
                "matched_count": len(matches),
                "loan_products": [m.to_dict() for m in matches],
            }
        )
    return out


def current_loan_eligibility(
    db: Session, user: User, product_type: Optional[str] = None
) -> List[LoanProductMatch]:
    """이벤트와 무관하게 '지금 자격이 되는 KB 대출상품' 전수 조회."""
    query = db.query(LoanProduct)
    if product_type:
        query = query.filter(LoanProduct.product_type == product_type)
    results = [_check_loan(user, p) for p in query.all()]
    results = [r for r in results if r.status == "eligible"]
    results.sort(key=lambda r: (r.product.min_rate, r.product.product_name))
    return results
