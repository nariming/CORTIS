"""
포트폴리오 결정 레이어의 마지막 단계: portfolio.py가 계산한 숫자를 받아
"언제/얼마나/효과/비교/바로할일" 5항목 고정 포맷으로 LLM이 문장을 조립한다.

핵심 원칙 (절대 어기면 안 됨):
  - LLM은 숫자를 단 하나도 새로 만들지 않는다. 전부 portfolio.py가 계산한 값을 그대로 인용만 한다.
  - 그래서 출력 검증가드(validate_summary_numbers)가 반드시 함께 실행되어야 한다.
    검증 실패 시 재생성(MockPortfolioSummarizer/AnthropicPortfolioSummarizer 둘 다 최대 1회 재시도).

reasoning.py의 MockReasoner/AnthropicReasoner 패턴을 그대로 따른다
(LLM_BACKEND=mock|anthropic 환경변수로 전환).

신규대출 경로 추가 (2026.8, Strategy Generator 연결)
  기존 대출을 REFINANCE/PREPAY하는 경우(PortfolioSummaryInput)와, 기존 대출이 아예 없어
  새로 자금을 조달하는 경우(NewLoanSummaryInput)는 "무엇과 비교하는지"의 프레임 자체가 다르다.
  전자는 "지금 상태 유지 대비 얼마나 절감되는가"가 핵심이고, 후자는 "여러 조달 방법 중 뭐가
  최선인가"가 핵심이라 "before/after"나 "절감액" 개념 자체가 없다. 억지로 같은 스키마에
  끼워맞추면 없는 개념(예: npv_savings=0)이 실제 값처럼 보여 오해를 부를 수 있어, 별도
  Input/Summarizer를 두되 출력 스키마(when/how_much/effect/comparison/next_action)와
  검증가드(validate_summary_numbers)는 그대로 재사용한다.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional

from .portfolio import PortfolioMetrics, ExistingLoan, RefinanceCandidate, LoanAction, classify_liquidity_risk, UserFinancialProfile
from .strategy_generator import NewLoanMetrics


SYSTEM_PROMPT = """당신은 청년 대상 부채 포트폴리오 재설계 결과를 설명하는 통역 엔진입니다.

절대 규칙:
1. 당신에게 주어진 [계산된 숫자] 안에 있는 값만 사용하세요. 어떤 숫자도 새로 만들거나
   반올림 이상으로 바꾸지 마세요 (원 단위 정확히, %도 소수점 그대로).
2. 계산되지 않은 정보(예: 정확한 신청 마감일, 확정 여부)를 지어내지 마세요.
3. "comparison" 항목은 [유동성 위험 분류]에 따라 근거가 달라집니다:
   - 저위험(비용우선)이면: 2등이 총비용(NPV)이 얼마나 더 드는지로 설명
   - 고위험(안정우선)이면: 2등이 총비용은 더 적을 수 있지만, 월상환액이 얼마나 더 높아서
     유동성 안정성 때문에 제외했는지로 설명 (총비용 숫자를 숨기지 말고 같이 언급할 것)
4. 반드시 아래 5개 항목을 모두, 이 순서로, 이 JSON 스키마로만 채우세요. 다른 텍스트를 추가하지 마세요.

{
  "when": "언제 실행해야 하는지 (계산된 timing_basis 값을 그대로 서술)",
  "how_much": "얼마나 (대상 대출/금액을 구체적으로)",
  "effect": "효과 (월상환액 변화, 총 NPV 절감액, DSR 변화 - 전부 숫자 그대로)",
  "comparison": "왜 2등 후보가 아닌 이 안인지 (규칙 3번에 따라 숫자로 비교)",
  "next_action": "바로 할 일 (신청 방법/필요서류 - 계산값에 있는 정보만)"
}"""


@dataclass
class PortfolioSummaryInput:
    """LLM에게 넘길 '계산된 숫자' 전부. 이 객체 밖의 숫자는 LLM이 절대 못 만든다."""
    user_name: str
    event_name: str
    best_actions_desc: List[str]        # 예: ["신용대출(2,400만원) -> KB 보증부월세대출로 대환"]
    timing_basis: str                   # 예: "'독립(월세)' 이벤트 확정 시" / "지금 바로 (이벤트 무관)"
    monthly_payment_before: int
    monthly_payment_after: int
    npv_savings: int                    # best.npv_savings_vs_keep
    dsr_before: float
    dsr_after: float
    dsr_stress_after: float
    liquidity_risk: str                 # "고위험" | "저위험" - 어떤 기준으로 골랐는지의 근거
    second_best_desc: Optional[str]
    second_best_npv_diff: Optional[int]      # 2등의 NPV가 1등보다 얼마나 큰지(양수=2등이 더 비쌈, 음수=2등이 더 쌈)
    second_best_monthly_diff: Optional[int]  # 2등의 월상환액이 1등보다 얼마나 큰지(양수=2등이 더 높음)
    next_action_hint: str                # 예: "임대차계약서·재직증명서 지참, KB 앱에서 신청"

    def allowed_numbers(self) -> set:
        """검증가드가 대조할 '진짜 나올 수 있는 숫자' 전체 집합.

        음수는 문장에서 보통 부호 없이 표기되므로(예: "41만원 더 저렴"), 절댓값도 함께 허용한다.

        best_actions_desc/second_best_desc 문자열 안에도 숫자가 박혀 있다(예: 조기상환액,
        대환 대상 대출 잔액 등) — 이 숫자들도 프롬프트에 그대로 노출되므로 LLM이 그대로
        인용하면 정당한 인용이지 지어낸 숫자가 아니다. 이 목록에서 빠뜨리면(과거에 실제로
        빠뜨려져 있었음) LLM이 프롬프트에 있는 숫자를 그대로 썼는데도 검증에 걸려버린다.
        """
        raw = [
            self.monthly_payment_before, self.monthly_payment_after,
            abs(self.monthly_payment_before - self.monthly_payment_after),
            self.npv_savings, self.dsr_before, self.dsr_after, self.dsr_stress_after,
        ]
        if self.second_best_npv_diff is not None:
            raw.append(self.second_best_npv_diff)
        if self.second_best_monthly_diff is not None:
            raw.append(self.second_best_monthly_diff)
        numbers = set()
        for n in raw:
            if isinstance(n, float):
                numbers.add(str(n))
                numbers.add(str(int(n)))
            else:
                numbers.add(str(n))
                numbers.add(str(abs(n)))  # 부호 없는 표기 허용
                if abs(n) >= 10000:
                    numbers.add(str(round(abs(n) / 10000)))  # 만원 단위 축약 표기 허용

        # best_actions_desc/second_best_desc 문자열 속 숫자(조기상환액, 대환 대상 대출 잔액 등)도
        # 프롬프트에 그대로 노출되는 값이므로 허용 목록에 포함한다.
        desc_text = " ".join(self.best_actions_desc)
        if self.second_best_desc:
            desc_text += " " + self.second_best_desc
        for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", desc_text):
            cleaned = m.replace(",", "")
            numbers.add(cleaned)
            if cleaned.isdigit() and int(cleaned) >= 10000:
                numbers.add(str(round(int(cleaned) / 10000)))  # 만원 단위 축약 표기도 허용

        return numbers


def build_summary_prompt(data: PortfolioSummaryInput) -> str:
    return f"""[유저] {data.user_name}
[예측/확정 이벤트] {data.event_name}
[유동성 위험 분류] {data.liquidity_risk}

[계산된 숫자 - 이것만 사용할 것]
- 추천 액션: {', '.join(data.best_actions_desc)}
- 실행 시점 근거: {data.timing_basis}
- 월상환액: {data.monthly_payment_before:,}원 -> {data.monthly_payment_after:,}원
- 총 NPV 절감액: {data.npv_savings:,}원
- DSR: {data.dsr_before}% -> {data.dsr_after}% (스트레스 금리 적용 시 {data.dsr_stress_after}%)
- 2등 후보: {data.second_best_desc or '없음'}
  - 2등의 NPV 총비용 차이(1등 대비): {data.second_best_npv_diff if data.second_best_npv_diff is not None else '해당없음'}원 (양수=2등이 더 비쌈)
  - 2등의 월상환액 차이(1등 대비): {data.second_best_monthly_diff if data.second_best_monthly_diff is not None else '해당없음'}원 (양수=2등이 더 높음)
- 다음 행동 정보: {data.next_action_hint}

위 숫자만 사용해서, [유동성 위험 분류]에 맞는 근거로 5개 항목 JSON으로 응답하세요."""


class PortfolioSummarizer:
    def summarize(self, data: PortfolioSummaryInput) -> dict:
        raise NotImplementedError


class MockPortfolioSummarizer(PortfolioSummarizer):
    """API 키 없이 배선 검증용. 실제 LLM 없이 템플릿에 값만 그대로 꽂아 넣는다."""

    def summarize(self, data: PortfolioSummaryInput) -> dict:
        monthly_diff = data.monthly_payment_before - data.monthly_payment_after
        if not data.second_best_desc:
            comparison = "다른 실행가능한 대안 없음"
        elif data.liquidity_risk == "고위험":
            # 고위험: 월상환액 우선이라, 2등은 월상환액이 더 높아서 제외 (총비용은 더 쌀 수도 있음)
            comparison = (
                f"2등({data.second_best_desc})은 월상환액이 {abs(data.second_best_monthly_diff):,}원 더 높아 "
                f"유동성 안정성 기준으로 제외 "
                f"(총비용은 2등이 {abs(data.second_best_npv_diff):,}원 "
                f"{'더 적지만' if data.second_best_npv_diff < 0 else '더 들어서'} 이번엔 안정성을 우선함)"
            )
        else:
            # 저위험: 총비용 우선이라, 2등은 총비용이 더 들어서 제외
            comparison = f"2등({data.second_best_desc})은 총비용이 {abs(data.second_best_npv_diff):,}원 더 들어 제외"

        return {
            "when": f"[MOCK] {data.timing_basis}",
            "how_much": f"[MOCK] {', '.join(data.best_actions_desc)}",
            "effect": (
                f"[MOCK] 월상환액 {data.monthly_payment_before:,}원 -> {data.monthly_payment_after:,}원 "
                f"({'-' if monthly_diff >= 0 else '+'}{abs(monthly_diff):,}원), 총 {data.npv_savings:,}원 절감(NPV), "
                f"DSR {data.dsr_before}%->{data.dsr_after}% (스트레스 적용 시 {data.dsr_stress_after}%)"
            ),
            "comparison": f"[MOCK] {comparison}",
            "next_action": f"[MOCK] {data.next_action_hint}",
        }


class AnthropicPortfolioSummarizer(PortfolioSummarizer):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise RuntimeError("anthropic 패키지가 필요합니다: pip install anthropic --break-system-packages")

    def summarize(self, data: PortfolioSummaryInput) -> dict:
        prompt = build_summary_prompt(data)
        for attempt in range(2):  # 검증 실패 시 1회 재시도
            resp = self._client.messages.create(
                model=self.model, max_tokens=1000, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            ok, bad_numbers = validate_summary_numbers(result, data.allowed_numbers())
            if ok:
                return result
            prompt += f"\n\n[재생성 요청] 이전 응답에 허용되지 않은 숫자({bad_numbers})가 있었습니다. 계산된 숫자만 다시 사용하세요."
        raise ValueError(f"LLM이 검증 통과하는 응답을 2회 시도에도 생성하지 못함: {bad_numbers}")


# ---------------------------------------------------------------------------
# 검증 가드 - "묻지마 LLM 아님"을 코드로 증명하는 핵심 장치
# ---------------------------------------------------------------------------

def validate_summary_numbers(result: dict, allowed_numbers: set) -> tuple:
    """LLM 출력(JSON) 안의 모든 숫자를 뽑아서, 허용된 숫자 집합에 있는지 대조.

    반환: (통과 여부, 위반한 숫자 리스트)
    콤마(1,870,000)나 % 기호가 붙어도 정규식이 알아서 순수 숫자만 뽑아낸다.
    """
    full_text = " ".join(str(v) for v in result.values())
    numbers_found = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", full_text)
    normalized = [n.replace(",", "") for n in numbers_found]

    bad = []
    for n in normalized:
        if n in allowed_numbers:
            continue
        # 정수 <-> 소수 표기 차이(38 vs 38.0)는 허용
        try:
            if str(float(n)) in allowed_numbers or str(int(float(n))) in allowed_numbers:
                continue
        except ValueError:
            pass
        # 아주 작은 숫자(1등/2등처럼 서수 표현 등)는 오탐 방지로 제외
        if n in {"1", "2", "3"}:
            continue
        bad.append(n)

    return (len(bad) == 0, bad)


def get_summarizer() -> PortfolioSummarizer:
    backend = os.environ.get("LLM_BACKEND", "mock")
    if backend == "anthropic":
        return AnthropicPortfolioSummarizer()
    return MockPortfolioSummarizer()


# ---------------------------------------------------------------------------
# portfolio.py의 PortfolioMetrics -> PortfolioSummaryInput 변환 헬퍼
# ---------------------------------------------------------------------------

def build_summary_input(
    user_name: str,
    event_name: str,
    loans: List[ExistingLoan],
    profile: UserFinancialProfile,
    best: PortfolioMetrics,
    second: Optional[PortfolioMetrics],
    dsr_before: float,
    next_action_hint: str,
) -> PortfolioSummaryInput:
    """agent_loop.py에서 select_best_portfolio() 결과를 받아 바로 이 함수에 넘기면 됨."""

    def describe_combo(metrics: PortfolioMetrics) -> str:
        parts = []
        for a in metrics.combo:
            if a.action == "KEEP":
                continue
            if a.action == "PREPAY":
                parts.append(f"{a.loan_id} {a.prepay_amount:,}원 조기상환" if a.prepay_amount else f"{a.loan_id} 전액상환")
            elif a.action == "REFINANCE" and a.target_product:
                parts.append(f"{a.loan_id} -> {a.target_product.product_name} 대환")
        return ", ".join(parts) if parts else "기존 대출 유지"

    has_event_dependent_action = any(
        a.action == "REFINANCE" for a in best.combo
    )
    timing_basis = f"'{event_name}' 이벤트 확정 시" if has_event_dependent_action else "지금 바로 (이벤트 확정 여부 무관)"

    return PortfolioSummaryInput(
        user_name=user_name,
        event_name=event_name,
        best_actions_desc=[describe_combo(best)],
        timing_basis=timing_basis,
        monthly_payment_before=sum(l.monthly_payment for l in loans),
        monthly_payment_after=best.monthly_payment_total,
        npv_savings=best.npv_savings_vs_keep,
        dsr_before=dsr_before,
        dsr_after=best.dsr_pct,
        dsr_stress_after=best.dsr_stress_pct,
        liquidity_risk=classify_liquidity_risk(profile),
        second_best_desc=describe_combo(second) if second else None,
        second_best_npv_diff=(round(second.npv_total_cost - best.npv_total_cost) if second else None),
        second_best_monthly_diff=(second.monthly_payment_total - best.monthly_payment_total if second else None),
        next_action_hint=next_action_hint,
    )


# ---------------------------------------------------------------------------
# 신규대출 경로 (기존 대출 없음 - Strategy Generator 연결)
# ---------------------------------------------------------------------------

NEW_LOAN_SYSTEM_PROMPT = """당신은 청년의 신규 자금조달 전략을 설명하는 통역 엔진입니다.
(이 유저는 기존 대출이 없어 새로 자금이 필요해진 상황입니다 - "대환/절감"이 아니라
"여러 조달 방법 중 무엇이 최선인지"를 설명하는 것이 핵심입니다.)

절대 규칙:
1. 당신에게 주어진 [계산된 숫자] 안에 있는 값만 사용하세요. 새로 만들거나 반올림 이상으로
   바꾸지 마세요 (원 단위 정확히, %도 소수점 그대로).
2. 계산되지 않은 정보(예: 정확한 신청 마감일)를 지어내지 마세요.
3. "before/after"나 "절감액" 표현은 쓰지 마세요 - 기존 대출이 없어서 비교할 이전 상태 자체가
   없습니다. 대신 "이 전략을 실행하면"이라는 프레임으로 서술하세요.
4. 반드시 아래 5개 항목을 모두, 이 순서로, 이 JSON 스키마로만 채우세요. 다른 텍스트를
   추가하지 마세요.

{
  "when": "언제 실행해야 하는지 (계산된 timing_basis 값을 그대로 서술)",
  "how_much": "얼마나 (조달 방법과 금액을 구체적으로)",
  "effect": "효과 (월상환액, 총비용(NPV), DSR - 전부 숫자 그대로)",
  "comparison": "왜 2등 후보가 아닌 이 안인지 (총비용/월상환액 차이로 숫자 근거)",
  "next_action": "바로 할 일 (신청 방법/필요서류 - 계산값에 있는 정보만)"
}"""


@dataclass
class NewLoanSummaryInput:
    """PortfolioSummaryInput의 신규대출 버전. 필드가 다른 이유는 클래스 상단 docstring 참고."""
    user_name: str
    event_name: str
    action_desc: str
    timing_basis: str
    monthly_payment: int
    npv_total_cost: float
    dsr_pct: float
    second_best_desc: Optional[str]
    second_best_npv_diff: Optional[float]     # 양수 = 2등이 더 비쌈
    second_best_monthly_diff: Optional[int]   # 양수 = 2등이 더 높음
    next_action_hint: str

    def allowed_numbers(self) -> set:
        raw = [self.monthly_payment, self.npv_total_cost, self.dsr_pct]
        if self.second_best_npv_diff is not None:
            raw.append(self.second_best_npv_diff)
        if self.second_best_monthly_diff is not None:
            raw.append(self.second_best_monthly_diff)
        numbers = set()
        for n in raw:
            if isinstance(n, float):
                numbers.add(str(n))
                numbers.add(str(int(n)))
            else:
                numbers.add(str(n))
                numbers.add(str(abs(n)))
                if abs(n) >= 10000:
                    numbers.add(str(round(abs(n) / 10000)))

        desc_text = self.action_desc + " " + (self.second_best_desc or "")
        for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", desc_text):
            cleaned = m.replace(",", "")
            numbers.add(cleaned)
            if cleaned.isdigit() and int(cleaned) >= 10000:
                numbers.add(str(round(int(cleaned) / 10000)))
        return numbers


def build_new_loan_summary_prompt(data: NewLoanSummaryInput) -> str:
    return f"""[유저] {data.user_name}
[확정 이벤트] {data.event_name}

[계산된 숫자 - 이것만 사용할 것]
- 추천 전략: {data.action_desc}
- 실행 시점 근거: {data.timing_basis}
- 월상환액: {data.monthly_payment:,}원
- 총비용(NPV): {data.npv_total_cost:,.0f}원
- DSR: {data.dsr_pct}%
- 2등 후보: {data.second_best_desc or '없음'}
  - 2등과의 총비용 차이: {f'{data.second_best_npv_diff:,.0f}원' if data.second_best_npv_diff is not None else '해당없음'} (양수=2등이 더 비쌈)
  - 2등과의 월상환액 차이: {f'{data.second_best_monthly_diff:,}원' if data.second_best_monthly_diff is not None else '해당없음'} (양수=2등이 더 높음)
- 다음 행동 정보: {data.next_action_hint}

위 숫자만 사용해서 5개 항목 JSON으로 응답하세요."""


class MockNewLoanSummarizer:
    def summarize(self, data: NewLoanSummaryInput) -> dict:
        if data.second_best_desc and data.second_best_npv_diff is not None:
            comparison = f"2등({data.second_best_desc})은 총비용이 {abs(data.second_best_npv_diff):,.0f}원 더 들어 제외"
        else:
            comparison = "다른 실행가능한 대안 없음"
        return {
            "when": f"[MOCK] {data.timing_basis}",
            "how_much": f"[MOCK] {data.action_desc}",
            "effect": (
                f"[MOCK] 월상환액 {data.monthly_payment:,}원, "
                f"총비용(NPV) {data.npv_total_cost:,.0f}원, DSR {data.dsr_pct}%"
            ),
            "comparison": f"[MOCK] {comparison}",
            "next_action": f"[MOCK] {data.next_action_hint}",
        }


class AnthropicNewLoanSummarizer:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise RuntimeError("anthropic 패키지가 필요합니다: pip install anthropic --break-system-packages")

    def summarize(self, data: NewLoanSummaryInput) -> dict:
        prompt = build_new_loan_summary_prompt(data)
        for attempt in range(2):
            resp = self._client.messages.create(
                model=self.model, max_tokens=1000, system=NEW_LOAN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            ok, bad_numbers = validate_summary_numbers(result, data.allowed_numbers())
            if ok:
                return result
            prompt += f"\n\n[재생성 요청] 이전 응답에 허용되지 않은 숫자({bad_numbers})가 있었습니다. 계산된 숫자만 다시 사용하세요."
        raise ValueError(f"LLM이 검증 통과하는 응답을 2회 시도에도 생성하지 못함: {bad_numbers}")


def get_new_loan_summarizer():
    backend = os.environ.get("LLM_BACKEND", "mock")
    if backend == "anthropic":
        return AnthropicNewLoanSummarizer()
    return MockNewLoanSummarizer()


def build_new_loan_summary_input(
    user_name: str,
    event_name: str,
    best: NewLoanMetrics,
    second: Optional[NewLoanMetrics],
    next_action_hint: str,
) -> NewLoanSummaryInput:
    """agent_loop.py에서 select_best_new_loan_strategy() 결과를 받아 바로 이 함수에 넘기면 됨."""
    return NewLoanSummaryInput(
        user_name=user_name,
        event_name=event_name,
        action_desc=best.candidate.label,
        timing_basis=f"'{event_name}' 이벤트 확정 시",
        monthly_payment=best.monthly_payment,
        npv_total_cost=best.npv_total_cost,
        dsr_pct=best.dsr_pct,
        second_best_desc=second.candidate.label if second else None,
        second_best_npv_diff=(round(second.npv_total_cost - best.npv_total_cost) if second else None),
        second_best_monthly_diff=(second.monthly_payment - best.monthly_payment if second else None),
        next_action_hint=next_action_hint,
    )