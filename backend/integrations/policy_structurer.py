"""
정책 원문 자유텍스트 -> 구조화 필드 변환 (LLM 기반).

배경
  youthcenter_mapper.py의 기존 방식은 정규식(AMOUNT_PATTERNS) + 고정 키워드 리스트
  (EVENT_KEYWORD_MAP, CATEGORY_KEYWORD_MAP)로 지원금액/생애주기이벤트/카테고리를 뽑았다.
  이 방식은 정확히 미리 정의한 문장 패턴·단어에만 반응하므로, 기관마다 다른 공고문
  표현(예: "월세"가 아니라 "자취", "1인가구 주거비")을 새 기관이 추가될 때마다 사람이
  손으로 계속 규칙을 늘려야 확장 가능하다는 근본적 한계가 있다.

이 파일이 하는 일
  정책 원문(정책명+설명+지원내용+키워드+대분류를 합친 자유텍스트)을 LLM에게 그대로 읽혀서,
  아래 스키마로 구조화된 값을 뽑아낸다. 새 기관/새 표현이 와도 코드 수정 없이 처리된다.

절대 원칙 (financial 정보이므로 반드시 지킬 것)
  - LLM은 원문에 명시되지 않은 숫자를 만들어내지 않는다. 애매하면 null.
  - 이 파일은 숫자를 "추출"만 한다 — 자격 판정(누가 진짜 자격이 되는지)은 여전히
    policy_matcher.py가 SQL 비교로 결정론적으로 수행한다. 여기서 바뀌는 건 없다.
  - LLM이 뽑은 금액은 원문에 실제로 등장하는 숫자인지 코드가 재검증한다
    (validate_extracted_amount) — portfolio_summary.py의 validate_summary_numbers와
    동일한 안전장치 패턴.

이벤트/카테고리는 우리 도메인의 고정 taxonomy 안에서만 고르게 강제한다 (자유생성 금지) —
A파트의 trigger_events 매칭 로직이 정확히 이 문자열들과 일치해야 작동하기 때문이다.
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

EVENT_TAXONOMY = [
    "취업", "이직", "퇴직", "독립(월세)", "독립(전세)",
    "내집마련", "결혼", "출산", "창업",
]
CATEGORY_TAXONOMY = ["주거", "고용", "창업", "결혼출산", "교육", "금융"]
EMPLOYMENT_TAXONOMY = ["정규직", "계약직", "프리랜서", "일용직", "무직", "자영업", "학생"]
HOUSING_TAXONOMY = ["자가", "전세", "월세", "부모동거", "기타"]


@dataclass
class StructuredPolicyFields:
    benefit_amount_krw: Optional[int]
    benefit_period_month: Optional[int]
    benefit_rate_pct: Optional[float]
    allowed_employment: Optional[List[str]]   # None = 무관(제한 없음)
    allowed_housing: Optional[List[str]]
    trigger_events: List[str] = field(default_factory=list)
    category: str = "금융"
    extraction_notes: str = ""                # 뭘 못 찾았는지, 왜 null인지 (디버깅/검수용)
    verified: bool = True                      # 코드 재검증 통과 여부 (금액이 원문에 실제 있었는지)


SYSTEM_PROMPT = f"""당신은 청년 정책 공고문에서 사실만 정확히 추출하는 엔진입니다.

절대 규칙
- 원문에 명시적으로 나오지 않은 숫자를 절대 만들어내지 마세요. 애매하거나 안 나와 있으면
  반드시 null로 응답하세요. "대략 이 정도일 것이다" 같은 추측은 금지입니다.
- trigger_events는 반드시 아래 목록 중에서만 고르세요 (목록에 없으면 절대 새로 만들지 말 것):
  {EVENT_TAXONOMY}
  이 공고가 해당 이벤트를 "겪을 예정이거나 겪은" 사람을 대상으로 할 때만 포함하세요.
  해당하는 이벤트가 하나도 없으면 빈 리스트 [].
- category는 반드시 아래 중 하나: {CATEGORY_TAXONOMY}
- allowed_employment는 반드시 아래 목록의 부분집합이거나 null(고용형태 제한 없음): {EMPLOYMENT_TAXONOMY}
- allowed_housing은 반드시 아래 목록의 부분집합이거나 null(거주형태 제한 없음): {HOUSING_TAXONOMY}
- 지원금액(benefit_amount_krw)은 "총 지급액" 기준 원 단위 정수로 계산하세요
  (예: "월 20만원 x 12개월"이면 2400000). 원문에 계산 근거가 없으면 null.
- benefit_period_month는 지원이 지속되는 개월 수. 없으면 null.
- benefit_rate_pct는 금리 우대/지원 정책일 때만, 원문에 숫자가 있을 때만. 없으면 null.
- extraction_notes에는 "지원금액을 특정 못 함 - 원문에 구체적 숫자 없음" 처럼 무엇을
  근거로 null 처리했는지 한 줄로 남기세요.

반드시 아래 JSON 스키마로만 응답하세요. 다른 텍스트를 추가하지 마세요.
{{
  "benefit_amount_krw": 숫자 | null,
  "benefit_period_month": 숫자 | null,
  "benefit_rate_pct": 숫자 | null,
  "allowed_employment": ["정규직", ...] | null,
  "allowed_housing": ["월세", ...] | null,
  "trigger_events": ["독립(월세)", ...],
  "category": "주거",
  "extraction_notes": "설명"
}}"""


def build_user_prompt(combined_text: str) -> str:
    return f"""[정책 공고 원문 (정책명+설명+지원내용+키워드+대분류를 이어붙인 텍스트)]
{combined_text}

위 원문만 근거로 구조화하세요. 원문에 없는 정보는 만들지 말고 null/[]/무관으로 처리하세요."""


def _extract_digits(text: Optional[str]) -> set:
    """숫자만 뽑아서 비교용 집합으로. '2,400,000원'과 '2400000'을 같은 걸로 취급하기 위함."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text.replace(",", "")))


def validate_extracted_amount(amount: Optional[int], source_text: str) -> bool:
    """LLM이 뽑은 금액이 원문에 실제로 등장하는 숫자 조합으로 설명 가능한지 대략 검증한다.

    완벽한 검증은 아니다(예: '월 20만원 x 12개월'->2400000처럼 계산이 들어간 값은
    원문에 그 숫자가 그대로 안 나옴). 그래서 '원문 안의 숫자들로 곱/합이 설명되는지'까지
    느슨하게 확인한다. 그래도 설명 안 되면 verified=False로 표시해 검수 대상으로 남긴다
    (자동 폐기하지 않고, 화면에 '검수 필요' 표시로 노출하는 걸 권장 — 데이터 손실 방지).
    """
    if amount is None:
        return True
    digits_in_text = [int(d) for d in _extract_digits(source_text) if d]
    if amount in digits_in_text:
        return True
    if any(amount == d * 10_000 for d in digits_in_text):
        return True
    for a in digits_in_text:
        for b in digits_in_text:
            if a != b and amount == a * 10_000 * b:
                return True
    return False


class PolicyStructurer:
    def structure(self, combined_text: str) -> StructuredPolicyFields:
        raise NotImplementedError


class RegexPolicyStructurer(PolicyStructurer):
    """API 키 없이도 배선을 검증하기 위한 폴백. 기존 정규식/키워드 로직 그대로 보존.

    주의: 이 클래스는 '규칙기반으로 하면 이 정도 수준'이라는 대조군으로도 의미가 있다.
    발표용 최종 데모에는 반드시 AnthropicPolicyStructurer를 써야 한다.
    """

    AMOUNT_PATTERNS = [
        re.compile(r"월\s?(\d{1,3})\s?만\s?원.{0,10}?(\d{1,2})\s?개월"),
        re.compile(r"연\s?(\d{1,4})\s?만\s?원"),
        re.compile(r"최대\s?(\d{1,4})\s?만\s?원"),
    ]
    EVENT_KEYWORD_MAP = {
        "취업": ["취업", "채용", "구직", "일자리", "인턴", "고용"],
        "이직": ["이직"],
        "퇴직": ["퇴직", "실직", "구직촉진", "실업"],
        "독립(월세)": ["월세"],
        "독립(전세)": ["전세", "보증금"],
        "내집마련": ["내집마련", "주택구입", "매매", "디딤돌"],
        "결혼": ["결혼", "신혼"],
        "출산": ["출산", "육아", "임신", "양육"],
        "창업": ["창업", "사업화", "예비창업"],
    }
    CATEGORY_KEYWORD_MAP = {
        "주거": ["주거", "주택", "전세", "월세", "임대"],
        "고용": ["일자리", "취업", "고용", "구직"],
        "창업": ["창업"],
        "결혼출산": ["결혼", "출산", "육아", "임신"],
        "교육": ["교육", "훈련", "역량"],
    }

    def _keyword_hit(self, text: str, keyword_map: dict) -> List[str]:
        return [label for label, kws in keyword_map.items() if any(kw in text for kw in kws)]

    def structure(self, combined_text: str) -> StructuredPolicyFields:
        cleaned = combined_text.replace(",", "")
        amount, period = None, None
        m = self.AMOUNT_PATTERNS[0].search(cleaned)
        if m:
            monthly_man, months = int(m.group(1)), int(m.group(2))
            amount, period = monthly_man * 10_000 * months, months
        else:
            for pattern in self.AMOUNT_PATTERNS[1:]:
                m = pattern.search(cleaned)
                if m:
                    amount = int(m.group(1)) * 10_000
                    break

        return StructuredPolicyFields(
            benefit_amount_krw=amount,
            benefit_period_month=period,
            benefit_rate_pct=None,
            allowed_employment=None,
            allowed_housing=None,
            trigger_events=self._keyword_hit(combined_text, self.EVENT_KEYWORD_MAP),
            category=(self._keyword_hit(combined_text, self.CATEGORY_KEYWORD_MAP) or ["금융"])[0],
            extraction_notes="[규칙기반] 정규식/키워드 매칭만 사용 — 등록된 패턴 밖 표현은 놓칠 수 있음",
            verified=True,
        )


class AnthropicPolicyStructurer(PolicyStructurer):
    """실서비스용. ANTHROPIC_API_KEY 환경변수 필요."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise RuntimeError(
                "anthropic 패키지가 필요합니다: pip install anthropic --break-system-packages"
            )

    def structure(self, combined_text: str) -> StructuredPolicyFields:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(combined_text)}],
        )
        raw_text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print("\n[디버그] Claude가 실제로 응답한 내용:")
            print(repr(raw_text))
            print("[디버그] 여기까지가 응답 내용\n")
            raise

        trigger_events = [e for e in data.get("trigger_events", []) if e in EVENT_TAXONOMY]
        category = data.get("category") if data.get("category") in CATEGORY_TAXONOMY else "금융"
        allowed_employment = data.get("allowed_employment")
        if allowed_employment is not None:
            allowed_employment = [e for e in allowed_employment if e in EMPLOYMENT_TAXONOMY] or None
        allowed_housing = data.get("allowed_housing")
        if allowed_housing is not None:
            allowed_housing = [h for h in allowed_housing if h in HOUSING_TAXONOMY] or None

        amount = data.get("benefit_amount_krw")
        verified = validate_extracted_amount(amount, combined_text)
        notes = data.get("extraction_notes", "")
        if amount is not None and not verified:
            notes = f"[검수필요] LLM이 뽑은 금액({amount:,}원)이 원문 숫자로 설명 안 됨 - {notes}"

        return StructuredPolicyFields(
            benefit_amount_krw=amount,
            benefit_period_month=data.get("benefit_period_month"),
            benefit_rate_pct=data.get("benefit_rate_pct"),
            allowed_employment=allowed_employment,
            allowed_housing=allowed_housing,
            trigger_events=trigger_events,
            category=category,
            extraction_notes=notes,
            verified=verified,
        )


def get_policy_structurer() -> PolicyStructurer:
    backend = os.environ.get("POLICY_STRUCTURER_BACKEND", "regex")
    if backend == "anthropic":
        return AnthropicPolicyStructurer()
    return RegexPolicyStructurer()