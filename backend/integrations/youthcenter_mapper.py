"""
온통청년 API 원본 필드 → CORTIS policies 테이블 컬럼 매핑.

⚠️ 실제 응답을 검증 못 한 상태로 작성했다 (youthcenter_client.py 상단 설명 참고).
온통청년 API는 세대가 바뀌면서 필드명이 한 번 크게 바뀐 이력이 있어(구버전 polyBizSjnm 계열,
신버전 plcyNm 계열), FIELD_CANDIDATES 에 두 세대 후보를 모두 넣어 어느 쪽이 와도 최대한 매핑되게 했다.
실제 응답을 받아보면 CANDIDATES 순서/목록만 다듬으면 되고, 나머지 로직은 안 건드려도 된다.

숫자 추출(정액지원금/개월수)은 지원내용이 자유텍스트라 정규식 best-effort다.
실패하면 해당 필드는 NULL로 남긴다 — 잘못된 숫자를 넣는 것보다 "모름"이 안전하다.
"""

import re
from typing import List, Optional

# 우리 컬럼명 -> 온통청년 응답에서 시도해볼 후보 태그명(우선순위 순)
FIELD_CANDIDATES = {
    "policy_no": ["plcyNo", "bizId", "polyBizId"],
    "policy_name": ["plcyNm", "polyBizSjnm", "polyItcnNm"],
    "description": ["plcyExplnCn", "polyItcnCn"],
    "support_content": ["plcySprtCn", "sporCn", "plcyCn"],
    "provider": ["sprvsnInstCdNm", "rgtrInstCdNm", "operInstCdNm"],
    "apply_url": ["aplyUrlAddr", "refUrlAddr1", "rqutUrlAddr"],
    "min_age": ["sprtTrgtMinAge", "minAge"],
    "max_age": ["sprtTrgtMaxAge", "maxAge"],
    "marital_code": ["mrgSttsCd"],
    "income_min": ["earnMinAmt"],
    "income_max": ["earnMaxAmt"],
    "region_code": ["zipCd", "rgtrUpInstCd"],
    "keyword": ["plcyKywdNm", "polyKywdNm"],
    "large_category": ["lclsfNm", "polyRlmCd"],
    "mid_category": ["mclsfNm"],
}

# 대분류/중분류/키워드/정책명을 합친 텍스트에서 우리 도메인의 생애주기 이벤트로 매핑.
# 매칭되는 게 하나도 없으면 trigger_events=[] 로 남긴다 — A파트 트리거 매칭에서는 안 잡히지만
# GET /users/{id}/policies(현재 자격 무관 조회)에는 계속 노출되니 데이터 자체가 사라지진 않는다.
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

# 정책 카테고리(우리 ENUM)로의 대략적 매핑 — 못 찾으면 '금융'을 기본값으로 (가장 무난한 폴백)
CATEGORY_KEYWORD_MAP = {
    "주거": ["주거", "주택", "전세", "월세", "임대"],
    "고용": ["일자리", "취업", "고용", "구직"],
    "창업": ["창업"],
    "결혼출산": ["결혼", "출산", "육아", "임신"],
    "교육": ["교육", "훈련", "역량"],
}

AMOUNT_PATTERNS = [
    re.compile(r"월\s?(\d{1,3})\s?만\s?원.{0,10}?(\d{1,2})\s?개월"),  # "월 20만원 x 12개월"
    re.compile(r"연\s?(\d{1,4})\s?만\s?원"),                          # "연 240만원"
    re.compile(r"최대\s?(\d{1,4})\s?만\s?원"),                        # "최대 1,200만원" (콤마는 미리 제거)
]


def _first_present(raw: dict, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        val = raw.get(key)
        if val:
            return val
    return None


def _to_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def _keyword_hit(text: str, keyword_map: dict) -> List[str]:
    return [label for label, kws in keyword_map.items() if any(kw in text for kw in kws)]


def extract_benefit_amount(support_text: str) -> tuple:
    """지원내용 자유텍스트에서 (총 지원금액(원), 지원개월수) 를 best-effort로 뽑는다.

    둘 다 못 찾으면 (None, None). "월 X만원 x N개월" 패턴이 잡히면 총액도 함께 계산.
    """
    if not support_text:
        return None, None
    cleaned = support_text.replace(",", "")

    m = AMOUNT_PATTERNS[0].search(cleaned)
    if m:
        monthly_man, months = int(m.group(1)), int(m.group(2))
        return monthly_man * 10_000 * months, months

    for pattern in AMOUNT_PATTERNS[1:]:
        m = pattern.search(cleaned)
        if m:
            return int(m.group(1)) * 10_000, None

    return None, None


def map_raw_to_policy(raw: dict) -> Optional[dict]:
    """온통청년 원본 1건 -> policies 테이블 insert용 dict.

    필수 필드(policy_no, policy_name)가 아예 없으면 이 레코드는 태그명 후보가
    하나도 안 맞았다는 뜻이라 None을 반환한다 (호출부에서 스킵 카운트로 집계).
    """
    policy_no = _first_present(raw, FIELD_CANDIDATES["policy_no"])
    name = _first_present(raw, FIELD_CANDIDATES["policy_name"])
    if not policy_no or not name:
        return None

    description = _first_present(raw, FIELD_CANDIDATES["description"]) or ""
    support = _first_present(raw, FIELD_CANDIDATES["support_content"]) or ""
    keyword = _first_present(raw, FIELD_CANDIDATES["keyword"]) or ""
    large_cat = _first_present(raw, FIELD_CANDIDATES["large_category"]) or ""
    combined_text = " ".join([name, description, support, keyword, large_cat])

    benefit_amount, benefit_period = extract_benefit_amount(support)

    marital_code = _first_present(raw, FIELD_CANDIDATES["marital_code"])
    allowed_marital = None
    if marital_code:
        if "기혼" in marital_code:
            allowed_marital = ["기혼"]
        elif "미혼" in marital_code:
            allowed_marital = ["미혼"]
        # "제한없음"류 코드면 allowed_marital=None(무관) 유지

    region_raw = _first_present(raw, FIELD_CANDIDATES["region_code"])
    region_code = None
    if region_raw and region_raw not in ("00", "0000000", ""):
        region_code = region_raw[:2]

    trigger_events = _keyword_hit(combined_text, EVENT_KEYWORD_MAP)
    categories = _keyword_hit(combined_text, CATEGORY_KEYWORD_MAP)

    return {
        "policy_id": f"YC-{policy_no}",
        "policy_name": name.strip()[:150],
        "provider": (_first_present(raw, FIELD_CANDIDATES["provider"]) or "온통청년").strip()[:50],
        "category": categories[0] if categories else "금융",
        "benefit_summary": (support or description or name).strip()[:500],
        "apply_url": (_first_present(raw, FIELD_CANDIDATES["apply_url"]) or None),
        "min_age": _to_int(_first_present(raw, FIELD_CANDIDATES["min_age"])),
        "max_age": _to_int(_first_present(raw, FIELD_CANDIDATES["max_age"])),
        "max_annual_income": _to_int(_first_present(raw, FIELD_CANDIDATES["income_max"])),
        "allowed_employment": None,   # API에 구조화된 고용형태 필드가 없어 무관(NULL) 처리
        "allowed_housing": None,      # 마찬가지로 무관 처리 (오탐으로 과도하게 걸러내지 않기 위함)
        "allowed_marital": allowed_marital,
        "region_code": region_code,
        "trigger_events": trigger_events,
        "priority": 3,                # 수작업 큐레이션(우선순위 1~2)보다 낮게, 기본 노출은 되게
        "benefit_amount_krw": benefit_amount,
        "benefit_period_month": benefit_period,
        "benefit_rate_pct": None,     # 지원내용 텍스트에서 금리는 신뢰도 낮아 자동 추출 안 함
        "source": "youthcenter_api",
        "external_policy_no": policy_no,
    }
