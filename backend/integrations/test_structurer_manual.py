"""
policy_structurer.py의 AnthropicPolicyStructurer를 실제 API로 한 번 확인하기 위한 수동 테스트.
자동 테스트 스위트(tests/)에 넣지 않은 이유: 실제 API 비용이 발생하고, 응답이 결정론적이지
않아 CI에서 자동 검증하기엔 안 맞음. 발표 준비 중 "진짜로 되는지" 눈으로 확인하는 용도.

실행:
    ANTHROPIC_API_KEY가 backend/.env 또는 최상위 .env에 있어야 함.
    python -m backend.integrations.test_structurer_manual
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

os.environ["POLICY_STRUCTURER_BACKEND"] = "anthropic"

from backend.integrations.policy_structurer import get_policy_structurer  # noqa: E402

TEST_CASES = [
    {
        "name": "독립가구 이사정착 지원 (키워드 없음 케이스)",
        "combined_text": (
            "청년 독립가구 이사정착 지원 "
            "부모로부터 독립해 별도 세대를 꾸린 청년의 초기 정착을 돕는 사업 "
            "가전·가구 구입비 등 정착비용 최대 100만원 지원 "
            "독립,정착 생활안정"
        ),
        "expect_event": "독립(월세)",
    },
    {
        "name": "재도전 청년 창업 디딤돌 (사업화 키워드는 있으나 표현 변형)",
        "combined_text": (
            "재도전 청년 창업 디딤돌 사업 "
            "실패를 경험한 청년이 다시 자기 사업을 꾸릴 수 있도록 초기 자금을 지원 "
            "사업 재기 자금 최대 2000만원, 24개월 분할 지급 "
            "재창업,재도전 창업지원"
        ),
        "expect_event": "창업",
    },
]


def main():
    structurer = get_policy_structurer()
    print(f"백엔드: {type(structurer).__name__}\n")

    for case in TEST_CASES:
        print(f"{'=' * 60}\n{case['name']}\n{'=' * 60}")
        result = structurer.structure(case["combined_text"])
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

        got_events = result.trigger_events
        ok = case["expect_event"] in got_events
        print(f"\n[판정] 기대 이벤트 '{case['expect_event']}' 포함 여부: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  (실제 trigger_events: {got_events})")
        print()


if __name__ == "__main__":
    main()