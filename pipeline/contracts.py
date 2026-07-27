"""
C파트가 A파트(정책 매칭)/B파트(상환관리)에 넘길 요청의 데이터 형식(계약).

지금은 재령이의 A/B파트 API가 없어서 실제로 호출은 못 하지만,
"이런 모양의 데이터를 이렁 이유로 보낸다"를 코드로 못박아두면
나중에 재령이 API 주소만 받아서 꽂으면 되는 구조가 된다.
"""

from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class PolicyPrefetchRequest:
    """C -> A파트: '이 유저, 곧 이런 이벤트 겪을 것 같으니 관련 정책 미리 찾아놔' 요청.

    A파트가 만들 엔드포인트 예상 스펙 (재령이 확정 필요):
      POST /api/policy/prefetch
      Content-Type: application/json
    """
    user_id: str
    predicted_event: str          # 예: "독립(월세)"
    evidence_count: int           # 근거로 삼은 유사 코호트 수
    confidence_level: str         # "high" | "medium" | "low"
    reasoning: str                # LLM이 생성한 판단 근거 (사용자에게 보여줄 수도 있음)
    requested_action: str         # 예: "청년 전세자금대출 상품 사전 조회"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class RepaymentReplanRequest:
    """C -> B파트: '이 유저 생애주기 이벤트 확정됐으니 상환계획 다시 짜' 요청.

    B파트가 만들 엔드포인트 예상 스펙 (재령이 확정 필요):
      POST /api/repayment/replan
      Content-Type: application/json
    """
    user_id: str
    confirmed_event: str          # 방금 확정된 이벤트 (예: "취업")
    updated_history: List[str]    # 갱신된 전체 히스토리
    trigger_reason: str           # 예: "생애주기 이벤트 확정으로 인한 재설계"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
