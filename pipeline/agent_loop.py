"""
Agentic 순환 로직.

C(이벤트 감지+예측)가 매번 새 이벤트를 확정할 때마다:
  1. 유저 히스토리에 이벤트 추가
  2. CohortIndex로 재검색
  3. Reasoner로 재예측 (근거 포함)
  4. 예측 결과를 A(정책 재탐색)·B(상환계획 재설계)에 다시 넘김 (여기선 스텁 콜백)

"한 번 쓰고 끝나는 챗봇"이 아니라 "이벤트가 생길 때마다 스스로 다시 판단하는 에이전트"라는
지점을 코드 구조로 직접 보여주는 게 목적. A/B는 아직 개발자1 쪽 API가 없으므로 스텁으로 대체.
"""

from typing import Callable, List, Optional
from dataclasses import dataclass

from .similarity import CohortIndex
from .reasoning import Reasoner, PredictionResult
from .contracts import PolicyPrefetchRequest, RepaymentReplanRequest


@dataclass
class AgentState:
    user_id: str
    confirmed_history: List[str]


class CortisAgent:
    def __init__(
        self,
        index: CohortIndex,
        reasoner: Reasoner,
        on_predict_callback: Optional[Callable[[str, PredictionResult, List[str]], None]] = None,
        top_k: int = 5,
    ):
        self.index = index
        self.reasoner = reasoner
        self.top_k = top_k
        # A/B 재호출 콜백 — 개발자1의 API가 준비되면 이 함수 안의 print를
        # requests.post(A_ENDPOINT, json=request.to_json()) 형태로 바꾸기만 하면 됨
        self.on_predict_callback = on_predict_callback or self._default_stub_callback

    def _default_stub_callback(self, user_id: str, result: PredictionResult, updated_history: List[str]):
        """지금은 실제로 호출할 A/B 엔드포인트가 없어서, '이런 요청을 이런 형태로 보낼 것이다'를
        실제 요청 객체(contracts.py)로 만들어서 JSON으로 출력만 한다.

        재령이 API 완성되면 아래 print(...) 자리를 그대로 이렇게 바꾸면 됨:
            import requests
            requests.post("http://재령이서버주소/api/policy/prefetch", json=asdict(prefetch_req))
        """
        print(f"  [STUB] A/B 재호출 트리거 (user={user_id})")

        # B파트: 상환계획 재설계 요청 (이벤트 확정 자체가 트리거)
        replan_req = RepaymentReplanRequest(
            user_id=user_id,
            confirmed_event=updated_history[-1] if updated_history else "",
            updated_history=updated_history,
            trigger_reason="생애주기 이벤트 확정으로 인한 재설계",
        )
        print("  [STUB] -> B파트 요청 (실제 연결 시 POST /api/repayment/replan):")
        print("    " + replan_req.to_json().replace("\n", "\n    "))

        # A파트: 예측된 각 후보 이벤트에 대해 정책 사전조회 요청
        for pred in result.predictions:
            prefetch_req = PolicyPrefetchRequest(
                user_id=user_id,
                predicted_event=pred["event"],
                evidence_count=pred.get("evidence_count", 0),
                confidence_level=result.confidence_level,
                reasoning=pred.get("reasoning", ""),
                requested_action=next(
                    (p["action"] for p in result.suggested_preparations if p["event"] == pred["event"]),
                    f"{pred['event']} 관련 정책 사전조회",
                ),
            )
            print(f"  [STUB] -> A파트 요청 (실제 연결 시 POST /api/policy/prefetch, event={pred['event']}):")
            print("    " + prefetch_req.to_json().replace("\n", "\n    "))

    def on_event_confirmed(self, state: AgentState, new_event: str, user_context: Optional[str] = None) -> PredictionResult:
        """이벤트가 규칙기반 감지+확인질문을 거쳐 '확정'되었을 때 호출되는 진입점.

        (규칙기반 감지 자체는 A파트 모듈의 detect_* 함수들이 담당 — 여기선 이미
        확정된 이벤트가 들어온다고 가정)
        """
        state.confirmed_history.append(new_event)

        matches = self.index.search(state.confirmed_history, top_k=self.top_k)
        counts = self.index.aggregate_next_events(matches)
        result = self.reasoner.predict(
            confirmed_history=state.confirmed_history,
            matches=matches,
            next_event_counts=counts,
            user_context=user_context,
        )

        self.on_predict_callback(state.user_id, result, state.confirmed_history)
        return result
