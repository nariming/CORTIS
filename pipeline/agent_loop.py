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


@dataclass
class AgentState:
    user_id: str
    confirmed_history: List[str]


class CortisAgent:
    def __init__(
        self,
        index: CohortIndex,
        reasoner: Reasoner,
        on_predict_callback: Optional[Callable[[str, PredictionResult], None]] = None,
        top_k: int = 5,
    ):
        self.index = index
        self.reasoner = reasoner
        self.top_k = top_k
        # A/B 재호출 스텁 — 개발자1의 API가 준비되면 실제 호출로 교체
        self.on_predict_callback = on_predict_callback or self._default_stub_callback

    def _default_stub_callback(self, user_id: str, result: PredictionResult):
        print(f"  [STUB] A/B 재호출 트리거 (user={user_id})")
        for prep in result.suggested_preparations:
            print(f"  [STUB] -> A파트에 '{prep['event']}' 관련 정책 사전조회 요청: {prep['action']}")

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

        self.on_predict_callback(state.user_id, result)
        return result
