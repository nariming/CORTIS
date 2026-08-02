"""
Scenario Tree — 단일 다음 이벤트 예측을 depth-2 분기 시나리오로 확장.

"다음 이벤트 1개"만 예측하면 시간축/경로 개념이 없어 "생애주기(life-cycle)"라는
이름에 걸맞지 않다는 문제의식에서 나온 모듈. 예:
  Scenario A: 독립(전세) -> 결혼   (결합확률 24.6%)
  Scenario B: 이직 -> 독립(월세)   (결합확률 11.2%)

설계 원칙
  - LLM을 호출하지 않는다. depth-1/depth-2 모두 CohortIndex.search()+aggregate_next_events()를
    재귀적으로 한 번 더 태우는 것뿐이라, cash_need/probability와 동일하게 전부 코호트
    집계값에서 결정론적으로 계산된다 — "Generator=규칙기반 후보 나열, Portfolio=계산,
    LLM=설명만"이라는 기존 역할 분담 원칙과 같은 선상.
  - depth-2 검색 시 query_state/query_tx는 depth-1과 동일한 값을 그대로 쓴다. 실제로는
    event1이 일어난 뒤 유저의 소득/거주 상태가 달라질 수 있지만(예: 독립 이후 소득 변화),
    그 변화를 시뮬레이션하는 State Transition 모델은 이 버전의 스코프 밖이다 — 이 한계는
    "State는 이벤트가 확정된 시점의 스냅샷"이라는 전제로 남겨둔다.
  - joint_probability_pct = P(event1) x P(event2 | history+event1) / 100. depth-2 검색의
    확률은 이미 "event1이 일어난 뒤"라는 조건 하에서 집계된 값이라 조건부 확률의 정의와 맞는다.
"""

from dataclasses import dataclass
from typing import List, Optional

from .similarity import CohortIndex

SCENARIO_LABELS = ["Scenario A", "Scenario B", "Scenario C", "Scenario D", "Scenario E"]


@dataclass
class ScenarioStep:
    event: str
    probability_pct: float  # 부모 조건 하에서의 조건부 확률(해당 depth 검색 결과 내 비중)
    expected_timing_months: Optional[float]
    expected_cash_need_krw: Optional[int]
    cash_need_source: Optional[str]


@dataclass
class Scenario:
    label: str
    steps: List[ScenarioStep]  # [depth-1 스텝, depth-2 스텝]
    joint_probability_pct: float

    def path_str(self) -> str:
        return " -> ".join(s.event for s in self.steps)


def _agg_to_step(event: str, agg: dict) -> ScenarioStep:
    return ScenarioStep(
        event=event,
        probability_pct=agg["probability_pct"],
        expected_timing_months=agg["avg_event_interval_months"],
        expected_cash_need_krw=agg["avg_cash_need_krw"],
        cash_need_source=agg["cash_need_source"],
    )


def build_scenario_tree(
    index: CohortIndex,
    confirmed_history: List[str],
    query_state: Optional[dict] = None,
    query_tx: Optional[dict] = None,
    top_k: int = 15,
    breadth: int = 3,
) -> List[Scenario]:
    """depth-1 top-`breadth` 후보 각각에 대해 depth-2까지 재검색해 시나리오 트리를 만든다.

    breadth=3이면 depth-1 확률 상위 3개 이벤트에 대해서만 depth-2를 확장한다 — 전체
    이벤트 종류(약 12개)를 다 확장하면 계산량 대비 얻는 정보가 적고, 상위 후보 몇 개만
    보는 게 실제 시나리오 브리핑 용도에 더 맞는다.
    """
    depth1_matches = index.search(confirmed_history, query_state=query_state, query_tx=query_tx, top_k=top_k)
    depth1_agg = index.aggregate_next_events(depth1_matches)
    if not depth1_agg:
        return []

    top_events = list(depth1_agg.items())[:breadth]

    scenarios: List[Scenario] = []
    for i, (event1, agg1) in enumerate(top_events):
        step1 = _agg_to_step(event1, agg1)

        extended_history = confirmed_history + [event1]
        depth2_matches = index.search(
            extended_history, query_state=query_state, query_tx=query_tx, top_k=top_k
        )
        depth2_agg = index.aggregate_next_events(depth2_matches)

        if depth2_agg:
            # 같은 이벤트가 바로 다음에 다시 나오는 경우("결혼 -> 결혼")는 논리적으로 부자연스러운
            # 케이스가 대부분이라, depth-2에서는 event1과 동일한 이벤트를 최우선 후보로 쓰지 않는다.
            # (State/Tx가 depth-1과 동일한 스냅샷이라 생기는 부작용의 표면 증상 완화용 가드 —
            # 근본 해결(State Transition 모델)은 이번 범위 밖. weighted_score/probability_pct
            # 자체는 그대로 두고, depth2_agg 안에서 어떤 항목을 고를지만 바꾼다 — 계산 로직 불변)
            non_repeat = [(e, a) for e, a in depth2_agg.items() if e != event1]
            if non_repeat:
                event2, agg2 = non_repeat[0]
            else:
                event2, agg2 = next(iter(depth2_agg.items()))
            step2 = _agg_to_step(event2, agg2)
            joint_pct = round(step1.probability_pct * step2.probability_pct / 100, 1)
            steps = [step1, step2]
        else:
            # depth-2 검색 결과가 없으면(코호트 부족 등) depth-1만으로 시나리오를 구성한다.
            joint_pct = step1.probability_pct
            steps = [step1]

        scenarios.append(
            Scenario(
                label=SCENARIO_LABELS[i] if i < len(SCENARIO_LABELS) else f"Scenario {i + 1}",
                steps=steps,
                joint_probability_pct=joint_pct,
            )
        )

    scenarios.sort(key=lambda s: -s.joint_probability_pct)
    return scenarios