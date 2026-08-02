"""
LLM 추론 단계: 코호트 검색 결과 -> 다음 이벤트 예측 + 근거 (JSON 출력)

prompts/next_event_prediction.md 의 프롬프트를 그대로 코드로 옮긴 것.
API 키 없이도 파이프라인 전체를 검증할 수 있도록 MockReasoner를 기본 제공하고,
LLM_BACKEND=anthropic 환경변수로 실제 API 호출로 전환 가능하게 구성.

역할 분리 원칙 (2026.8 확장)
  확률(probability_pct)/예상시점(expected_timing_months)/예상필요자금(expected_cash_need_krw)은
  전부 similarity.CohortIndex.aggregate_next_events()가 코호트 집계값으로 미리 계산해 둔다.
  LLM은 이 숫자를 새로 만들거나 바꾸지 않고, 아래 두 가지만 담당한다:
    1. Rerank/Critic — 각 이벤트 후보가 [유저 현재 상황]과 논리적으로 모순되지 않는지 확인
       (예: 이미 월세 거주 중인데 "독립(월세)" 후보가 나오면 모순)
    2. Reasoning — population 수준 근거(코호트 검색 결과)에 개인 신호(State/Transaction
       유사도, 유저 현재 상황)가 어떻게 결합되어 이 결과가 나왔는지 설명

  이렇게 나누는 이유: LLM이 확률/금액까지 자유롭게 생성하면 "근거 없는 숫자를 지어내지 않는다"는
  이 파이프라인의 핵심 방어 논리가 깨진다. 반대로 Rerank/Critic/설명까지 코드 규칙으로만
  처리하면 "이 유저는 이미 월세인데 다시 독립(월세)를 추천하는" 것 같은 문맥적 모순을
  잡아낼 수 없다 — 이건 자연어 이해가 필요한 영역이라 LLM에게 맡긴다.
"""

import os
import math
import json
from dataclasses import dataclass, field
from typing import List, Optional

from .similarity import CohortMatch
from .scenario_tree import Scenario

# backend/.env 의 COLD_START_THRESHOLD 와 반드시 같은 값을 참조해야 한다.
# (재령이 백엔드 설정과 여기 값이 따로 놀면, 콜드스타트 판단 기준이 두 곳에서 어긋날 수 있음)
COLD_START_THRESHOLD = int(os.environ.get("COLD_START_THRESHOLD", "2"))

SYSTEM_PROMPT = """당신은 청년 금융 생애주기 이벤트를 예측하는 분석 엔진입니다.

[역할 분리 - 반드시 지킬 것]
각 이벤트 후보의 확률(probability_pct)·예상시점(expected_timing_months)·예상필요자금
(expected_cash_need_krw)은 이미 코드가 검색된 코호트 집계값으로 계산해서 프롬프트에 제공합니다.
이 숫자를 스스로 새로 만들거나 다른 값으로 바꾸지 마세요. 당신의 역할은 다음 두 가지뿐입니다:

1. Rerank/Critic: 각 이벤트 후보가 [유저 현재 상황]과 논리적으로 모순되지 않는지 확인하세요.
   모순 예시: 유저가 이미 월세로 거주 중인데 "독립(월세)" 후보가 있는 경우,
   유저가 이미 기혼인데 "결혼" 후보가 있는 경우 등. 모순이 있으면 is_consistent를 false로,
   consistency_note에 이유를 적으세요. 모순이 없으면 is_consistent는 true, consistency_note는
   빈 문자열로 두세요.
2. Reasoning: population 수준 사전분포(코호트 검색 결과, History/State/Transaction 유사도
   breakdown)에 이 유저의 개인 신호가 어떻게 결합되어 이 결과가 나왔는지 1-2문장으로
   설명하세요. 검색된 유사 코호트 사례만을 근거로 사용하고, 근거 없는 확률/금액/개월수를
   새로 지어내지 마세요.

반드시 아래 JSON 스키마로만 응답하세요. 다른 텍스트를 추가하지 마세요.

{
  "predictions": [
    {"event": "이벤트명", "is_consistent": true, "consistency_note": "", "reasoning": "근거 1-2문장"}
  ],
  "confidence_note": "이벤트 분포가 특정 이벤트에 집중돼 있는지, 유저 상태와 모순되는 후보가
    있는지 등을 참고해 확신도 판단 이유를 서술 (confidence_level 자체는 코드가 결정합니다)",
  "suggested_preparations": [
    {"event": "이벤트명", "action": "A/B파트에 미리 요청할 사전 조치"}
  ]
}"""


@dataclass
class PredictionResult:
    predictions: List[dict]
    confidence_level: str
    confidence_note: str
    suggested_preparations: List[dict] = field(default_factory=list)
    # Scenario Tree(depth-2 분기 시나리오). agent_loop.py가 예측 직후 채워 넣는다 —
    # Reasoner.predict() 자체는 LLM 호출 1회 단위 책임만 지고, 시나리오는 LLM을 호출하지
    # 않는 별도 계산(코호트 재검색)이라 Reasoner 책임 밖에 둔다.
    scenarios: List[Scenario] = field(default_factory=list)


def compute_confidence(next_event_counts: dict, is_cold_start: bool) -> "tuple[str, str]":
    """유사도 분포의 엔트로피 기반 confidence 계산 (코드가 결정 — LLM에 맡기지 않음).

    이전에는 "evidence_count(건수) >= 3이면 high" 식의 단순 규칙이었다. 문제는 건수만 보면
    "독립(월세) 14건, 나머지 1건씩"처럼 한 이벤트에 쏠린 경우와 "독립/결혼/이직 각 5건씩"처럼
    고르게 갈린 경우를 구분하지 못한다는 점이다. 엔트로피는 이 "쏠림 정도"를 직접 측정한다.

    normalized_entropy가 낮을수록(0에 가까울수록) 특정 이벤트에 근거가 집중된 것이고,
    높을수록(1에 가까울수록) 여러 이벤트에 고르게 분산돼 어느 하나로 확신하기 어렵다는 뜻이다.
    콜드스타트는 이 계산과 무관하게 항상 low로 강제한다 — 히스토리/코호트 자체가 부족한
    상태에서는 분포가 우연히 집중돼 보여도 신뢰할 수 없기 때문이다.
    """
    if is_cold_start:
        return "low", "초기 데이터 부족으로 신뢰도가 낮음 (히스토리 또는 유사 사례 부족, 코드 레벨에서 강제)"

    scores = [v["weighted_score"] for v in next_event_counts.values()]
    total = sum(scores)
    if total <= 0 or not scores:
        return "low", "검색된 유사 사례 없음"

    probs = [s / total for s in scores if s > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    top1_share = max(probs)

    if top1_share >= 0.4 and normalized_entropy < 0.7:
        level = "high"
        note = f"최상위 이벤트 비중 {top1_share:.0%}로 근거가 집중됨(분포 집중도 {max(0.0, 1 - normalized_entropy):.0%})"
    elif top1_share < 0.25 or normalized_entropy > 0.85:
        level = "low"
        note = f"여러 이벤트에 근거가 고르게 분산돼(분포 집중도 {max(0.0, 1 - normalized_entropy):.0%}) 특정 이벤트로 확신하기 어려움"
    else:
        level = "medium"
        note = f"근거는 있으나 확신 수준은 중간(최상위 비중 {top1_share:.0%}, 분포 집중도 {max(0.0, 1 - normalized_entropy):.0%})"
    return level, note


def _merge_predictions(next_event_counts: dict, llm_items: List[dict]) -> List[dict]:
    """코드가 계산한 확률/시점/필요자금(next_event_counts)과 LLM이 판단한
    is_consistent/reasoning(llm_items)을 이벤트명으로 매칭해 합친다.

    LLM 응답에 없는 이벤트(파싱 누락 등)는 보수적으로 is_consistent=True로 두되
    reasoning은 빈 문자열로 남겨, 화면에 "근거 없음"이 아니라 "설명 누락"임을 구분할 수 있게 한다.
    정렬은 is_consistent(모순 없는 것 우선) -> probability_pct(코드 계산값) 순으로,
    모순으로 판단된 후보를 숨기지 않고 뒤로 내려서 사용자가 직접 판단할 여지를 남긴다.
    """
    llm_by_event = {item.get("event"): item for item in llm_items if isinstance(item, dict)}

    merged = []
    for event, agg in next_event_counts.items():
        llm_item = llm_by_event.get(event, {})
        merged.append({
            "event": event,
            "evidence_count": agg["count"],
            "weighted_score": agg["weighted_score"],
            "probability_pct": agg["probability_pct"],
            "expected_timing_months": agg["avg_event_interval_months"],
            "expected_cash_need_krw": agg["avg_cash_need_krw"],
            "cash_need_source": agg["cash_need_source"],
            "is_consistent": bool(llm_item.get("is_consistent", True)),
            "consistency_note": llm_item.get("consistency_note", ""),
            "reasoning": llm_item.get("reasoning", ""),
        })

    merged.sort(key=lambda p: (not p["is_consistent"], -p["probability_pct"]))
    return merged


def build_user_prompt(
    confirmed_history: List[str],
    matches: List[CohortMatch],
    next_event_counts: dict,
    user_context: Optional[str] = None,
) -> str:
    history_str = " -> ".join(confirmed_history) if confirmed_history else "(없음)"
    matches_str = "\n".join(
        f"- 히스토리: {' -> '.join(m.history)} / 다음 이벤트: {m.next_event} / "
        f"유사도: {m.similarity:.2f} (history={m.sim_history:.2f}, state={m.sim_state:.2f}, tx={m.sim_tx:.2f})"
        for m in matches
    ) or "(검색된 유사 사례 없음)"

    def _fmt_event_agg(event: str, agg: dict) -> str:
        timing = f"{agg['avg_event_interval_months']:.0f}개월 후" if agg["avg_event_interval_months"] is not None else "시점 추정 불가"
        cash = f"약 {agg['avg_cash_need_krw']:,}원 ({agg['cash_need_source']})" if agg["avg_cash_need_krw"] is not None else "해당 없음"
        return (
            f"- {event}: 확률 {agg['probability_pct']}% (근거 {agg['count']}건, 유사도합 {agg['weighted_score']:.2f}), "
            f"예상시점 {timing}, 예상필요자금 {cash}"
        )

    counts_str = "\n".join(_fmt_event_agg(k, v) for k, v in next_event_counts.items()) or "(집계 없음)"

    return f"""[확정된 이벤트 히스토리]
{history_str}

[검색된 유사 코호트 top-{len(matches)}] (유사도는 History/State/Transaction 3개 공간의 가중합)
{matches_str}

[이벤트별 집계 - 확률/시점/필요자금은 코드가 이미 계산한 값입니다. 이 숫자를 바꾸지 마세요]
{counts_str}

[유저 현재 상황 (참고 - Rerank/Critic 판단에 사용)]
{user_context or "정보 없음"}

각 이벤트에 대해 is_consistent(유저 현재 상황과 모순 여부)와 reasoning(설명)을 판단하고,
반드시 JSON 스키마로만 응답하세요."""


class Reasoner:
    def predict(
        self,
        confirmed_history: List[str],
        matches: List[CohortMatch],
        next_event_counts: dict,
        user_context: Optional[str] = None,
    ) -> PredictionResult:
        raise NotImplementedError


class MockReasoner(Reasoner):
    """API 키 없이 파이프라인 배선을 검증하기 위한 목업.

    실제 LLM 호출 없이 코드 계산값(확률/시점/필요자금/confidence)만 그대로 JSON화한다.
    Rerank/Critic(문맥 모순 판단)은 자연어 이해가 필요한 영역이라 목업으로는 구현할 수 없어,
    모든 후보를 is_consistent=True로 둔다 — 이는 목업의 한계이지 실제 서비스 동작이 아니다.
    데모/개발 단계에서 로직이 도는지 확인하는 용도로만 쓰고, 발표용 최종 데모에는
    반드시 AnthropicReasoner 등 실제 LLM으로 교체해야 한다(그래야 Rerank/Critic이 실제로 작동함).
    """

    def predict(self, confirmed_history, matches, next_event_counts, user_context=None):
        mock_llm_items = [
            {
                "event": event,
                "is_consistent": True,
                "consistency_note": "",
                "reasoning": (
                    f"[MOCK] 유사 코호트 {agg['count']}건(확률 {agg['probability_pct']}%)이 "
                    f"'{event}'을(를) 다음 이벤트로 겪음 — Rerank/Critic은 목업에서 지원 안 함"
                ),
            }
            for event, agg in next_event_counts.items()
        ]
        predictions = _merge_predictions(next_event_counts, mock_llm_items)

        is_cold_start = len(confirmed_history) < COLD_START_THRESHOLD or len(matches) < COLD_START_THRESHOLD
        confidence_level, confidence_note = compute_confidence(next_event_counts, is_cold_start)

        return PredictionResult(
            predictions=predictions,
            confidence_level=confidence_level,
            confidence_note=confidence_note,
            suggested_preparations=[
                {"event": p["event"], "action": f"[MOCK] {p['event']} 관련 정책/상품 사전 조회 요청"}
                for p in predictions[:1]
            ],
        )


class AnthropicReasoner(Reasoner):
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

    def predict(self, confirmed_history, matches, next_event_counts, user_context=None) -> PredictionResult:
        user_prompt = build_user_prompt(confirmed_history, matches, next_event_counts, user_context)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
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

        # 확률/시점/필요자금/confidence_level은 LLM 응답과 무관하게 전부 코드 계산값으로 대체한다.
        # (LLM이 확률·금액을 스스로 바꿔 응답했더라도 무시 — 역할 분리 원칙을 코드 레벨에서 강제)
        predictions = _merge_predictions(next_event_counts, data.get("predictions", []))

        is_cold_start = len(confirmed_history) < COLD_START_THRESHOLD or len(matches) < COLD_START_THRESHOLD
        confidence_level, code_note = compute_confidence(next_event_counts, is_cold_start)
        # 콜드스타트가 아니면 LLM이 쓴 confidence_note를 우선 사용한다 — 실제로 LLM이
        # "state/transaction 정보가 없어 신뢰도가 제한적"과 같이 코드가 못 보는 맥락을
        # 짚어내는 경우가 있어(agent_loop.py 실행 로그에서 확인됨), 이 설명력을 살린다.
        # 콜드스타트일 때는 코드 강제 문구를 그대로 쓴다(기존 방어 동작 유지).
        confidence_note = code_note if is_cold_start else (data.get("confidence_note") or code_note)

        return PredictionResult(
            predictions=predictions,
            confidence_level=confidence_level,
            confidence_note=confidence_note,
            suggested_preparations=data.get("suggested_preparations", []),
        )


def get_reasoner() -> Reasoner:
    backend = os.environ.get("LLM_BACKEND", "mock")
    if backend == "anthropic":
        return AnthropicReasoner()
    return MockReasoner()