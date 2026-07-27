"""
LLM 추론 단계: 코호트 검색 결과 -> 다음 이벤트 예측 + 근거 (JSON 출력)

prompts/next_event_prediction.md 의 프롬프트를 그대로 코드로 옮긴 것.
API 키 없이도 파이프라인 전체를 검증할 수 있도록 MockReasoner를 기본 제공하고,
LLM_BACKEND=anthropic 환경변수로 실제 API 호출로 전환 가능하게 구성.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

from .similarity import CohortMatch

SYSTEM_PROMPT = """당신은 청년 금융 생애주기 이벤트를 예측하는 분석 엔진입니다.
검색된 유사 코호트 사례만을 근거로 사용하고, 근거 없는 확률을 지어내지 마세요.
반드시 아래 JSON 스키마로만 응답하세요. 다른 텍스트를 추가하지 마세요.

{
  "predictions": [
    {"event": "이벤트명", "evidence_count": 숫자, "reasoning": "근거 1-2문장"}
  ],
  "confidence_level": "high" | "medium" | "low",
  "confidence_note": "확신도 판단 이유",
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


def build_user_prompt(
    confirmed_history: List[str],
    matches: List[CohortMatch],
    next_event_counts: dict,
    user_context: Optional[str] = None,
) -> str:
    history_str = " -> ".join(confirmed_history) if confirmed_history else "(없음)"
    matches_str = "\n".join(
        f"- 히스토리: {' -> '.join(m.history)} / 다음 이벤트: {m.next_event} / 유사도: {m.similarity:.2f}"
        for m in matches
    ) or "(검색된 유사 사례 없음)"
    counts_str = ", ".join(f"{k} {v}건" for k, v in next_event_counts.items()) or "(집계 없음)"

    return f"""[확정된 이벤트 히스토리]
{history_str}

[검색된 유사 코호트 top-{len(matches)}]
{matches_str}

[다음 이벤트 집계 (top-k 중)]
{counts_str}

[유저 현재 상황 (참고)]
{user_context or "정보 없음"}

위 근거만으로 다음 이벤트를 예측하고, 반드시 JSON 스키마로만 응답하세요."""


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

    실제 LLM 호출 없이, 집계 결과를 그대로 규칙적으로 JSON화한다.
    데모/개발 단계에서 로직이 도는지 확인하는 용도 — 발표용 최종 데모에는
    반드시 AnthropicReasoner 등 실제 LLM으로 교체해야 함 (그래야 '근거 문장 생성'이라는
    AI다움이 실제로 증명됨).
    """

    def predict(self, confirmed_history, matches, next_event_counts, user_context=None):
        predictions = [
            {
                "event": event,
                "evidence_count": count,
                "reasoning": f"[MOCK] 유사 코호트 {count}건이 '{event}'을(를) 다음 이벤트로 겪음",
            }
            for event, count in next_event_counts.items()
        ]

        is_cold_start = len(confirmed_history) < 2 or len(matches) < 2
        confidence = "low" if is_cold_start else (
            "high" if predictions and predictions[0]["evidence_count"] >= 3 else "medium"
        )
        confidence_note = (
            "초기 데이터 부족으로 신뢰도가 낮음 (히스토리 또는 유사 사례 부족)"
            if is_cold_start
            else "유사 코호트 검색 결과 기반 판단"
        )

        return PredictionResult(
            predictions=predictions,
            confidence_level=confidence,
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

        # 콜드스타트는 LLM 판단에만 맡기지 않고 코드로 강제 (방어 목적)
        is_cold_start = len(confirmed_history) < 2 or len(matches) < 2
        if is_cold_start:
            data["confidence_level"] = "low"
            data["confidence_note"] = "초기 데이터 부족으로 신뢰도가 낮음 (코드 레벨에서 강제)"

        return PredictionResult(
            predictions=data.get("predictions", []),
            confidence_level=data.get("confidence_level", "low"),
            confidence_note=data.get("confidence_note", ""),
            suggested_preparations=data.get("suggested_preparations", []),
        )


def get_reasoner() -> Reasoner:
    backend = os.environ.get("LLM_BACKEND", "mock")
    if backend == "anthropic":
        return AnthropicReasoner()
    return MockReasoner()
