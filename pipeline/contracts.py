"""
C파트가 재령이 백엔드(A파트/predictions 저장소)와 실제로 주고받는 요청 형식.

실제 Swagger 스펙(/docs)에서 확인한 필드명 그대로 맞춤:
  POST /users/{user_id}/events        -> event_id 반환
  POST /users/{user_id}/predictions   -> prediction_id 반환 (trigger_event_id 필요)
  POST /users/{user_id}/policy-match  -> prediction_id 필요
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json


@dataclass
class EventCreateRequest:
    """POST /users/{user_id}/events 요청 형식."""
    event_type: str
    occurred_at: str          # "YYYY-MM-DD"
    status: str = "confirmed"
    confidence: float = 1.0


@dataclass
class PredictionSaveRequest:
    """POST /users/{user_id}/predictions 요청 형식 (실제 PredictionIn 스키마에 맞춤)."""
    trigger_event_id: int
    input_history: List[str]
    predictions: List[dict]         # [{"event": ..., "evidence_count": ..., "reasoning": ...}, ...]
    confidence_level: str
    confidence_note: str
    matched_cohorts: List[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class PolicyMatchRequest:
    """POST /users/{user_id}/policy-match 요청 형식 (실제 PolicyMatchIn 스키마에 맞춤)."""
    prediction_id: int
    event_types: List[str]
    include_ineligible: bool = False
    prospective: bool = True
    persist: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
