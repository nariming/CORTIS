"""
재령이의 FastAPI 백엔드(GET /cohorts)에서 실제 코호트 데이터를 가져와
CohortIndex에 로드하는 연결 모듈.

사용법 (데모 스크립트에서):
    from pipeline.backend_client import load_cohort_index_from_backend
    index = load_cohort_index_from_backend(embedder)
"""

import requests
import datetime
from pipeline.embedding import EmbeddingProvider
from pipeline.similarity import CohortIndex
from pipeline.contracts import PredictionSaveRequest, PolicyMatchRequest

BACKEND_BASE_URL = "http://localhost:8000"


def fetch_cohorts_from_backend(base_url: str = BACKEND_BASE_URL) -> list:
    """GET /cohorts 호출해서 원본 row 리스트를 그대로 반환.

    서버(uvicorn backend.main:app --reload)가 켜져 있어야 함.
    """
    resp = requests.get(f"{base_url}/cohorts", timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_cohort_index_from_backend(embedder: EmbeddingProvider, base_url: str = BACKEND_BASE_URL) -> CohortIndex:
    """실제 백엔드에서 코호트를 가져와 CohortIndex를 만들어 반환.

    rows의 embedding_vector를 그대로 쓰므로, 임베딩을 다시 계산하지 않는다.
    (단, 재령이 시드 스크립트와 우리 EMBEDDING_BACKEND가 같은 방식이어야
     유사도가 의미를 가짐 — 지금은 둘 다 offline-hash-64라 문제 없음)
    """
    rows = fetch_cohorts_from_backend(base_url)
    index = CohortIndex(embedder)
    index.load_from_mysql_rows(rows)
    return index


def get_user_history(user_id: str, base_url: str = BACKEND_BASE_URL) -> dict:
    """GET /users/{user_id}/history 호출."""
    resp = requests.get(f"{base_url}/users/{user_id}/history", timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_life_event(user_id: str, event_type: str, base_url: str = BACKEND_BASE_URL) -> dict:
    """POST /users/{user_id}/events 호출해서 이벤트를 '확정' 상태로 등록하고,
    응답(event_id 포함)을 그대로 반환.
    """
    payload = {
        "event_type": event_type,
        "occurred_at": datetime.date.today().isoformat(),
        "status": "confirmed",
        "confidence": 1.0,
    }
    resp = requests.post(f"{base_url}/users/{user_id}/events", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def save_prediction(
    user_id: str,
    trigger_event_id: int,
    input_history: list,
    predictions: list,
    confidence_level: str,
    confidence_note: str,
    matched_cohorts: list,
    base_url: str = BACKEND_BASE_URL,
) -> dict:
    """POST /users/{user_id}/predictions 호출해서 예측 결과를 저장하고,
    응답(prediction_id 포함)을 그대로 반환.
    """
    req = PredictionSaveRequest(
        trigger_event_id=trigger_event_id,
        input_history=input_history,
        predictions=predictions,
        confidence_level=confidence_level,
        confidence_note=confidence_note,
        matched_cohorts=matched_cohorts,
    )
    resp = requests.post(f"{base_url}/users/{user_id}/predictions", json=asdict_safe(req), timeout=10)
    resp.raise_for_status()
    return resp.json()


def request_policy_match(
    user_id: str,
    prediction_id: int,
    event_types: list,
    include_ineligible: bool = False,
    base_url: str = BACKEND_BASE_URL,
) -> dict:
    """POST /users/{user_id}/policy-match 호출해서 A파트 정책 매칭을 실제로 요청."""
    req = PolicyMatchRequest(
        prediction_id=prediction_id,
        event_types=event_types,
        include_ineligible=include_ineligible,
    )
    resp = requests.post(f"{base_url}/users/{user_id}/policy-match", json=asdict_safe(req), timeout=10)
    resp.raise_for_status()
    return resp.json()


def asdict_safe(dataclass_obj) -> dict:
    from dataclasses import asdict
    return asdict(dataclass_obj)
