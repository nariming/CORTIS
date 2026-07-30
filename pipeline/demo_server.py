"""
데모 웹 화면 전용 경량 서버 (포트 8001).

재령이의 backend(포트 8000)는 건드리지 않고, "이벤트 확정 -> 검색 -> LLM추론 ->
예측저장 -> 정책매칭"을 한 번의 호출로 묶어서 브라우저에 노출하기 위한 서버.

demo_full_flow.py 의 로직을 그대로 함수화해서 재사용한다.

실행: uvicorn pipeline.demo_server:app --port 8001 --reload
(backend 서버(8000)도 같이 켜져 있어야 함)
"""

import os
from dotenv import load_dotenv
load_dotenv()
load_dotenv("backend/.env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.embedding import get_embedding_provider
from pipeline.reasoning import get_reasoner
from pipeline import backend_client as bc

app = FastAPI(title="CORTIS 데모 UI 서버 (C엔진)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedder = get_embedding_provider()
_reasoner = get_reasoner()


class ConfirmEventRequest(BaseModel):
    user_id: str
    event_type: str
    include_ineligible: bool = False


class PredictRequest(BaseModel):
    user_id: str
    trigger_event_id: int | None = None
    include_ineligible: bool = False


@app.get("/demo/latest-prediction/{user_id}")
def demo_latest_prediction(user_id: str):
    """재령이 백엔드의 예측 이력(GET /users/{id}/predictions)에서 가장 최근 것만 반환.

    새로고침해도 마지막 예측 결과를 다시 보여주기 위한 용도.
    """
    import requests
    resp = requests.get(f"{bc.BACKEND_BASE_URL}/users/{user_id}/predictions", params={"limit": 1}, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    row = rows[0]
    return {
        "predictions": row["predictions_json"],
        "confidence_level": row["confidence_level"],
        "confidence_note": row["confidence_note"],
        "created_at": row["created_at"],
    }


@app.get("/demo/users")
def demo_users():
    """재령이 백엔드의 유저 목록을 그대로 중계."""
    import requests
    resp = requests.get(f"{bc.BACKEND_BASE_URL}/users", timeout=10)
    resp.raise_for_status()
    return resp.json()


@app.get("/demo/history/{user_id}")
def demo_history(user_id: str):
    """재령이 백엔드의 히스토리 조회를 그대로 중계."""
    try:
        return bc.get_user_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/demo/detect/{user_id}")
def demo_detect(user_id: str):
    """재령이 백엔드의 규칙기반 감지(POST /users/{id}/detect)를 그대로 중계.

    거래내역을 스캔해서 나온 이벤트 후보(확인 질문 문구 포함)를 그대로 돌려준다.
    """
    import requests
    resp = requests.post(
        f"{bc.BACKEND_BASE_URL}/users/{user_id}/detect", params={"persist": True}, timeout=10
    )
    resp.raise_for_status()
    return resp.json()


@app.post("/demo/confirm/{event_id}")
def demo_confirm(event_id: int):
    """재령이 백엔드의 이벤트 확정(POST /events/{id}/confirm)을 그대로 중계.

    이 호출이 곧 agentic 순환의 트리거 — 확정 직후 프런트가 /demo/predict 를 이어서 부른다.
    """
    import requests
    resp = requests.post(f"{bc.BACKEND_BASE_URL}/events/{event_id}/confirm", timeout=10)
    resp.raise_for_status()
    return resp.json()


@app.post("/demo/predict")
def demo_predict(payload: PredictRequest):
    """확정된 현재 히스토리를 기준으로 검색 -> LLM 추론 -> 예측저장 -> 정책매칭까지 한 번에 실행.

    (이벤트를 새로 만들지 않는다 — /demo/confirm 으로 이미 확정된 이후에 호출하는 게 맞는 순서)
    """
    try:
        history_info = bc.get_user_history(payload.user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"유저 조회 실패: {e}")

    updated_history = history_info["confirmed_history"]

    try:
        index = bc.load_cohort_index_from_backend(_embedder)
        top_k = int(os.environ.get("COHORT_TOP_K", "15"))
        matches = index.search(updated_history, top_k=top_k)
        counts = index.aggregate_next_events(matches)
        result = _reasoner.predict(
            confirmed_history=updated_history,
            matches=matches,
            next_event_counts=counts,
            user_context=history_info.get("user_context"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 추론 실패: {e}")

    matched_cohorts_payload = [
        {"history": m.history, "next_event": m.next_event, "similarity": m.similarity}
        for m in matches
    ]
    try:
        saved = bc.save_prediction(
            user_id=payload.user_id,
            trigger_event_id=payload.trigger_event_id,
            input_history=updated_history,
            predictions=result.predictions,
            confidence_level=result.confidence_level,
            confidence_note=result.confidence_note,
            matched_cohorts=matched_cohorts_payload,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 저장 실패: {e}")

    predicted_event_types = [p["event"] for p in result.predictions]
    try:
        match_result = bc.request_policy_match(
            user_id=payload.user_id,
            prediction_id=saved["prediction_id"],
            event_types=predicted_event_types,
            include_ineligible=payload.include_ineligible,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정책매칭 실패: {e}")

    return {
        "updated_history": updated_history,
        "prediction": {
            "predictions": result.predictions,
            "confidence_level": result.confidence_level,
            "confidence_note": result.confidence_note,
        },
        "prediction_id": saved["prediction_id"],
        "policy_match": match_result,
    }
