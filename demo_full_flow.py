"""
진짜 API로 전체 Agentic 순환을 실행하는 데모.

흐름: POST /events(이벤트 확정) -> 코호트 검색+LLM 예측(로컬) ->
      POST /predictions(저장) -> POST /policy-match(A파트 실제 호출)

사전 조건: uvicorn backend.main:app --reload 서버가 켜져 있어야 함.

실행: python demo_full_flow.py
"""

from dotenv import load_dotenv
load_dotenv()

from dataclasses import asdict

from pipeline.embedding import get_embedding_provider
from pipeline.reasoning import get_reasoner
from pipeline import backend_client as bc

USER_ID = "U_A"  # 김하늘
NEW_EVENT = "취업"


def run():
    embedder = get_embedding_provider()
    reasoner = get_reasoner()

    # 1) 유저의 현재 히스토리 확인
    history_info = bc.get_user_history(USER_ID)
    print(f"[1] 현재 확정 히스토리: {history_info['confirmed_history']}")

    # 2) 새 이벤트를 실제로 '확정' 상태로 등록
    event = bc.create_life_event(USER_ID, NEW_EVENT)
    print(f"[2] 이벤트 등록 완료: event_id={event['event_id']}, event_type={event['event_type']}")

    updated_history = history_info["confirmed_history"] + [NEW_EVENT]

    # 3) 코호트 검색 + LLM 추론 (로컬 파이프라인, 이미 만들어둔 것 재사용)
    index = bc.load_cohort_index_from_backend(embedder)
    matches = index.search(updated_history, top_k=15)
    counts = index.aggregate_next_events(matches)
    result = reasoner.predict(
        confirmed_history=updated_history,
        matches=matches,
        next_event_counts=counts,
        user_context=history_info.get("user_context"),
    )
    print(f"[3] 예측 완료: confidence={result.confidence_level}")
    for p in result.predictions:
        print(f"    - {p['event']} (근거 {p['evidence_count']}건)")

    # 4) 예측 결과를 실제로 저장 (trigger_event_id는 방금 확정한 이벤트)
    matched_cohorts_payload = [
        {"history": m.history, "next_event": m.next_event, "similarity": m.similarity}
        for m in matches
    ]
    saved = bc.save_prediction(
        user_id=USER_ID,
        trigger_event_id=event["event_id"],
        input_history=updated_history,
        predictions=result.predictions,
        confidence_level=result.confidence_level,
        confidence_note=result.confidence_note,
        matched_cohorts=matched_cohorts_payload,
    )
    print(f"[4] 예측 저장 완료: prediction_id={saved['prediction_id']}")

    # 5) A파트에 실제로 정책 매칭 요청
    predicted_event_types = [p["event"] for p in result.predictions]
    match_result = bc.request_policy_match(
        user_id=USER_ID,
        prediction_id=saved["prediction_id"],
        event_types=predicted_event_types,
    )
    print(f"[5] A파트 정책매칭 요청 완료. 응답:")
    print(match_result)


if __name__ == "__main__":
    run()
