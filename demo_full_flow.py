"""
진짜 API로 전체 Agentic 순환을 실행하는 데모.

흐름: POST /events(이벤트 확정) -> 코호트 검색+LLM 예측(로컬) ->
      POST /predictions(저장) -> POST /policy-match(A파트 실제 호출)

사전 조건: uvicorn backend.main:app --reload 서버가 켜져 있어야 함.

실행: python demo_full_flow.py
"""

from dotenv import load_dotenv
load_dotenv()
load_dotenv("backend/.env")

import os
import sys
import requests
from dataclasses import asdict

from pipeline.embedding import get_embedding_provider
from pipeline.reasoning import get_reasoner
from pipeline import backend_client as bc

# 명령줄 인자로 유저/이벤트/불합격정책포함여부 지정 가능: python demo_full_flow.py U_B 이직 true
USER_ID = sys.argv[1] if len(sys.argv) > 1 else "U_A"
NEW_EVENT = sys.argv[2] if len(sys.argv) > 2 else "취업"
INCLUDE_INELIGIBLE = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else False


def run():
    embedder = get_embedding_provider()
    reasoner = get_reasoner()

    # 1) 유저의 현재 히스토리 확인
    try:
        history_info = bc.get_user_history(USER_ID)
    except requests.exceptions.ConnectionError:
        print("[에러] 서버에 연결할 수 없습니다. 'uvicorn backend.main:app --reload'가 켜져 있는지 확인하세요.")
        return
    except requests.exceptions.HTTPError as e:
        print(f"[에러] 유저 히스토리 조회 실패 (user_id='{USER_ID}'가 존재하지 않을 수 있음): {e}")
        return
    print(f"[1] 현재 확정 히스토리: {history_info['confirmed_history']}")

    # 2) 새 이벤트를 실제로 '확정' 상태로 등록
    try:
        event = bc.create_life_event(USER_ID, NEW_EVENT)
    except requests.exceptions.HTTPError as e:
        print(f"[에러] 이벤트 등록 실패: {e}")
        return
    print(f"[2] 이벤트 등록 완료: event_id={event['event_id']}, event_type={event['event_type']}")

    updated_history = history_info["confirmed_history"] + [NEW_EVENT]

    # 3) 코호트 검색 + LLM 추론 (로컬 파이프라인, 이미 만들어둔 것 재사용)
    try:
        index = bc.load_cohort_index_from_backend(embedder)
        matches = index.search(updated_history, top_k=int(os.environ.get("COHORT_TOP_K", "15")))
        counts = index.aggregate_next_events(matches)
        result = reasoner.predict(
            confirmed_history=updated_history,
            matches=matches,
            next_event_counts=counts,
            user_context=history_info.get("user_context"),
        )
    except Exception as e:
        print(f"[에러] 코호트 검색/LLM 추론 단계 실패: {e}")
        print("  (LLM_BACKEND=anthropic 이면 ANTHROPIC_API_KEY와 크레딧 잔액을 확인하세요)")
        return
    print(f"[3] 예측 완료: confidence={result.confidence_level}")
    for p in result.predictions:
        timing = f"{p['expected_timing_months']:.0f}개월 후" if p['expected_timing_months'] is not None else "시점 추정 불가"
        cash = f"약 {p['expected_cash_need_krw']:,}원" if p['expected_cash_need_krw'] is not None else "해당 없음"
        print(f"    - {p['event']}: 확률 {p['probability_pct']}% / 예상시점 {timing} / 예상필요자금 {cash} (근거 {p['evidence_count']}건)")

    # 4) 예측 결과를 실제로 저장 (trigger_event_id는 방금 확정한 이벤트)
    matched_cohorts_payload = [
        {"history": m.history, "next_event": m.next_event, "similarity": m.similarity}
        for m in matches
    ]
    try:
        saved = bc.save_prediction(
            user_id=USER_ID,
            trigger_event_id=event["event_id"],
            input_history=updated_history,
            predictions=result.predictions,
            confidence_level=result.confidence_level,
            confidence_note=result.confidence_note,
            matched_cohorts=matched_cohorts_payload,
        )
    except requests.exceptions.HTTPError as e:
        print(f"[에러] 예측 저장 실패: {e}")
        return
    print(f"[4] 예측 저장 완료: prediction_id={saved['prediction_id']}")

    # 5) A파트에 실제로 정책 매칭 요청
    predicted_event_types = [p["event"] for p in result.predictions]
    try:
        match_result = bc.request_policy_match(
            user_id=USER_ID,
            prediction_id=saved["prediction_id"],
            event_types=predicted_event_types,
            include_ineligible=INCLUDE_INELIGIBLE,
        )
    except requests.exceptions.HTTPError as e:
        print(f"[에러] A파트 정책매칭 요청 실패: {e}")
        return
    print(f"[5] A파트 정책매칭 요청 완료. 응답:")
    print(match_result)


if __name__ == "__main__":
    run()