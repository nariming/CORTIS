"""
포트폴리오 결정 레이어까지 실제 API로 끝까지 도는 데모.

demo_full_flow.py의 [1]~[5](이벤트확정 -> 예측 -> A파트 정책매칭)에 이어서,
[6] A파트 대출상품 매칭(loan-match) -> [7] 포트폴리오 선택(NPV/DSR/우선순위)
-> [8] LLM 요약(검증가드 포함) -> [9] 감시조건까지 확인한다.

사전 조건: uvicorn backend.main:app --reload 서버가 켜져 있어야 함.
대출 데이터가 있는 유저로 테스트해야 의미있는 결과가 나옴 (기본값: U_B 박지훈).

실행: python demo_portfolio_flow.py
      python demo_portfolio_flow.py U_B 독립(월세)
"""

from dotenv import load_dotenv
load_dotenv()
load_dotenv("backend/.env")

import sys
import requests

from pipeline.embedding import get_embedding_provider
from pipeline.reasoning import get_reasoner
from pipeline import backend_client as bc
from pipeline.portfolio import select_best_portfolio, build_watch_conditions
from pipeline.portfolio_summary import build_summary_input, get_summarizer, validate_summary_numbers

USER_ID = sys.argv[1] if len(sys.argv) > 1 else "U_B"
NEW_EVENT = sys.argv[2] if len(sys.argv) > 2 else "독립(월세)"


def run():
    embedder = get_embedding_provider()
    reasoner = get_reasoner()

    try:
        history_info = bc.get_user_history(USER_ID)
    except requests.exceptions.ConnectionError:
        print("[에러] 서버에 연결할 수 없습니다. 'uvicorn backend.main:app --reload'가 켜져 있는지 확인하세요.")
        return
    except requests.exceptions.HTTPError as e:
        print(f"[에러] 유저 히스토리 조회 실패: {e}")
        return
    print(f"[1] 현재 확정 히스토리: {history_info['confirmed_history']}")

    event = bc.create_life_event(USER_ID, NEW_EVENT)
    print(f"[2] 이벤트 등록 완료: event_id={event['event_id']}, event_type={event['event_type']}")
    updated_history = history_info["confirmed_history"] + [NEW_EVENT]

    index = bc.load_cohort_index_from_backend(embedder)
    matches = index.search(updated_history, top_k=15)
    counts = index.aggregate_next_events(matches)
    result = reasoner.predict(
        confirmed_history=updated_history, matches=matches,
        next_event_counts=counts, user_context=history_info.get("user_context"),
    )
    print(f"[3] 예측 완료: confidence={result.confidence_level}")
    for p in result.predictions:
        print(f"    - {p['event']} (근거 {p['evidence_count']}건)")

    matched_cohorts_payload = [
        {"history": m.history, "next_event": m.next_event, "similarity": m.similarity} for m in matches
    ]
    saved = bc.save_prediction(
        user_id=USER_ID, trigger_event_id=event["event_id"], input_history=updated_history,
        predictions=result.predictions, confidence_level=result.confidence_level,
        confidence_note=result.confidence_note, matched_cohorts=matched_cohorts_payload,
    )
    print(f"[4] 예측 저장 완료: prediction_id={saved['prediction_id']}")

    # --- 여기부터 신규: 대출상품 매칭 -> 포트폴리오 결정 -> LLM 요약 ---

    user_detail = bc.fetch_user_detail(USER_ID)
    profile = bc.build_profile_from_user_detail(user_detail)
    loans = bc.build_existing_loans_from_user_detail(user_detail)

    if not loans:
        print(f"[중단] {USER_ID}는 보유 대출이 없어 포트폴리오 재설계 대상이 아닙니다.")
        return

    try:
        loan_match_response = bc.request_loan_match(USER_ID, event_types=[NEW_EVENT])
    except requests.exceptions.HTTPError as e:
        print(f"[에러] A파트 대출상품 매칭 요청 실패: {e}")
        return
    loan_refinance_map = bc.build_refinance_map_from_loan_match(loans, loan_match_response)
    print(f"[5] A파트 대출상품(KB자체) 매칭 완료: {len(loan_match_response)}개 이벤트 그룹")

    # [신규] 정책형 대출(버팀목전세자금대출 등)도 대환 후보로 함께 조회
    try:
        policy_match_response = bc.request_policy_match(
            user_id=USER_ID, prediction_id=saved["prediction_id"], event_types=[NEW_EVENT], persist=False,
        )
        policy_catalog = bc.fetch_policy_catalog()
        policy_refinance_map = bc.build_refinance_map_from_policy_match(loans, policy_match_response, policy_catalog)
        n_policy_candidates = sum(len(v) for v in policy_refinance_map.values())
        print(f"[5-1] 정책형 대출 후보 {n_policy_candidates}건 추가 발견")
    except requests.exceptions.HTTPError as e:
        print(f"[경고] 정책형 대출 후보 조회 실패, KB자체상품만으로 진행: {e}")
        policy_refinance_map = {}

    refinance_map = bc.merge_refinance_maps(loan_refinance_map, policy_refinance_map)

    best, second, infeasible = select_best_portfolio(loans, refinance_map, profile)
    if best is None:
        print(f"[중단] 실행가능한 포트폴리오 없음. 탈락 사유: {infeasible}")
        return

    print(f"[6] 포트폴리오 선택 완료:")
    for a in best.combo:
        action_desc = a.action
        if a.action == "REFINANCE" and a.target_product:
            action_desc += f" -> {a.target_product.product_name}"
        print(f"    - {a.loan_id}: {action_desc}")
    print(f"    월상환액: {best.monthly_payment_total:,}원 / DSR: {best.dsr_pct}% (스트레스 {best.dsr_stress_pct}%)")
    print(f"    KEEP 대비 NPV 절감액: {best.npv_savings_vs_keep:,}원")

    dsr_before = round((sum(l.monthly_payment for l in loans) * 12 / profile.annual_income) * 100, 1)
    summary_input = build_summary_input(
        user_name=user_detail.get("name", USER_ID), event_name=NEW_EVENT,
        loans=loans, profile=profile, best=best, second=second, dsr_before=dsr_before,
        next_action_hint="정확한 신청 방법·필요서류는 해당 상품 안내 페이지에서 최종 확인 필요",
    )

    summarizer = get_summarizer()
    summary = summarizer.summarize(summary_input)
    validated, bad_numbers = validate_summary_numbers(summary, summary_input.allowed_numbers())

    print(f"\n[7] LLM 요약 (검증가드 통과: {validated}):")
    for k, v in summary.items():
        print(f"    [{k}] {v}")
    if not validated:
        print(f"    [경고] 허용 안 된 숫자 발견: {bad_numbers}")

    watch_conditions = build_watch_conditions(best, NEW_EVENT, event_confirmed=True)
    print(f"\n[8] 감시조건 (재검토 트리거):")
    for c in watch_conditions:
        print(f"    - {c}")


if __name__ == "__main__":
    run()