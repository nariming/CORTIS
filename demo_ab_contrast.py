"""
심사용 핵심 데모: 같은 '취업' 이벤트를 겪은 유저 A(대학생->졸업->취업)와
유저 B(이직->취업)가 서로 다른 다음 이벤트를 예측받는 걸 보여준다.

이게 "고정 전이확률표가 아니라 개인 히스토리에 조건화된 진짜 추론"이라는 증거.

실행: python demo_ab_contrast.py
(API 키 없이 MockReasoner로 바로 실행 가능. 실제 LLM 근거 문장을 보려면
 ANTHROPIC_API_KEY 설정 후 LLM_BACKEND=anthropic python demo_ab_contrast.py)
"""

from dotenv import load_dotenv
load_dotenv()  # 최상위 .env: ANTHROPIC_API_KEY, LLM_BACKEND
load_dotenv("backend/.env")  # backend/.env: COHORT_TOP_K, COLD_START_THRESHOLD (Backend 설정과 통일)

import os
from pipeline.embedding import get_embedding_provider
from pipeline.similarity import CohortIndex
from pipeline.reasoning import get_reasoner
from pipeline.agent_loop import CortisAgent, AgentState
from data.cohort_sequences_300 import COHORT_SEQUENCES_300 as COHORT_SEQUENCES
from data.dummy_cohorts import DEMO_USER_KIMHANEUL, DEMO_USER_CONTRAST


def run():
    embedder = get_embedding_provider()
    index = CohortIndex(embedder)
    index.build_from_sequences(COHORT_SEQUENCES)

    reasoner = get_reasoner()
    agent = CortisAgent(index, reasoner, top_k=int(os.environ.get("COHORT_TOP_K", "15")))

    for demo_user in (DEMO_USER_KIMHANEUL, DEMO_USER_CONTRAST):
        print(f"\n{'=' * 60}")
        print(f"유저: {demo_user['user_id']}")
        print(f"확정 히스토리: {' -> '.join(demo_user['confirmed_history'])}")
        print(f"{'=' * 60}")

        state = AgentState(
            user_id=demo_user["user_id"],
            confirmed_history=demo_user["confirmed_history"][:-1],  # 마지막 이벤트는 지금 막 확정되는 걸로
        )
        last_event = demo_user["confirmed_history"][-1]

        result = agent.on_event_confirmed(state, last_event)

        print(f"\n[예측 결과] confidence={result.confidence_level} ({result.confidence_note})")
        for p in result.predictions:
            print(f"  - {p['event']} (근거 {p['evidence_count']}건): {p['reasoning']}")


if __name__ == "__main__":
    run()