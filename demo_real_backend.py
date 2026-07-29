"""
실제 백엔드(재령이의 FastAPI + MySQL)에서 코호트 데이터를 가져와서 돌리는 데모.

사전 조건:
1. MySQL 켜져 있고 시드 완료 (python -m backend.db.seed.run_all 실행됨)
2. 서버 켜져 있음: uvicorn backend.main:app --reload (다른 터미널 창에서)

실행: python demo_real_backend.py
"""

from dotenv import load_dotenv
load_dotenv()
load_dotenv("backend/.env")

import os
from pipeline.embedding import get_embedding_provider
from pipeline.backend_client import load_cohort_index_from_backend
from pipeline.reasoning import get_reasoner
from pipeline.agent_loop import CortisAgent, AgentState
from data.dummy_cohorts import DEMO_USER_KIMHANEUL, DEMO_USER_CONTRAST


def run():
    embedder = get_embedding_provider()

    print("백엔드(GET /cohorts)에서 실제 코호트 데이터 가져오는 중...")
    index = load_cohort_index_from_backend(embedder)
    print("코호트 로드 완료.\n")

    reasoner = get_reasoner()
    agent = CortisAgent(index, reasoner, top_k=int(os.environ.get("COHORT_TOP_K", "15")))

    for demo_user in (DEMO_USER_KIMHANEUL, DEMO_USER_CONTRAST):
        print(f"\n{'=' * 60}")
        print(f"유저: {demo_user['user_id']}")
        print(f"확정 히스토리: {' -> '.join(demo_user['confirmed_history'])}")
        print(f"{'=' * 60}")

        state = AgentState(
            user_id=demo_user["user_id"],
            confirmed_history=demo_user["confirmed_history"][:-1],
        )
        last_event = demo_user["confirmed_history"][-1]

        result = agent.on_event_confirmed(state, last_event)

        print(f"\n[예측 결과] confidence={result.confidence_level} ({result.confidence_note})")
        for p in result.predictions:
            print(f"  - {p['event']} (근거 {p['evidence_count']}건): {p['reasoning']}")


if __name__ == "__main__":
    run()
