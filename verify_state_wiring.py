"""
agent_loop.py의 query_state/query_tx 자동조회가 실제 서버에서 작동하는지 검증하는 스크립트.
서버(uvicorn backend.main:app --reload)를 먼저 켜둔 상태에서 실행할 것.

실행: python verify_state_wiring.py
"""

from dotenv import load_dotenv
load_dotenv()
load_dotenv("backend/.env")

from pipeline.embedding import get_embedding_provider
from pipeline.reasoning import get_reasoner
from pipeline.agent_loop import CortisAgent, AgentState
from pipeline import backend_client as bc

embedder = get_embedding_provider()
index = bc.load_cohort_index_from_backend(embedder)
reasoner = get_reasoner()
agent = CortisAgent(index, reasoner, top_k=15)

# 실제 DB에 있는 U_A로 테스트 (김하늘 페르소나의 진짜 user_id)
state = AgentState(user_id="U_A", confirmed_history=["대학생", "졸업"])
result = agent.on_event_confirmed(state, "취업")

print()
print("=== query_state/query_tx가 실제로 채워졌는지 확인 ===")
print("query_state:", state.query_state)
print("query_tx   :", state.query_tx)

print()
print("=== 검색 결과에 sim_state/sim_tx가 0이 아닌 값이 있는지 확인 ===")
matches = index.search(state.confirmed_history, query_state=state.query_state, query_tx=state.query_tx, top_k=5)
for m in matches:
    print(f"  {m.next_event:10s} sim_total={m.similarity:.3f} (history={m.sim_history:.2f}, state={m.sim_state:.2f}, tx={m.sim_tx:.2f})")