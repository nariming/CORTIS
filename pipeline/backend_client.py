"""
재령이의 FastAPI 백엔드(GET /cohorts)에서 실제 코호트 데이터를 가져와
CohortIndex에 로드하는 연결 모듈.

사용법 (데모 스크립트에서):
    from pipeline.backend_client import load_cohort_index_from_backend
    index = load_cohort_index_from_backend(embedder)
"""

import requests
from pipeline.embedding import EmbeddingProvider
from pipeline.similarity import CohortIndex

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
