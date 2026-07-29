"""
numpy 코사인 유사도 기반 유사 코호트 검색.

ChromaDB 등 전용 벡터DB 대신, 코호트 규모(수백 건)에서는
전체 벡터를 메모리에 올려두고 코사인 유사도로 top-k를 뽑는 것만으로 충분하다.
저장 자체는 MySQL이 source of truth이고, 이 모듈은 "검색 인덱스" 역할만 한다.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from .embedding import EmbeddingProvider, history_to_sentence


@dataclass
class CohortMatch:
    history: List[str]
    next_event: str
    similarity: float


class CohortIndex:
    """MySQL에서 로드한 코호트 시퀀스 + 임베딩 벡터를 메모리에 올려두고 검색하는 인덱스.

    실제 배선: 개발자1의 MySQL cohort_sequences 테이블에서
    (event_history_text, embedding_vector, next_event)를 SELECT 해서 이 클래스에 로드.
    지금은 dummy_cohorts.py의 더미 데이터로 빌드하는 함수를 함께 제공.
    """

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self._histories: List[List[str]] = []
        self._next_events: List[str] = []
        self._vectors: np.ndarray | None = None

    def build_from_sequences(self, sequences: List[dict]):
        """sequences: [{"history": [...], "next_event": "..."}]"""
        sentences = [history_to_sentence(s["history"]) for s in sequences]
        self._histories = [s["history"] for s in sequences]
        self._next_events = [s["next_event"] for s in sequences]
        self._vectors = self.embedder.embed_batch(sentences)
        # 코사인 유사도 계산 편하게 미리 정규화
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = self._vectors / norms

    def load_from_mysql_rows(self, rows: List[dict]):
        """개발자1 스키마 연결 시 사용할 진입점.

        rows 예시: [{"event_history_text": "취업 -> 독립(월세)",
                     "embedding_vector": [...],  # MySQL에 JSON/BLOB으로 저장된 벡터
                     "next_event": "결혼"}, ...]
        임베딩을 다시 계산할 필요 없이 저장된 벡터를 그대로 로드한다.
        """
        self._histories = [r.get("history", []) for r in rows]
        self._next_events = [r["next_event"] for r in rows]
        vecs = np.array([r["embedding_vector"] for r in rows], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vecs / norms

    def search(self, query_history: List[str], top_k: int = 5) -> List[CohortMatch]:
        if self._vectors is None or len(self._vectors) == 0:
            return []
        query_sentence = history_to_sentence(query_history)
        q_vec = self.embedder.embed(query_sentence)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        sims = self._vectors @ q_vec  # 코사인 유사도 (이미 정규화됨)
        top_k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:top_k]

        return [
            CohortMatch(
                history=self._histories[i],
                next_event=self._next_events[i],
                similarity=float(sims[i]),
            )
            for i in top_idx
        ]

    def aggregate_next_events(self, matches: List[CohortMatch]) -> dict:
        """검색된 top-k 중 다음 이벤트 분포 집계. {"독립(월세)": 3, "결혼": 2} 형태.

        이 집계값이 바로 "검증 가능한 근거 숫자" — LLM이 지어낸 확률이 아니라
        실제 검색 결과를 세어서 만든 값이라 심사 방어가 쉽다.
        """
        counts: dict = {}
        for m in matches:
            counts[m.next_event] = counts.get(m.next_event, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
