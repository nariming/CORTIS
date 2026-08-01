"""
numpy 코사인 유사도 기반 유사 코호트 검색.

ChromaDB 등 전용 벡터DB 대신, 코호트 규모(수백 건)에서는
전체 벡터를 메모리에 올려두고 코사인 유사도로 top-k를 뽑는 것만으로 충분하다.
저장 자체는 MySQL이 source of truth이고, 이 모듈은 "검색 인덱스" 역할만 한다.

History/State/Transaction 3분리 임베딩 (2026.8 확장)
  - 과거에는 이벤트 히스토리 문장 하나만 임베딩해서 검색했다. 그러면 "취업→독립" 이벤트
    순서가 같은 두 사람은, 현재 소득·주거·신용 상태가 완전히 달라도 거의 같은 코호트로
    검색됐다 — 즉 State가 검색 단계에서 아예 반영되지 않는 문제가 있었다.
  - 이를 해결하기 위해 History/State/Transaction을 하나로 합치지 않고 3개의 독립된
    임베딩 공간으로 분리했다. 합치지 않는 이유: State만 강조하면 이 서비스의 원래
    차별점("이벤트 순서에 따른 조건부 추론")이 희석될 수 있기 때문이다.
  - 세 공간의 유사도는 가중합으로 결합한다. 가중치는 새 하이퍼파라미터를 만들지 않고
    기존 COLD_START_THRESHOLD를 그대로 재사용한다 — 확정 히스토리가 짧을 때
    (콜드스타트) State/Transaction 신호에 더 의존해야 한다는 것은 이미 있던 콜드스타트
    개념과 정확히 같은 맥락이라는 판단.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .embedding import EmbeddingProvider, history_to_sentence
from .state_builder import state_dict_to_sentence
from .tx_features import tx_features_dict_to_sentence

# backend/.env 의 COLD_START_THRESHOLD 와 반드시 같은 값을 참조해야 한다.
# (reasoning.py가 이미 같은 패턴으로 이 값을 읽고 있음 — 두 곳이 서로 다른 값을 읽으면
# "콜드스타트 판단"과 "검색 가중치"가 서로 다른 기준을 쓰게 되어 방어 논리가 어긋난다)
COLD_START_THRESHOLD = int(os.environ.get("COLD_START_THRESHOLD", "2"))

# 히스토리가 쌓였을 때(콜드스타트 아님) vs 짧을 때(콜드스타트)의 가중치.
# 이분법인 이유: len(history)에 비례해 부드럽게 증가시키는 것도 가능하나, 이미 있는
# 콜드스타트 임계값을 그대로 재사용하는 이분법이 "왜 이 가중치냐"는 질문에
# "기존 콜드스타트 기준을 검색 가중치에도 그대로 적용했다"고 답할 수 있어 방어가 더 쉽다.
WEIGHTS_COLD_START = {"history": 0.2, "state": 0.6, "tx": 0.2}
WEIGHTS_NORMAL = {"history": 0.6, "state": 0.25, "tx": 0.15}


def _adaptive_weights(history_len: int, cold_start_threshold: int = COLD_START_THRESHOLD) -> dict:
    return WEIGHTS_COLD_START if history_len < cold_start_threshold else WEIGHTS_NORMAL


@dataclass
class CohortMatch:
    history: List[str]
    next_event: str
    similarity: float  # 3개 공간 가중합 최종 유사도 (기존 호출부와의 하위 호환을 위해 필드명 유지)
    sim_history: float = 0.0
    sim_state: float = 0.0
    sim_tx: float = 0.0
    cash_need_krw: Optional[int] = None
    cash_need_source: Optional[str] = None
    event_interval_months: Optional[int] = None


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class CohortIndex:
    """MySQL에서 로드한 코호트 시퀀스 + History/State/Transaction 3종 임베딩 벡터를
    메모리에 올려두고 검색하는 인덱스.

    실제 배선: 개발자1의 MySQL cohort_sequences 테이블에서
    (event_history_text, embedding_vector, state_embedding_vector, tx_embedding_vector,
     next_event, cash_need_krw, cash_need_source, event_interval_months)를 SELECT 해서
    이 클래스에 로드.
    """

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self._histories: List[List[str]] = []
        self._next_events: List[str] = []
        self._cash_need_krw: List[Optional[int]] = []
        self._cash_need_source: List[Optional[str]] = []
        self._event_interval_months: List[Optional[int]] = []
        self._history_vectors: Optional[np.ndarray] = None
        self._state_vectors: Optional[np.ndarray] = None
        self._tx_vectors: Optional[np.ndarray] = None

    def build_from_sequences(self, sequences: List[dict]):
        """sequences: generate_cohorts.py 산출물 형태
        [{"history": [...], "next_event": "...", "state": {...}, "tx_features": {...},
          "cash_need_krw": int|None, "cash_need_source": str|None,
          "event_interval_months": int|None}, ...]
        """
        history_sentences = [history_to_sentence(s["history"]) for s in sequences]
        state_sentences = [state_dict_to_sentence(s["state"]) for s in sequences]
        tx_sentences = [tx_features_dict_to_sentence(s["tx_features"]) for s in sequences]

        self._histories = [s["history"] for s in sequences]
        self._next_events = [s["next_event"] for s in sequences]
        self._cash_need_krw = [s.get("cash_need_krw") for s in sequences]
        self._cash_need_source = [s.get("cash_need_source") for s in sequences]
        self._event_interval_months = [s.get("event_interval_months") for s in sequences]

        self._history_vectors = _normalize_rows(self.embedder.embed_batch(history_sentences))
        self._state_vectors = _normalize_rows(self.embedder.embed_batch(state_sentences))
        self._tx_vectors = _normalize_rows(self.embedder.embed_batch(tx_sentences))

    def load_from_mysql_rows(self, rows: List[dict]):
        """개발자1 스키마 연결 시 사용할 진입점.

        rows 예시: [{"history_json": [...], "next_event": "결혼",
                     "embedding_vector": [...], "state_embedding_vector": [...],
                     "tx_embedding_vector": [...], "cash_need_krw": 68000000,
                     "cash_need_source": "...", "event_interval_months": 24}, ...]
        임베딩을 다시 계산할 필요 없이 저장된 벡터를 그대로 로드한다.
        """
        self._histories = [r.get("history_json", r.get("history", [])) for r in rows]
        self._next_events = [r["next_event"] for r in rows]
        self._cash_need_krw = [r.get("cash_need_krw") for r in rows]
        self._cash_need_source = [r.get("cash_need_source") for r in rows]
        self._event_interval_months = [r.get("event_interval_months") for r in rows]

        self._history_vectors = _normalize_rows(
            np.array([r["embedding_vector"] for r in rows], dtype=np.float32)
        )
        self._state_vectors = _normalize_rows(
            np.array([r["state_embedding_vector"] for r in rows], dtype=np.float32)
        )
        self._tx_vectors = _normalize_rows(
            np.array([r["tx_embedding_vector"] for r in rows], dtype=np.float32)
        )

    def _embed_query(self, text: str) -> np.ndarray:
        vec = self.embedder.embed(text)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search(
        self,
        query_history: List[str],
        query_state: Optional[dict] = None,
        query_tx: Optional[dict] = None,
        top_k: int = 5,
    ) -> List[CohortMatch]:
        """History/State/Transaction 3개 공간의 코사인 유사도를 가중합해 top-k를 뽑는다.

        query_state/query_tx가 없으면(예: 아직 State Builder 연결 전 호출부) 해당 공간은
        검색에서 제외하고 남은 공간끼리 가중치를 재분배한다 — 값이 없다고 0점 처리하면
        "그 유저는 상태가 전혀 안 맞는 사람"으로 취급하는 셈이 되어 왜곡이 생기기 때문이다.
        """
        if self._history_vectors is None or len(self._history_vectors) == 0:
            return []

        weights = dict(_adaptive_weights(len(query_history)))
        sim_history = self._history_vectors @ self._embed_query(history_to_sentence(query_history))

        if query_state is not None and self._state_vectors is not None:
            sim_state = self._state_vectors @ self._embed_query(state_dict_to_sentence(query_state))
        else:
            sim_state = np.zeros_like(sim_history)
            weights["state"] = 0.0

        if query_tx is not None and self._tx_vectors is not None:
            sim_tx = self._tx_vectors @ self._embed_query(tx_features_dict_to_sentence(query_tx))
        else:
            sim_tx = np.zeros_like(sim_history)
            weights["tx"] = 0.0

        weight_sum = sum(weights.values()) or 1.0
        weights = {k: v / weight_sum for k, v in weights.items()}

        sim_total = (
            weights["history"] * sim_history
            + weights["state"] * sim_state
            + weights["tx"] * sim_tx
        )

        top_k = min(top_k, len(sim_total))
        top_idx = np.argsort(-sim_total)[:top_k]

        return [
            CohortMatch(
                history=self._histories[i],
                next_event=self._next_events[i],
                similarity=float(sim_total[i]),
                sim_history=float(sim_history[i]),
                sim_state=float(sim_state[i]),
                sim_tx=float(sim_tx[i]),
                cash_need_krw=self._cash_need_krw[i],
                cash_need_source=self._cash_need_source[i],
                event_interval_months=self._event_interval_months[i],
            )
            for i in top_idx
        ]

    def aggregate_next_events(self, matches: List[CohortMatch]) -> dict:
        """검색된 top-k 중 다음 이벤트 분포 집계.

        반환: {event: {"count": int, "weighted_score": float}}
        count만으로는 유사도 0.95인 사례 3건과 0.3인 사례 3건을 구분 못 한다. weighted_score는
        해당 이벤트로 매칭된 코호트들의 유사도 합이라, "몇 건인데 유사도는 이 정도"라는 근거를
        함께 제시할 수 있다. 정렬 기준도 count가 아니라 weighted_score로 바꿨다 — 이 집계의
        원래 취지(유사도가 높은 근거를 우선한다)에 더 맞기 때문이다.
        """
        agg: dict = {}
        for m in matches:
            entry = agg.setdefault(m.next_event, {"count": 0, "weighted_score": 0.0})
            entry["count"] += 1
            entry["weighted_score"] += m.similarity
        for entry in agg.values():
            entry["weighted_score"] = round(entry["weighted_score"], 4)
        return dict(sorted(agg.items(), key=lambda kv: -kv[1]["weighted_score"]))