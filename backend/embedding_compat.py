"""
임베딩 호환 레이어 (개발자1 ↔ 개발자2 연결부).

코호트를 MySQL에 적재할 때 쓴 임베딩과 C파트가 검색할 때 쓰는 임베딩이 반드시 같아야
코사인 유사도가 의미를 갖는다. 그래서 이 모듈은 **C파트(pipeline/embedding.py)의 구현을 우선 임포트**하고,
아직 그 브랜치가 머지되지 않았을 때만 동일 로직의 폴백을 쓴다.

폴백 구현은 pipeline/embedding.py 의 OfflineHashEmbedding / history_to_sentence 와
같은 알고리즘이어야 한다. (둘 중 하나를 고치면 다른 쪽도 같이 고칠 것)
"""

import hashlib
from typing import List

import numpy as np

from backend import config

try:  # C파트 브랜치가 머지된 뒤에는 이쪽이 쓰인다 (원본 1개 유지)
    from pipeline.embedding import history_to_sentence, OfflineHashEmbedding  # type: ignore

    USING_PIPELINE_IMPL = True

except ImportError:  # 아직 머지 전이면 동일 로직 폴백
    USING_PIPELINE_IMPL = False

    def history_to_sentence(history: List[str]) -> str:
        if not history:
            return "아직 확정된 이벤트가 없는 유저"
        if len(history) == 1:
            return f"{history[0]} 이벤트만 확정된 유저"
        return " -> ".join(history) + " 순서로 이벤트를 겪은 유저"

    class OfflineHashEmbedding:
        """API 키 없이 파이프라인을 검증하기 위한 결정적 해시 임베딩."""

        def __init__(self, dim: int = 64):
            self.dim = dim

        def embed(self, text: str) -> np.ndarray:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in text.replace("->", " ").split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        def embed_batch(self, texts: List[str]) -> np.ndarray:
            return np.stack([self.embed(t) for t in texts])


def get_embedder():
    """config.EMBEDDING_BACKEND 에 따라 임베더와 모델 식별자를 함께 돌려준다.

    모델 식별자를 cohort_sequences.embedding_model 에 같이 저장해 두면,
    나중에 백엔드를 offline -> openai 로 바꿨을 때 "옛날 벡터와 섞여서 유사도가 이상해지는" 사고를 막을 수 있다.
    """
    if config.EMBEDDING_BACKEND == "openai":
        from pipeline.embedding import OpenAIEmbedding  # type: ignore

        model = "text-embedding-3-small"
        return OpenAIEmbedding(model=model), model, 1536

    dim = config.EMBEDDING_DIM
    return OfflineHashEmbedding(dim=dim), f"offline-hash-{dim}", dim
