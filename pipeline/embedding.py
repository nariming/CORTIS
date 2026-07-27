"""
이벤트 히스토리 -> 문장 인코딩 -> 임베딩 벡터

실제 서비스에서는 OpenAI text-embedding-3-small 등을 쓰지만,
API 키 없이도 로직 검증/데모가 가능하도록 오프라인 폴백(해시 기반)을 기본 제공한다.
환경변수 EMBEDDING_BACKEND=openai 로 설정하면 실제 API를 쓰도록 전환된다.
"""

import os
import hashlib
import numpy as np
from typing import List


def history_to_sentence(history: List[str]) -> str:
    """이벤트 히스토리 리스트를 하나의 문장으로 인코딩.

    예: ["대학생", "졸업", "취업"] -> "대학생 이후 졸업, 그 다음 취업을 겪은 유저"
    LLM이 문맥을 더 잘 읽도록 자연어 문장 형태로 만든다 (단순 리스트보다 임베딩 품질이 좋음).
    """
    if not history:
        return "아직 확정된 이벤트가 없는 유저"
    if len(history) == 1:
        return f"{history[0]} 이벤트만 확정된 유저"
    return " -> ".join(history) + " 순서로 이벤트를 겪은 유저"


class EmbeddingProvider:
    """임베딩 제공자 인터페이스. embed(text) -> np.ndarray 만 구현하면 교체 가능."""

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class OfflineHashEmbedding(EmbeddingProvider):
    """API 없이 개발/테스트하기 위한 결정적 해시 기반 임베딩.

    실제 의미 유사도는 반영 못하지만(진짜 서비스에는 절대 쓰면 안 됨),
    같은 입력 -> 같은 벡터를 보장하고 파이프라인 배선을 검증하는 용도로 충분.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # 토큰 단위로 해시 -> 각 토큰이 특정 차원에 가중치를 더하는 방식
        # (완전 랜덤보다는 겹치는 이벤트가 있으면 벡터도 겹치게 만들기 위함)
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.replace("->", " ").split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class OpenAIEmbedding(EmbeddingProvider):
    """실서비스용. OPENAI_API_KEY 환경변수 필요."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except ImportError:
            raise RuntimeError(
                "openai 패키지가 필요합니다: pip install openai --break-system-packages"
            )

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return np.stack([np.array(d.embedding, dtype=np.float32) for d in resp.data])


def get_embedding_provider() -> EmbeddingProvider:
    backend = os.environ.get("EMBEDDING_BACKEND", "offline")
    if backend == "openai":
        return OpenAIEmbedding()
    return OfflineHashEmbedding()
