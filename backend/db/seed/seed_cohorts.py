"""
합성 코호트 300개를 임베딩해서 MySQL cohort_sequences 테이블에 적재.

원본 데이터: data/cohort_sequences_300.py (C파트에서 generate_cohorts.py 로 생성)
여기서 하는 일은 "문장 인코딩 → 임베딩 → 벡터까지 함께 저장"이다.
벡터를 미리 저장해 두므로 서버가 뜰 때마다 300건을 재임베딩할 필요가 없고,
C파트는 CohortIndex.load_from_mysql_rows() 로 그대로 읽어 쓰면 된다.

History/State/Transaction 3분리 임베딩 (2026.8 확장): 이벤트 순서만 보던 History Embedding
외에, State/Transaction Embedding도 함께 계산해 저장한다. 문장 생성은 pipeline/state_builder.py의
state_dict_to_sentence(), pipeline/tx_features.py의 tx_features_dict_to_sentence()를 그대로
재사용한다 — 코호트 쪽과 실유저 쿼리 쪽(pipeline/similarity.py)이 같은 함수로 문장을 만들어야
두 벡터가 같은 공간에서 비교 가능하다.
"""

from sqlalchemy.orm import Session

from backend.db.models import CohortSequence
from backend.embedding_compat import get_embedder, history_to_sentence
from pipeline.state_builder import state_dict_to_sentence
from pipeline.tx_features import tx_features_dict_to_sentence


def _load_source_sequences():
    from data.cohort_sequences_300 import COHORT_SEQUENCES_300

    return COHORT_SEQUENCES_300


def seed_cohorts(db: Session) -> dict:
    sequences = _load_source_sequences()
    embedder, model_name, dim = get_embedder()

    # 같은 모델로 만든 기존 벡터는 지우고 다시 넣는다 (다른 모델 벡터는 보존)
    db.query(CohortSequence).filter(CohortSequence.embedding_model == model_name).delete(
        synchronize_session=False
    )
    db.flush()

    history_sentences = [history_to_sentence(s["history"]) for s in sequences]
    state_sentences = [state_dict_to_sentence(s["state"]) for s in sequences]
    tx_sentences = [tx_features_dict_to_sentence(s["tx_features"]) for s in sequences]

    history_vectors = embedder.embed_batch(history_sentences)
    state_vectors = embedder.embed_batch(state_sentences)
    tx_vectors = embedder.embed_batch(tx_sentences)

    for seq, h_sentence, h_vec, s_vec, t_vec in zip(
        sequences, history_sentences, history_vectors, state_vectors, tx_vectors
    ):
        db.add(
            CohortSequence(
                history_json=seq["history"],
                event_history_text=h_sentence,
                next_event=seq["next_event"],
                history_length=len(seq["history"]),
                state_json=seq["state"],
                tx_features_json=seq["tx_features"],
                event_interval_months=seq.get("event_interval_months"),
                cash_need_krw=seq.get("cash_need_krw"),
                cash_need_source=seq.get("cash_need_source"),
                embedding_vector=[float(x) for x in h_vec],
                state_embedding_vector=[float(x) for x in s_vec],
                tx_embedding_vector=[float(x) for x in t_vec],
                embedding_model=model_name,
                embedding_dim=dim,
            )
        )

    db.flush()
    return {"cohorts": len(sequences), "embedding_model": model_name, "dim": dim}