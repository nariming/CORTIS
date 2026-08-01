"""
합성 코호트 300개를 임베딩해서 MySQL cohort_sequences 테이블에 적재.

원본 데이터: data/cohort_sequences_300.py (C파트에서 generate_cohorts.py 로 생성)
여기서 하는 일은 "문장 인코딩 → 임베딩 → 벡터까지 함께 저장"이다.
벡터를 미리 저장해 두므로 서버가 뜰 때마다 300건을 재임베딩할 필요가 없고,
C파트는 CohortIndex.load_from_mysql_rows() 로 그대로 읽어 쓰면 된다.

state_json/tx_features_json/event_interval_months/cash_need_krw는 generate_cohorts.py가
합성한 값을 그대로 옮겨 적재한다. State Embedding/Transaction Embedding 벡터를 만드는 작업은
history 임베딩과 임베딩 모델/문장 구성 방식을 함께 결정해야 하는 별도 단계(embedding.py 확장)에서
다루므로, 이 단계에서는 원본 값 저장까지만 담당한다.
"""

from sqlalchemy.orm import Session

from backend.db.models import CohortSequence
from backend.embedding_compat import get_embedder, history_to_sentence


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

    sentences = [history_to_sentence(s["history"]) for s in sequences]
    vectors = embedder.embed_batch(sentences)

    for seq, sentence, vec in zip(sequences, sentences, vectors):
        db.add(
            CohortSequence(
                history_json=seq["history"],
                event_history_text=sentence,
                next_event=seq["next_event"],
                history_length=len(seq["history"]),
                state_json=seq["state"],
                tx_features_json=seq["tx_features"],
                event_interval_months=seq.get("event_interval_months"),
                cash_need_krw=seq.get("cash_need_krw"),
                cash_need_source=seq.get("cash_need_source"),
                embedding_vector=[float(x) for x in vec],
                embedding_model=model_name,
                embedding_dim=dim,
            )
        )

    db.flush()
    return {"cohorts": len(sequences), "embedding_model": model_name, "dim": dim}
