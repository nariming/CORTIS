"""
DB 없이(SQLite 인메모리) 백엔드 로직을 검증하는 스모크 테스트.

목적
  - MySQL이 안 깔린 팀원 PC에서도 "감지 → 확정 → 히스토리 → 정책매칭" 배선이 도는지 바로 확인.
  - 실제 운영은 MySQL이지만, ORM 계층만 검증하면 되는 로직은 SQLite로도 충분하다.

실행: python -m backend.tests.test_smoke
"""

import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base
from backend.db import models  # noqa: F401  (모델 등록에 필요)
from backend.db.seed.seed_catalog import seed_catalog
from backend.db.seed.seed_demo_users import seed_demo_users
from backend.matcher_a import policy_matcher
from backend.matcher_a.detector import detect_all
from backend.repositories import event_repo

CHECKS = []


def check(label: str, condition: bool, detail: str = ""):
    CHECKS.append((label, condition, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()

    print("\n=== 1. 시드 적재 ===")
    catalog = seed_catalog(db)
    demo = seed_demo_users(db)
    db.commit()
    check("정책 DB 적재", catalog["policies"] >= 10, f"{catalog['policies']}건")
    check("데모 유저 적재", demo["users"] == 2, f"확정 이벤트 {demo['confirmed_events']}건")

    print("\n=== 2. 확정 히스토리 (C엔진 입력) ===")
    hist_a = event_repo.confirmed_history(db, "U_A")
    hist_b = event_repo.confirmed_history(db, "U_B")
    check("유저 A 히스토리", hist_a == ["대학생", "졸업"], str(hist_a))
    check("유저 B 히스토리", hist_b[0] == "취업" and "독립(월세)" in hist_b, str(hist_b))
    check("히스토리가 서로 다름 (데모 대비의 전제)", hist_a != hist_b)

    print("\n=== 3. 규칙기반 이벤트 감지 ===")
    detected = {}
    for user_id in ("U_A", "U_B"):
        user = db.get(models.User, user_id)
        txs = sorted(user.transactions, key=lambda t: t.tx_date)
        confirmed = [(e.event_type, e.occurred_at) for e in event_repo.event_timeline(db, user_id, "confirmed")]
        cands = detect_all(txs, confirmed)
        detected[user_id] = cands
        types = [c.event_type for c in cands]
        check(f"{user_id} 취업 감지", "취업" in types, f"후보={types}")

    top_a = next(c for c in detected["U_A"] if c.event_type == "취업")
    check("감지 근거(tx_id) 존재", len(top_a.evidence_tx_ids) > 0, top_a.reason)

    # B는 이미 독립(월세)가 확정돼 있으므로 같은 월세 흐름을 다시 물어보면 안 된다
    check(
        "B: 확정된 독립(월세)를 재감지하지 않음",
        "독립(월세)" not in [c.event_type for c in detected["U_B"]],
    )
    # B의 재취업은 공백 뒤 '새' 직장을 가리켜야 한다 (옛 직장이 아니라)
    top_b = next(c for c in detected["U_B"] if c.event_type == "취업")
    check("B: 재취업이 새 직장을 지목", "디자인랩" in top_b.reason, top_b.reason)

    print("\n=== 4. 이벤트 확정 → 히스토리 갱신 (agentic 순환 트리거) ===")
    user_a = db.get(models.User, "U_A")
    txs = sorted(user_a.transactions, key=lambda t: t.tx_date)
    cand = next(c for c in detect_all(txs, []) if c.event_type == "취업")
    saved = event_repo.add_event(
        db, "U_A", cand.event_type, cand.occurred_at, status="detected", confidence=cand.confidence
    )
    event_repo.confirm_event(db, saved.event_id)
    db.commit()
    hist_a2 = event_repo.confirmed_history(db, "U_A")
    check("확정 후 히스토리에 취업 추가", hist_a2 == ["대학생", "졸업", "취업"], str(hist_a2))
    check("prev_gap_month 자동 계산", saved.prev_gap_month is not None, f"{saved.prev_gap_month}개월")

    print("\n=== 5. A파트 정책 매칭 — 예측 이벤트 사전 안내 (데모의 대비 장면) ===")
    groups_a = policy_matcher.match_for_events(db, db.get(models.User, "U_A"), ["독립(월세)"])
    groups_b = policy_matcher.match_for_events(db, db.get(models.User, "U_B"), ["결혼"])
    names_a = [p["policy_name"] for p in groups_a[0]["policies"]]
    names_b = [p["policy_name"] for p in groups_b[0]["policies"]]
    check("A: 독립(월세) 예측 → 월세 지원 정책 사전 안내", len(names_a) >= 2, str(names_a))
    check("B: 결혼 예측 → 신혼부부 정책 사전 안내", any("신혼" in n for n in names_b), str(names_b))
    check(
        "가정한 상태 변화가 응답에 명시됨",
        all(p["assumption"] for p in groups_a[0]["policies"]),
        groups_a[0]["policies"][0]["assumption"],
    )

    # prospective=False 면 '아직 부모동거'라서 월세 정책이 빠져야 한다 (모드가 실제로 다르게 동작하는지)
    strict_a = policy_matcher.match_for_events(
        db, db.get(models.User, "U_A"), ["독립(월세)"], prospective=False
    )
    check(
        "현재 상태 기준(prospective=False)에서는 결과가 더 좁음",
        len(strict_a[0]["policies"]) < len(groups_a[0]["policies"]),
        f"{len(strict_a[0]['policies'])}건 < {len(groups_a[0]['policies'])}건",
    )

    print("\n=== 6. 자격 판정이 유저별로 갈리는지 (규칙이 실제로 동작하는지) ===")
    elig_a = {m.policy.policy_id for m in policy_matcher.current_eligibility(db, db.get(models.User, "U_A"))}
    elig_b = {m.policy.policy_id for m in policy_matcher.current_eligibility(db, db.get(models.User, "U_B"))}
    check("A/B 자격 정책 집합이 다름", elig_a != elig_b, f"A={len(elig_a)}건 B={len(elig_b)}건")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 50}")
    print(f"결과: {len(CHECKS) - len(failed)}/{len(CHECKS)} 통과")
    if failed:
        for label, _, detail in failed:
            print(f"  실패: {label} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
