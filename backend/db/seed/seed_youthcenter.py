"""
온통청년 오픈API를 실제로 호출해 policies 테이블에 적재하는 1회성 ETL.

seed_catalog.py(수작업 큐레이션 15건)를 대체하는 게 아니라 **위에 덧붙인다**:
  - source='manual'          : seed_catalog.py 가 만든 기존 정책 (정량 필드 큐레이션 완료)
  - source='youthcenter_api' : 이 스크립트가 실제 API에서 받아온 정책 (정량 필드는 best-effort)

이렇게 나누는 이유: API가 발표 당일 응답이 느리거나 실패해도(외부 서비스라 보장 못함)
manual 15건은 항상 살아있어서 데모가 완전히 죽지 않는다.

실행: python -m backend.db.seed.seed_youthcenter
      (run_all.py 에도 통합돼 있어 YOUTHCENTER_API_KEY 가 있으면 자동으로 같이 돈다)
"""

import sys
from typing import Optional

from sqlalchemy.orm import Session

from backend.db.models import Policy
from backend.integrations.youthcenter_client import YouthCenterAPIError, fetch_policies
from backend.integrations.youthcenter_mapper import map_raw_to_policy


def seed_from_youthcenter(db: Session, max_records: int = 100, query: str = "") -> dict:
    """API 호출 -> 매핑 -> upsert. 실패하면 YouthCenterAPIError를 그대로 던진다

    (호출부인 run_all.py가 "실패하면 조용히 폴백"을 결정하므로, 여기서는 숨기지 않고
    그대로 올려서 원인을 알 수 있게 한다).
    """
    raw_rows = fetch_policies(query=query, display=max_records)

    inserted = 0
    skipped_no_mapping = 0
    skipped_no_trigger = 0
    needs_review = 0

    for raw in raw_rows:
        mapped = map_raw_to_policy(raw)
        if mapped is None:
            skipped_no_mapping += 1
            continue
        if not mapped["trigger_events"]:
            skipped_no_trigger += 1
            continue

        notes = mapped.pop("_structuring_notes", "")
        verified = mapped.pop("_structuring_verified", True)
        if not verified:
            needs_review += 1
            print(f"  [검수필요] {mapped['policy_name']}: {notes}")

        db.merge(Policy(**mapped))
        inserted += 1

    db.flush()
    return {
        "fetched": len(raw_rows),
        "inserted": inserted,
        "skipped_no_mapping": skipped_no_mapping,
        "skipped_no_trigger": skipped_no_trigger,
        "needs_review": needs_review,
    }


def main() -> int:
    from dotenv import load_dotenv
    from pathlib import Path

    from backend.db.database import session_scope

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    try:
        with session_scope() as db:
            result = seed_from_youthcenter(db)
    except YouthCenterAPIError as e:
        print(f"[실패] 온통청년 API 연동 실패: {e}", file=sys.stderr)
        print("(seed_catalog.py 의 수작업 정책 15건은 영향받지 않습니다)", file=sys.stderr)
        return 1

    print(
        f"온통청년 적재 완료 — 수신 {result['fetched']}건 / 적재 {result['inserted']}건 / "
        f"필드매핑 실패 {result['skipped_no_mapping']}건 / 트리거이벤트 없음(스킵) {result['skipped_no_trigger']}건 / "
        f"검수필요(금액 미검증) {result['needs_review']}건"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())