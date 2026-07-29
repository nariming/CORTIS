"""
초기 세팅 원커맨드 스크립트.

    python -m backend.db.seed.run_all            # 스키마 생성 + 전체 시드
    python -m backend.db.seed.run_all --no-drop  # 스키마는 건드리지 않고 시드만 다시

하는 일
  1. cortis 데이터베이스 생성 (없으면)
  2. schema.sql 실행 → 테이블 생성 (기본은 DROP 후 재생성)
  3. 대출상품 / 정책 DB 적재
  4. 데모 유저 A/B + 거래내역 + 확정 이벤트 적재
  5. 합성 코호트 300개 임베딩 후 적재
"""

import argparse
import sys
from pathlib import Path

import pymysql
from sqlalchemy import text

from backend import config
from backend.db.database import session_scope, engine
from backend.db.seed.seed_catalog import seed_catalog
from backend.db.seed.seed_cohorts import seed_cohorts
from backend.db.seed.seed_demo_users import seed_demo_users

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema.sql"


def _split_statements(sql: str):
    """schema.sql 을 문장 단위로 쪼갠다.

    pymysql은 기본적으로 multi-statement 실행을 허용하지 않으므로 직접 나눈다.
    '--' 주석 줄을 먼저 제거해야 주석 안의 문자로 파싱이 흔들리지 않는다.
    (이 스키마는 트리거/프로시저가 없어 DELIMITER 처리를 고려할 필요가 없다)
    """
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def apply_schema() -> int:
    """DB 생성 + schema.sql 실행. 실행한 문장 수를 돌려준다."""
    conn = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        statements = _split_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        return len(statements)
    finally:
        conn.close()


def verify_connection() -> str:
    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar()
        db_name = conn.execute(text("SELECT DATABASE()")).scalar()
    return f"MySQL {version} / database={db_name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="CORTIS DB 초기화 & 시드")
    parser.add_argument(
        "--no-drop", action="store_true", help="schema.sql 실행을 건너뛰고 시드만 다시 넣는다"
    )
    args = parser.parse_args()

    if not args.no_drop:
        count = apply_schema()
        print(f"[1/5] 스키마 적용 완료 — {count}개 문장 실행 ({SCHEMA_PATH.name})")
    else:
        print("[1/5] 스키마 적용 건너뜀 (--no-drop)")

    print(f"[2/5] 접속 확인 — {verify_connection()}")

    with session_scope() as db:
        result = seed_catalog(db)
        print(f"[3/5] 카탈로그 적재 — 대출상품 {result['loan_products']}건, 정책 {result['policies']}건")

        result = seed_demo_users(db)
        print(
            f"[4/5] 데모 유저 적재 — 유저 {result['users']}명, 대출 {result['loans']}건, "
            f"확정 이벤트 {result['confirmed_events']}건"
        )

        result = seed_cohorts(db)
        print(
            f"[5/5] 코호트 적재 — {result['cohorts']}건 "
            f"(임베딩 {result['embedding_model']}, {result['dim']}차원)"
        )

    print("\n완료. 서버 실행: uvicorn backend.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
