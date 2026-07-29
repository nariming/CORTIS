"""
실제 MySQL(backend/.env 접속 정보)에 붙여 라우터 로직까지 검증하는 스크립트.

test_smoke.py 는 SQLite로 ORM/로직만 보고, 이건 실제 cortis 데이터베이스 세션으로
라우터 함수를 직접 호출해 확인한다. (이 개발 환경의 Git Bash + Windows 조합에서
asyncio socketpair()가 불안정해 실제 HTTP서버/TestClient 대신 함수 직접 호출로 검증)

사전에 python -m backend.db.seed.run_all 실행 필요.
실행: python -m backend.tests.test_api_live
"""

import sys

from backend.db.database import session_scope
from backend.repositories import cohort_repo
from backend.routers import events as events_router
from backend.routers import policies as policies_router
from backend.routers import predictions as predictions_router
from backend.routers import users as users_router
from backend.schemas import PolicyMatchIn, PredictionIn

CHECKS = []


def check(label: str, condition: bool, detail: str = ""):
    CHECKS.append((label, condition))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    with session_scope() as db:
        print("=== 유저 조회 (실제 MySQL) ===")
        ua = users_router.get_user("U_A", db)
        check("U_A 확정 히스토리", ua.confirmed_history == ["대학생", "졸업"], str(ua.confirmed_history))

        print("\n=== 코호트 검색 인덱스 (C엔진 연동 지점) ===")
        rows = cohort_repo.load_cohort_rows(db)
        check("코호트 300건 로드", len(rows) == 300, f"{len(rows)}건")
        check("벡터 차원 64", len(rows[0]["embedding_vector"]) == 64, str(len(rows[0]["embedding_vector"])))

        print("\n=== 거래내역 규칙기반 감지 (persist 없이 조회만) ===")
        detect_result = events_router.detect_events("U_A", persist=False, db=db)
        types = [c.event_type for c in detect_result.candidates]
        check("U_A 취업 후보 감지", "취업" in types, str(types))

        print("\n=== A파트: 예측 이벤트 기반 정책 사전 안내 ===")
        groups_a = policies_router.match_policies(
            "U_A", PolicyMatchIn(event_types=["독립(월세)"], persist=False), db
        )
        names_a = [p["policy_name"] for p in groups_a[0]["policies"]]
        check("A: 독립(월세) 예측 → 월세 정책 2건 이상", len(names_a) >= 2, str(names_a))

        groups_b = policies_router.match_policies(
            "U_B", PolicyMatchIn(event_types=["결혼"], persist=False), db
        )
        names_b = [p["policy_name"] for p in groups_b[0]["policies"]]
        check("B: 결혼 예측 → 신혼부부 정책", any("신혼" in n for n in names_b), str(names_b))

        print("\n=== 예측 결과 저장 (근거 코호트 포함, 실제 커밋) ===")
        saved = predictions_router.save_prediction(
            "U_A",
            PredictionIn(
                input_history=["대학생", "졸업", "취업"],
                predictions=[{"event": "독립(월세)", "evidence_count": 3, "reasoning": "라이브 검증용"}],
                confidence_level="medium",
                matched_cohorts=[{"history": ["취업"], "next_event": "독립(월세)", "similarity": 0.8}],
            ),
            db,
        )
        check("예측 저장 성공 (prediction_id 발급)", saved.prediction_id is not None, str(saved.prediction_id))

        history = predictions_router.list_predictions("U_A", limit=5, db=db)
        check("예측 이력 조회", len(history) >= 1, f"{len(history)}건")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 50}\n결과: {len(CHECKS) - len(failed)}/{len(CHECKS)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
