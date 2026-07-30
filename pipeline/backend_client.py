"""
개발자1의 FastAPI 백엔드(GET /cohorts)에서 실제 코호트 데이터를 가져와
CohortIndex에 로드하는 연결 모듈.

사용법 (데모 스크립트에서):
    from pipeline.backend_client import load_cohort_index_from_backend
    index = load_cohort_index_from_backend(embedder)
"""

import os
import requests
import datetime
from pipeline.embedding import EmbeddingProvider
from pipeline.similarity import CohortIndex
from pipeline.contracts import PredictionSaveRequest, PolicyMatchRequest
from pipeline.portfolio import UserFinancialProfile, ExistingLoan, RefinanceCandidate

BACKEND_BASE_URL = "http://localhost:8000"

# 모든 요청에 API 키 헤더를 자동으로 붙이는 세션 (backend/security.py의 verify_api_key와 짝).
# .env의 API_KEY를 그대로 사용 - 서버와 같은 값이어야 인증 통과.
_session = requests.Session()
_session.headers.update({"X-API-Key": os.environ.get("API_KEY", "")})


def fetch_cohorts_from_backend(base_url: str = BACKEND_BASE_URL) -> list:
    """GET /cohorts 호출해서 원본 row 리스트를 그대로 반환.

    서버(uvicorn backend.main:app --reload)가 켜져 있어야 함.
    """
    resp = _session.get(f"{base_url}/cohorts", timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_cohort_index_from_backend(embedder: EmbeddingProvider, base_url: str = BACKEND_BASE_URL) -> CohortIndex:
    """실제 백엔드에서 코호트를 가져와 CohortIndex를 만들어 반환.

    rows의 embedding_vector를 그대로 쓰므로, 임베딩을 다시 계산하지 않는다.
    (단, 개발자1 시드 스크립트와 우리 EMBEDDING_BACKEND가 같은 방식이어야
     유사도가 의미를 가짐 — 지금은 둘 다 offline-hash-64라 문제 없음)
    """
    rows = fetch_cohorts_from_backend(base_url)
    index = CohortIndex(embedder)
    index.load_from_mysql_rows(rows)
    return index


def get_user_history(user_id: str, base_url: str = BACKEND_BASE_URL) -> dict:
    """GET /users/{user_id}/history 호출."""
    resp = _session.get(f"{base_url}/users/{user_id}/history", timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_life_event(user_id: str, event_type: str, base_url: str = BACKEND_BASE_URL) -> dict:
    """POST /users/{user_id}/events 호출해서 이벤트를 '확정' 상태로 등록하고,
    응답(event_id 포함)을 그대로 반환.
    """
    payload = {
        "event_type": event_type,
        "occurred_at": datetime.date.today().isoformat(),
        "status": "confirmed",
        "confidence": 1.0,
    }
    resp = _session.post(f"{base_url}/users/{user_id}/events", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def save_prediction(
    user_id: str,
    trigger_event_id: int,
    input_history: list,
    predictions: list,
    confidence_level: str,
    confidence_note: str,
    matched_cohorts: list,
    base_url: str = BACKEND_BASE_URL,
) -> dict:
    """POST /users/{user_id}/predictions 호출해서 예측 결과를 저장하고,
    응답(prediction_id 포함)을 그대로 반환.
    """
    req = PredictionSaveRequest(
        trigger_event_id=trigger_event_id,
        input_history=input_history,
        predictions=predictions,
        confidence_level=confidence_level,
        confidence_note=confidence_note,
        matched_cohorts=matched_cohorts,
    )
    resp = _session.post(f"{base_url}/users/{user_id}/predictions", json=asdict_safe(req), timeout=10)
    resp.raise_for_status()
    return resp.json()


def request_policy_match(
    user_id: str,
    prediction_id: int,
    event_types: list,
    include_ineligible: bool = False,
    persist: bool = True,
    base_url: str = BACKEND_BASE_URL,
) -> dict:
    """POST /users/{user_id}/policy-match 호출해서 A파트 정책 매칭을 실제로 요청.

    persist=False로 호출하면 policy_match_results 테이블에 기록을 안 남긴다
    (포트폴리오 결정 레이어가 후보 탐색용으로 내부적으로 호출할 때 중복 기록 방지용).
    """
    req = PolicyMatchRequest(
        prediction_id=prediction_id,
        event_types=event_types,
        include_ineligible=include_ineligible,
        persist=persist,
    )
    resp = _session.post(f"{base_url}/users/{user_id}/policy-match", json=asdict_safe(req), timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_user_detail(user_id: str, base_url: str = BACKEND_BASE_URL) -> dict:
    """GET /users/{user_id} 호출. UserDetailOut 그대로 반환 (loans[] 포함)."""
    resp = _session.get(f"{base_url}/users/{user_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def request_loan_match(
    user_id: str,
    event_types: list,
    include_ineligible: bool = False,
    prospective: bool = True,
    base_url: str = BACKEND_BASE_URL,
) -> list:
    """POST /users/{user_id}/loan-match 호출 (개발자1 PR #5).

    C엔진이 예측/확정한 이벤트를 넘기면, 그 이벤트로 새로 자격이 생기는
    KB 대출상품 목록을 EventLoanGroupOut[] 형태로 돌려준다.
    """
    payload = {
        "event_types": event_types,
        "include_ineligible": include_ineligible,
        "prospective": prospective,
    }
    resp = _session.post(f"{base_url}/users/{user_id}/loan-match", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 어댑터: API 응답(dict) -> pipeline/portfolio.py 데이터클래스
# ---------------------------------------------------------------------------

def build_profile_from_user_detail(user_detail: dict) -> UserFinancialProfile:
    """GET /users/{user_id} 응답 -> portfolio.UserFinancialProfile.

    liquid_assets_krw는 아직 backend에 필드가 없어 None으로 둔다
    (개발자1 필드 추가되면 user_detail.get("liquid_assets_krw")로 채우기만 하면 됨).
    """
    return UserFinancialProfile(
        user_id=user_detail["user_id"],
        annual_income=user_detail["annual_income"],
        income_volatility=user_detail["income_volatility"],
        employment_type=user_detail["employment_type"],
        region_code=user_detail["region_code"],
        liquid_assets_krw=user_detail.get("liquid_assets_krw"),  # 필드 생기면 자동 반영
    )


def build_existing_loans_from_user_detail(user_detail: dict, today: datetime.date = None) -> list:
    """GET /users/{user_id} 응답의 loans[] -> portfolio.ExistingLoan 리스트.

    remaining_months는 maturity_at - 오늘 날짜로 계산 (API가 직접 주지 않음).
    """
    today = today or datetime.date.today()
    loans = []
    for loan in user_detail.get("loans", []):
        if loan.get("status") != "정상":
            continue  # 이미 완제/연체된 대출은 재설계 대상에서 제외
        maturity = datetime.date.fromisoformat(loan["maturity_at"])
        remaining_months = max((maturity.year - today.year) * 12 + (maturity.month - today.month), 1)
        loans.append(ExistingLoan(
            loan_id=loan["loan_id"],
            product_id=loan["product_id"],
            balance=loan["balance"],
            interest_rate=loan["interest_rate"],
            monthly_payment=loan["monthly_payment"],
            remaining_months=remaining_months,
        ))
    return loans


def build_refinance_map_from_loan_match(
    existing_loans: list,
    loan_match_response: list,
) -> dict:
    """POST /users/{user_id}/loan-match 응답(EventLoanGroupOut[]) -> loan_id별 대환후보 딕셔너리.

    주의: loan-match는 "이벤트로 자격되는 상품 목록"을 주지, "어떤 기존 대출을 대체하는지"는
    안 알려준다. 그래서 같은 product_type을 가진 기존 대출에 후보로 붙이는 방식으로 근사한다
    (예: 기존 신용대출 -> 신규 신용대출류 상품만 대환후보로 취급).
    이 근사가 실제와 다를 수 있는 케이스(예: 신용대출 -> 전월세자금대출로 대환)는
    이번 버전 범위 밖으로 하고, 필요하면 나중에 product_type 매핑 룰을 추가할 것.
    """
    refinance_map: dict = {loan.loan_id: [] for loan in existing_loans}

    for group in loan_match_response:
        for product in group.get("loan_products", []):
            if product.get("status") not in ("eligible", "newly_eligible"):
                continue
            candidate = RefinanceCandidate(
                product_id=product["product_id"],
                product_name=product["product_name"],
                min_rate=product["min_rate"],
                max_rate=product["max_rate"],
                max_amount=product["max_amount"],
            )
            for loan in existing_loans:
                # product_type 정보가 없으므로(EventLoanGroupOut에 product_type은 있음),
                # 호출부에서 필요시 더 정교한 매핑으로 교체 가능하도록 전부 후보로 붙임.
                # (대출 개수가 적어 조합탐색 비용에 문제 없음 - 말이 안 되는 조합은
                #  NPV/DSR 계산에서 자연스럽게 걸러짐)
                refinance_map[loan.loan_id].append(candidate)

    return refinance_map


def fetch_policy_catalog(base_url: str = BACKEND_BASE_URL) -> list:
    """GET /policies 호출 (전체 카탈로그, benefit_rate_pct 등 숫자필드 포함).

    /policy-match 응답(PolicyMatchOut)에는 이 숫자필드가 없어서, policy_id로
    이 카탈로그와 매칭해 보강하는 용도로 쓴다.
    """
    resp = _session.get(f"{base_url}/policies", timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_refinance_map_from_policy_match(
    existing_loans: list,
    policy_match_response: list,
    policy_catalog: list,
) -> dict:
    """POST /users/{user_id}/policy-match 응답 -> loan_id별 '정책형 대출' 대환후보.

    benefit_rate_pct가 있는 정책만 "금리 조건이 있는 대출성 정책"으로 간주해 후보로 넣는다
    (청년내일채움공제처럼 benefit_amount_krw만 있는 건 대출이 아니라 지원금이라 제외).

    max_amount는 Policy 테이블에 없어서 정확한 한도를 모른다 - 보수적으로 기존 대출잔액을
    그대로 사용(실제 한도가 이보다 적을 수 있음을 감안해야 함, TODO: 카탈로그에 한도 필드
    추가되면 교체).
    """
    from pipeline.portfolio import RefinanceCandidate

    catalog_by_id = {p["policy_id"]: p for p in policy_catalog}
    refinance_map: dict = {loan.loan_id: [] for loan in existing_loans}

    for group in policy_match_response:
        for policy_match in group.get("policies", []):
            if policy_match.get("status") not in ("eligible", "newly_eligible"):
                continue
            catalog_entry = catalog_by_id.get(policy_match["policy_id"])
            if not catalog_entry or catalog_entry.get("benefit_rate_pct") is None:
                continue  # 금리 정보 없는 정책(지원금류)은 대출 후보가 아니므로 제외

            rate = catalog_entry["benefit_rate_pct"]
            for loan in existing_loans:
                candidate = RefinanceCandidate(
                    product_id=policy_match["policy_id"],
                    product_name=policy_match["policy_name"] + " (정책형)",
                    min_rate=rate,
                    max_rate=rate,
                    max_amount=loan.balance,  # 정확한 한도 불명 - 보수적 근사, 위 docstring 참고
                )
                refinance_map[loan.loan_id].append(candidate)

    return refinance_map


def merge_refinance_maps(*maps: dict) -> dict:
    """LoanProduct 기반 대환후보(build_refinance_map_from_loan_match)와 Policy 기반
    대환후보(build_refinance_map_from_policy_match)를 하나로 합친다.
    이렇게 후보군을 합쳐두면, 어느 쪽이 더 유리한지는 NPV 비교가 알아서 골라준다
    (KB 자체상품 vs 정책형 대출을 같은 기준으로 경쟁시킴 = '정책활용도' 고려의 실제 구현체).
    """
    merged: dict = {}
    for m in maps:
        for loan_id, candidates in m.items():
            merged.setdefault(loan_id, []).extend(candidates)
    return merged


def asdict_safe(dataclass_obj) -> dict:
    from dataclasses import asdict
    return asdict(dataclass_obj)