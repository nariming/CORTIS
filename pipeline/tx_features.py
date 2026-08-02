"""
Transaction Feature Extractor — C엔진(State Embedding + LLM user_context) 입력용
거래 트렌드/패턴 feature 추출.

detector.py(A파트)가 "이벤트가 발생했는가"를 규칙기반으로 감지한다면,
여기는 "이 유저의 최근 금융 상태가 어느 방향으로 움직이고 있는가"를 수치화한다.
detector.py처럼 확정 이벤트로 올리지는 않지만, State Retrieval과 LLM 판단의 입력 재료가 된다.

설계 의도
  - 스냅샷(현재 소득/자산)보다 추세(변화 방향)가 다음 생애주기 이벤트를 더 잘 설명한다는
    문제의식(청년 대출 연체의 원인은 소득 수준 자체가 아니라 소득 발생의 불안정성이라는 진단)을
    그대로 따른다.
  - 전부 "최근 N개월 vs 이전 N개월" 비교 구조로 통일해 계산 근거를 결정론적으로 제시한다.
  - 여기서 나온 값은 State Embedding 문장과 LLM 프롬프트(build_user_context)에 그대로 노출된다.
    LLM은 이 값을 해석할 수는 있어도 값 자체를 생성하지 않는다 — 코호트의 예상 필요자금(cash_need)을
    코호트 집계값으로만 산출하는 것과 동일한 원칙으로, 근거 없는 수치를 만들지 않기 위함이다.
  - saving_growth와 가구·가전 신호는 거래내역에 전용 category가 아직 세분화되지 않은 상태를
    가정해 counterparty/memo 키워드 매칭으로 판별한다(detector.py의 WEDDING_KEYWORDS 등과 동일
    패턴). category가 세분화되면 정확도가 더 높은 category 기반 판별로 손쉽게 교체 가능하도록
    이 판별 로직을 별도 상수(SAVING_KEYWORDS)로 분리해 두었다.

한계 및 향후 확장
  - 키워드 매칭은 거래처명에 의존하므로 이름 표기가 다르면(예: 은행명 축약) 누락될 수 있다.
    실 서비스 적용 시 category 세분화로 대체하는 것을 권장한다.
  - TREND_WINDOW_MONTHS/VOLATILITY_WINDOW_MONTHS는 최소 2*window개월의 거래내역이 있어야
    유의미한 값이 나온다. 데이터가 부족하면 None을 반환해 "산출 불가"를 명시적으로 표현한다.
"""

from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Dict, List, Optional

from backend.db.models import Transaction

# 최근/이전 구간 비교 윈도우(개월). 현재 데모 거래내역은 12개월치로,
# TREND_WINDOW=3이면 최근 3개월 대비 그 이전 3개월을 비교할 수 있다.
TREND_WINDOW_MONTHS = 3
VOLATILITY_WINDOW_MONTHS = 6

# 추세를 "증가/감소"로 문장화할 때의 최소 변화폭(노이즈 무시용)
TREND_NOISE_THRESHOLD = 0.03

# 저축성 이체 판별용 키워드(category 세분화 이전 임시 판별 기준)
SAVING_KEYWORDS = ("적금", "예금", "저축", "CMA", "펀드", "ISA")
# 가구·가전 구매 증가 → 독립 가능성 신호 판별용 키워드
FURNITURE_KEYWORDS = ("이케아", "가구", "가전", "하이마트", "리빙", "인테리어")

# 지출 추세 계산에서 제외하는 category. 두 항목 모두 "소비 습관 변화"가 아니라
# 별개의 생애주기 이벤트(신규 대출 실행, 이사/독립) 때문에 값이 바뀌는 고정성 채무 지출이며,
# 그 이벤트 자체는 이미 user_loans 테이블과 detect_independence()가 별도로 추적한다.
# 포함하면 "독립해서 월세가 새로 발생한 것"이나 "신규 대출로 상환액이 늘어난 것"이
# "소비 지출 증가"로 오인될 수 있어 제외한다.
# "대출상환"은 여기서 빠지는 대신 debt_growth(아래) 전용 feature로 별도 추적한다 -
# 소비 추세와 섞이면 안 되지만, "최근 대출부담이 늘고 있는가"는 그 자체로 생애주기 신호이기 때문.
EXPENSE_EXCLUDE_CATEGORIES = ("대출상환", "월세")


@dataclass
class TransactionFeatures:
    """State/LLM 프롬프트에 노출되는 거래 기반 feature 묶음.

    growth 계열은 데이터가 부족하면(윈도우 두 구간이 안 채워지면) None — LLM에게
    "모른다"를 명시적으로 전달하기 위해 0으로 채우지 않는다(콜드스타트와 같은 맥락).
    """

    income_growth: Optional[float]
    expense_growth: Optional[float]
    saving_growth: Optional[float]
    debt_growth: Optional[float]
    cashflow_volatility: Optional[float]
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "income_growth": self.income_growth,
            "expense_growth": self.expense_growth,
            "saving_growth": self.saving_growth,
            "debt_growth": self.debt_growth,
            "cashflow_volatility": self.cashflow_volatility,
            "signals": self.signals,
        }

    def to_sentence(self) -> str:
        """Transaction Embedding 문장/유저 컨텍스트에 이어붙일 한국어 조각."""
        return tx_features_dict_to_sentence(self.to_dict())


def tx_features_dict_to_sentence(d: dict) -> str:
    """tx_features dict(실유저 TransactionFeatures.to_dict() 또는 코호트 tx_features_json과
    동일 스키마) -> Transaction Embedding 대상 문장.

    실유저 쪽(TransactionFeatures.to_sentence())과 코호트 쪽(generate_cohorts.py가 합성한
    tx_features_json)이 이 함수 하나를 공유해야, 두 벡터가 같은 문장 구조에서 나와
    코사인 유사도가 의미를 갖는다 — 문장 생성 로직이 두 곳에서 따로 구현되면(이전에 as_of
    기본값이 두 함수에서 갈라졌던 것과 같은 종류의 정합성 문제) 검색 결과가 왜곡된다.
    """
    parts: List[str] = []

    def _direction(v: float) -> str:
        if v > TREND_NOISE_THRESHOLD:
            return "증가"
        if v < -TREND_NOISE_THRESHOLD:
            return "감소"
        return "유지"

    income_growth = d.get("income_growth")
    if income_growth is not None:
        parts.append(f"최근 소득 {_direction(income_growth)}({income_growth:+.0%})")

    expense_growth = d.get("expense_growth")
    if expense_growth is not None:
        parts.append(f"지출 {_direction(expense_growth)}({expense_growth:+.0%})")

    saving_growth = d.get("saving_growth")
    if saving_growth is not None:
        parts.append(f"저축성 이체 {_direction(saving_growth)}({saving_growth:+.0%})")

    debt_growth = d.get("debt_growth")
    if debt_growth is not None:
        parts.append(f"대출상환액 {_direction(debt_growth)}({debt_growth:+.0%})")

    cashflow_volatility = d.get("cashflow_volatility")
    if cashflow_volatility is not None:
        parts.append(f"현금흐름 변동성 {cashflow_volatility:.2f}")

    parts.extend(d.get("signals") or [])

    return ", ".join(parts) if parts else "거래내역 부족(추세 산출 불가)"


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _month_index(key: str) -> int:
    y, m = map(int, key.split("-"))
    return y * 12 + m


def _monthly_sum(txs: List[Transaction], predicate) -> Dict[str, int]:
    """predicate(tx)->bool 로 걸러낸 거래의 월별 합계."""
    out: Dict[str, int] = {}
    for t in txs:
        if not predicate(t):
            continue
        key = _month_key(t.tx_date)
        out[key] = out.get(key, 0) + t.amount
    return out


def _recent_prior_avg(monthly: Dict[str, int], as_of: date, window_months: int):
    """as_of 기준 [최근 window_months] 평균 vs [그 이전 window_months] 평균.

    두 구간 모두 데이터가 있어야 비교가 성립하므로, 하나라도 비면 (None, None).
    """
    ref_idx = as_of.year * 12 + as_of.month
    recent = [v for k, v in monthly.items() if 0 <= ref_idx - _month_index(k) < window_months]
    prior = [
        v for k, v in monthly.items() if window_months <= ref_idx - _month_index(k) < 2 * window_months
    ]
    if not recent or not prior:
        return None, None
    return sum(recent) / len(recent), sum(prior) / len(prior)


def _growth_ratio(recent_avg: Optional[float], prior_avg: Optional[float]) -> Optional[float]:
    if recent_avg is None or prior_avg is None or prior_avg == 0:
        return None
    return (recent_avg / prior_avg) - 1


def _cashflow_volatility(txs: List[Transaction], as_of: date, window_months: int) -> Optional[float]:
    """최근 window_months간 월별 순현금흐름(입금-출금)의 MAD/중앙값.

    users.income_volatility(소득만 대상, MAD 기반)와 계산 방식은 통일하되,
    대상을 순현금흐름 전체(소득-지출)로 넓혀 "생활비 압박" 신호까지 잡는다 — 별개 지표로 유지.
    """
    monthly_net = _monthly_sum(txs, lambda t: True)
    ref_idx = as_of.year * 12 + as_of.month
    values = [v for k, v in monthly_net.items() if 0 <= ref_idx - _month_index(k) < window_months]
    if len(values) < 2:
        return None
    med = median(values)
    if med == 0:
        return None
    mad = median([abs(v - med) for v in values])
    return abs(mad / med)


def _extract_qualitative_signals(txs: List[Transaction]) -> List[str]:
    """detector.py가 잡는 '확정 이벤트' 수준까지는 아니지만, State 문장에 얹을 만한
    약한 신호를 문장으로 뽑는다. (예: 가구·가전 구매 증가 → 독립 가능성)
    """
    signals: List[str] = []

    furniture_hits = [
        t
        for t in txs
        if t.amount < 0 and any(k in f"{t.counterparty} {t.memo or ''}" for k in FURNITURE_KEYWORDS)
    ]
    if len(furniture_hits) >= 2:
        signals.append("가구·가전 구매 증가(독립 가능성)")

    return signals


def extract_transaction_features(
    txs: List[Transaction], as_of: Optional[date] = None
) -> TransactionFeatures:
    """C엔진에 넘길 거래 기반 feature 계산 진입점.

    전부 결정론적 집계값이다 — LLM이 이 숫자를 재해석할 수는 있어도,
    숫자 자체를 LLM이 만들어내지 않는다.
    """
    if not txs:
        return TransactionFeatures(None, None, None, None, None, [])

    as_of = as_of or max(t.tx_date for t in txs)

    income_monthly = _monthly_sum(txs, lambda t: t.category == "급여" and t.amount > 0)
    expense_monthly = _monthly_sum(
        txs, lambda t: t.amount < 0 and t.category not in EXPENSE_EXCLUDE_CATEGORIES
    )
    saving_monthly = _monthly_sum(
        txs,
        lambda t: t.amount < 0 and any(k in f"{t.counterparty} {t.memo or ''}" for k in SAVING_KEYWORDS),
    )
    debt_monthly = _monthly_sum(txs, lambda t: t.amount < 0 and t.category == "대출상환")

    income_recent, income_prior = _recent_prior_avg(income_monthly, as_of, TREND_WINDOW_MONTHS)
    expense_recent, expense_prior = _recent_prior_avg(expense_monthly, as_of, TREND_WINDOW_MONTHS)
    saving_recent, saving_prior = _recent_prior_avg(saving_monthly, as_of, TREND_WINDOW_MONTHS)
    debt_recent, debt_prior = _recent_prior_avg(debt_monthly, as_of, TREND_WINDOW_MONTHS)

    income_growth = _growth_ratio(income_recent, income_prior)
    # 지출/저축/대출상환은 출금(음수)이라 절대값 기준으로 growth 계산 (금액이 커질수록 |값|이 커짐)
    expense_growth = _growth_ratio(
        abs(expense_recent) if expense_recent is not None else None,
        abs(expense_prior) if expense_prior is not None else None,
    )
    saving_growth = _growth_ratio(
        abs(saving_recent) if saving_recent is not None else None,
        abs(saving_prior) if saving_prior is not None else None,
    )
    debt_growth = _growth_ratio(
        abs(debt_recent) if debt_recent is not None else None,
        abs(debt_prior) if debt_prior is not None else None,
    )
    volatility = _cashflow_volatility(txs, as_of, VOLATILITY_WINDOW_MONTHS)
    signals = _extract_qualitative_signals(txs)

    return TransactionFeatures(
        income_growth=income_growth,
        expense_growth=expense_growth,
        saving_growth=saving_growth,
        debt_growth=debt_growth,
        cashflow_volatility=volatility,
        signals=signals,
    )