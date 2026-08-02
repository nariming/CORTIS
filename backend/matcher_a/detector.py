"""
A파트 ①: 거래내역 기반 규칙 이벤트 감지.

C엔진(예측)의 입력은 "확정된 이벤트 히스토리"인데, 그 히스토리를 채워주는 게 이 모듈이다.
흐름:  거래내역 → [여기] 감지(detected) → 사용자 확인 → 확정(confirmed) → C엔진 재예측

설계 의도
  - 감지는 일부러 규칙기반이다. "무엇을 근거로 취업이라 판단했는지"를 tx_id 단위로 남겨야
    사용자에게 확인 질문("○○에서 급여성 입금이 시작됐어요. 취업하셨나요?")을 던질 수 있기 때문.
  - 따라서 각 감지기는 event_type 뿐 아니라 evidence(근거 거래 id)와 confidence를 함께 돌려준다.
  - 모호한 판단은 여기서 확정하지 않는다. 확정 권한은 사용자에게 있다(life_events.status).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from backend.db.models import Transaction

# 급여성 입금으로 볼 최소 금액 (아르바이트 소액 입금과 구분)
SALARY_MIN_AMOUNT = 800_000
# 월세성 출금으로 볼 범위
RENT_MIN_AMOUNT = 150_000
RENT_MAX_AMOUNT = 2_000_000
# 정기성으로 인정할 최소 반복 횟수
RECURRING_MIN_COUNT = 2

WEDDING_KEYWORDS = ("웨딩", "예식", "스튜디오", "혼수", "예물")
MATERNITY_KEYWORDS = ("산부인과", "조리원", "산후", "육아", "유아")
STARTUP_KEYWORDS = ("사업자", "세무", "홈택스", "창업")
DEPOSIT_KEYWORDS = ("보증금", "전세", "임대차")


@dataclass
class EventCandidate:
    """감지된 이벤트 후보. 그대로 life_events 테이블에 status='detected' 로 들어간다."""

    event_type: str
    occurred_at: date
    confidence: float
    reason: str                                  # 사용자에게 보여줄 확인 질문의 근거 문장
    evidence_tx_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "evidence_tx_ids": self.evidence_tx_ids,
        }


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _group_by_counterparty(txs: List[Transaction], category: str) -> Dict[str, List[Transaction]]:
    grouped: Dict[str, List[Transaction]] = defaultdict(list)
    for tx in txs:
        if tx.category == category:
            grouped[tx.counterparty].append(tx)
    for rows in grouped.values():
        rows.sort(key=lambda t: t.tx_date)
    return grouped


# 급여 공백이 이 일수를 넘으면 '이직'이 아니라 '(재)취업'으로 본다.
# 퇴직 후 공백기를 거쳐 재취업하는 게 변동소득 청년의 전형적 패턴이라, 둘을 구분해야
# C엔진에 들어가는 히스토리가 실제 생애주기와 맞아떨어진다.
REEMPLOYMENT_GAP_DAYS = 90

# 거래내역 조회 구간이 열리자마자 이미 들어오고 있던 급여는 '새로 시작된 취업'이 아니라
# 원래 다니던 직장이다. 이걸 구분하지 않으면 조회 구간을 넓힐 때마다 과거 직장을
# 신규 취업으로 오탐한다. (구간 시작 후 이 일수 안에 첫 입금이면 '기존 진행 중'으로 간주)
ONGOING_GRACE_DAYS = 35


def _salary_source_timeline(txs: List[Transaction]):
    """급여처별 (첫 입금일, 마지막 입금일, 이름, 거래목록) 을 시간순으로.

    '정기성'은 서로 다른 달에 RECURRING_MIN_COUNT회 이상 들어온 것으로 판단한다
    (같은 달에 몰린 소액 다건은 급여로 보지 않음).
    """
    timeline = []
    for name, rows in _group_by_counterparty(txs, "급여").items():
        big = [t for t in rows if t.amount >= SALARY_MIN_AMOUNT]
        if len({_month_key(t.tx_date) for t in big}) < RECURRING_MIN_COUNT:
            continue
        timeline.append((big[0].tx_date, big[-1].tx_date, name, big))
    timeline.sort(key=lambda x: x[0])
    return timeline


def detect_employment(txs: List[Transaction]) -> List[EventCandidate]:
    """급여성 정기 입금이 새로 시작되면 취업 후보.

    첫 급여처 = 취업. 이후 급여처라도 공백이 REEMPLOYMENT_GAP_DAYS 를 넘으면 재취업이므로
    역시 '취업'으로 올린다 (공백이 짧으면 이직 — detect_job_change 담당).
    """
    timeline = _salary_source_timeline(txs)
    if not timeline:
        return []

    window_start = min(t.tx_date for t in txs)
    candidates: List[EventCandidate] = []

    for i, (start, _end, name, rows) in enumerate(timeline):
        if i == 0:
            # 조회 구간이 열릴 때 이미 급여가 들어오고 있었다면 기존 직장이므로 이벤트가 아니다
            if (start - window_start).days <= ONGOING_GRACE_DAYS:
                continue
            candidates.append(
                EventCandidate(
                    event_type="취업",
                    occurred_at=start,
                    confidence=0.85,
                    reason=f"'{name}'에서 매월 급여성 입금이 {len(rows)}회 확인됐어요. 취업하셨나요?",
                    evidence_tx_ids=[t.tx_id for t in rows[:3]],
                )
            )
            continue

        gap_days = (start - timeline[i - 1][1]).days
        if gap_days > REEMPLOYMENT_GAP_DAYS:
            candidates.append(
                EventCandidate(
                    event_type="취업",
                    occurred_at=start,
                    confidence=0.8,
                    reason=(
                        f"급여 공백 {gap_days}일 뒤 '{name}'에서 급여성 입금이 다시 시작됐어요. "
                        "재취업하셨나요?"
                    ),
                    evidence_tx_ids=[t.tx_id for t in rows[:3]],
                )
            )
    return candidates


def detect_job_change(txs: List[Transaction]) -> List[EventCandidate]:
    """급여처가 큰 공백 없이 교체되면 이직 후보 (공백이 길면 재취업이므로 제외)."""
    timeline = _salary_source_timeline(txs)
    candidates: List[EventCandidate] = []

    for i in range(1, len(timeline)):
        prev_end, prev_name = timeline[i - 1][1], timeline[i - 1][2]
        new_start, _, new_name, new_rows = timeline[i]
        gap_days = (new_start - prev_end).days
        if gap_days > REEMPLOYMENT_GAP_DAYS:
            continue
        candidates.append(
            EventCandidate(
                event_type="이직",
                occurred_at=new_start,
                confidence=0.8,
                reason=(
                    f"급여 입금처가 '{prev_name}' → '{new_name}'로 바뀌었어요"
                    f"(공백 {gap_days}일). 이직하셨나요?"
                ),
                evidence_tx_ids=[t.tx_id for t in new_rows[:2]],
            )
        )
    return candidates


def detect_income_stop(txs: List[Transaction], as_of: Optional[date] = None) -> List[EventCandidate]:
    """급여성 입금이 3개월 이상 끊기면 퇴직/실직 후보.

    변동소득 청년 타깃이라 오탐이 많을 수 있어 confidence를 낮게 잡고,
    사용자 확인 없이는 절대 확정하지 않는다.
    """
    salary_txs = [t for t in txs if t.category == "급여" and t.amount >= SALARY_MIN_AMOUNT]
    if not salary_txs:
        return []

    last = max(salary_txs, key=lambda t: t.tx_date)
    reference = as_of or max(t.tx_date for t in txs)
    months_idle = (reference.year - last.tx_date.year) * 12 + (reference.month - last.tx_date.month)
    if months_idle < 3:
        return []

    return [
        EventCandidate(
            event_type="퇴직",
            occurred_at=last.tx_date,
            confidence=0.6,
            reason=f"'{last.counterparty}' 급여 입금이 {months_idle}개월째 없어요. 퇴직/휴직 중이신가요?",
            evidence_tx_ids=[last.tx_id],
        )
    ]


def detect_independence(txs: List[Transaction]) -> List[EventCandidate]:
    """신규 월세 정기 출금 = 독립(월세) / 보증금 대규모 출금 = 독립(전세)."""
    candidates: List[EventCandidate] = []

    rent_sources = _group_by_counterparty(txs, "월세")
    for name, rows in rent_sources.items():
        valid = [t for t in rows if RENT_MIN_AMOUNT <= -t.amount <= RENT_MAX_AMOUNT]
        months = {_month_key(t.tx_date) for t in valid}
        if len(months) >= RECURRING_MIN_COUNT:
            candidates.append(
                EventCandidate(
                    event_type="독립(월세)",
                    occurred_at=valid[0].tx_date,
                    confidence=0.9,
                    reason=(
                        f"'{name}'로 매월 {abs(valid[0].amount):,}원이 {len(months)}개월 연속 빠져나갔어요. "
                        "월세 계약을 시작하셨나요?"
                    ),
                    evidence_tx_ids=[t.tx_id for t in valid[:3]],
                )
            )

    for tx in txs:
        if tx.amount >= 0:
            continue
        text = f"{tx.counterparty} {tx.memo or ''}"
        if any(k in text for k in DEPOSIT_KEYWORDS) and -tx.amount >= 10_000_000:
            candidates.append(
                EventCandidate(
                    event_type="독립(전세)",
                    occurred_at=tx.tx_date,
                    confidence=0.75,
                    reason=f"'{tx.counterparty}'로 {abs(tx.amount):,}원 보증금성 지출이 있었어요. 전세 계약을 하셨나요?",
                    evidence_tx_ids=[tx.tx_id],
                )
            )
    return candidates


def _detect_by_keyword(
    txs: List[Transaction], keywords, event_type: str, confidence: float, question: str
) -> List[EventCandidate]:
    hits = [
        t
        for t in txs
        if t.amount < 0 and any(k in f"{t.counterparty} {t.memo or ''}" for k in keywords)
    ]
    if not hits:
        return []
    hits.sort(key=lambda t: t.tx_date)
    return [
        EventCandidate(
            event_type=event_type,
            occurred_at=hits[0].tx_date,
            confidence=confidence,
            reason=f"'{hits[0].counterparty}' 등 {len(hits)}건의 결제가 확인됐어요. {question}",
            evidence_tx_ids=[t.tx_id for t in hits[:3]],
        )
    ]


def detect_marriage(txs: List[Transaction]) -> List[EventCandidate]:
    return _detect_by_keyword(txs, WEDDING_KEYWORDS, "결혼", 0.7, "결혼을 준비 중이신가요?")


def detect_childbirth(txs: List[Transaction]) -> List[EventCandidate]:
    return _detect_by_keyword(txs, MATERNITY_KEYWORDS, "출산", 0.7, "출산을 준비 중이신가요?")


def detect_startup(txs: List[Transaction]) -> List[EventCandidate]:
    return _detect_by_keyword(txs, STARTUP_KEYWORDS, "창업", 0.65, "사업을 시작하셨나요?")


# 등록된 감지기 목록. 새 규칙을 추가하면 여기에만 넣으면 된다.
DETECTORS = (
    detect_employment,
    detect_job_change,
    detect_income_stop,
    detect_independence,
    detect_marriage,
    detect_childbirth,
    detect_startup,
)


def detect_all(
    txs: List[Transaction],
    confirmed_events: Optional[List[tuple]] = None,
) -> List[EventCandidate]:
    """전체 감지기를 돌려 후보를 모은다.

    confirmed_events: [(event_type, occurred_at), ...] — 이미 확정된 이벤트 목록.
    중복 질문을 막되, **시점을 함께 본다**. 같은 '취업'이라도 이미 확정된 취업보다 뒤에 일어난
    급여 개시라면 재취업이므로 다시 후보로 올려야 한다.
    (퇴직 후 재취업이 변동소득 청년의 전형적인 패턴이라 이 구분이 실제로 중요하다)
    """
    latest_confirmed: Dict[str, date] = {}
    for event_type, occurred_at in confirmed_events or []:
        prev = latest_confirmed.get(event_type)
        if prev is None or occurred_at > prev:
            latest_confirmed[event_type] = occurred_at

    results: List[EventCandidate] = []
    for detector in DETECTORS:
        for cand in detector(txs):
            seen_at = latest_confirmed.get(cand.event_type)
            if seen_at is not None and cand.occurred_at <= seen_at:
                continue
            results.append(cand)

    results.sort(key=lambda c: (c.occurred_at, -c.confidence))
    return results