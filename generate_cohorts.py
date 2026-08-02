"""
합성 코호트 300개 시퀀스 생성기.

통계청/청년패널 등이 보여주는 대략적인 생애주기 순서
(대학생->졸업->취업->독립->전세->결혼->출산)를 "느슨한 가이드"로 삼되,
실제 사람들처럼 순서가 어긋나는 경우(이직 먼저, 결혼 먼저 등)를 의도적으로 섞어
개인차가 반영된 다양한 시퀀스 300개를 만든다.

이 파일이 하는 일은 "그럴듯한 합성 데이터를 다양하게 만드는 것"뿐이고,
실제 서비스 런타임 로직(예측)에 확률모델을 쓰는 게 아니므로
이전에 폐기하기로 한 HMM/전이확률표 결정과 모순되지 않는다.
(여기서 쓰는 random.choices 가중치는 "그럴듯한 학습/평가용 데이터를 만들기 위한 저작 도구"일 뿐,
 실제 예측 로직은 여전히 코호트 검색+LLM 추론임)

TRANSITION_WEIGHTS 방향성 근거 (2026.7 확인, 통계청/국가데이터처 공식 조사):
  - 졸업 -> 첫 취업까지 평균 11.2개월 (통계청 경제활동인구조사 청년층 부가조사)
    -> "졸업" 다음 "취업" 가중치를 압도적으로 높게 잡는 것이 타당함
  - 첫 직장 평균 근속기간 1년 6.8개월 (위와 동일 자료)
    -> "취업" 이후 "이직"이 비교적 이른 시점에 흔한 전환이라는 근거. 실제로 "취업"의
       이직/독립 가중치(각 3)가 "결혼" 가중치(2)보다 높게 설정된 현재 값과 방향이 일치함
  - 평균 초혼연령 남 33.9세 / 여 31.6세 (국가데이터처 2025년 혼인·이혼통계)
    -> 취업 시작 연령(20대 중반) 대비 결혼은 상당한 시간 차를 두고 발생하는 이벤트.
       "취업"에서 "결혼"으로 곧장 가는 가중치를 낮게 잡은 현재 설계와 방향이 일치함

검증 결과: 위 통계로 방향성을 대조했을 때 기존 가중치가 이미 크게 어긋나지 않아,
숫자 자체를 대폭 수정하기보다 근거를 명시하는 방식으로 보강함 (실제 통계 기반 검증을 거쳤다는
근거를 남기는 것이 목적이지, 근거 없이 숫자만 바꾸는 것은 오히려 임의성을 더할 뿐임).

State/Transaction feature/전이간격/필요자금 확장 (2026.8 추가)
  - 이 코호트 300개는 실거래 시뮬레이션 결과가 아니라 검색 코퍼스다. 즉 "가상의 300명분
    거래내역을 만들어서 거기서 feature를 추출"하는 우회 없이, State/거래 feature 값 자체를
    next_event와 논리적으로 맞물리게 직접 합성한다 — history/next_event를 랜덤 생성하는 것과
    동일한 저작 방식이다.
  - EVENT_INTERVAL_MONTHS, CASH_NEED_KRW는 가능한 한 실제 통계를 앵커로 삼았고(각 상수 옆에
    출처 명시), 구체 통계가 없는 항목은 "근사치"라고 값 자체에 표기해 두었다. 이 값들은
    실 서비스 전환 시 KB 실거래 통계로 교체되어야 할 항목이다.
  - cash_need는 "필요자금" 개념이 성립하는 이벤트(독립/결혼/출산/창업/내집마련)에만 부여하고,
    나머지(취업/이직/퇴직/졸업/휴학/대학생)는 의도적으로 None으로 둔다 — 개념이 없는 이벤트에
    억지로 숫자를 배정하지 않는다.

실행: python generate_cohorts.py
결과: data/cohort_sequences_300.py 에 COHORT_SEQUENCES_300 리스트로 저장
"""

import random

random.seed(42)  # 재현 가능하게 고정

# 이벤트 카테고리 (기획서에서 규칙 감지 가능한 것으로 추린 것)
EVENT_POOL = [
    "대학생", "졸업", "휴학", "취업", "이직", "퇴직",
    "창업", "독립(월세)", "독립(전세)", "내집마련",
    "결혼", "출산",
]

# "다음에 뭐가 올 확률이 높은지"에 대한 느슨한 가중치 (완전 고정 순서가 아니라 참고용 가중치)
# 형태: {현재까지의 마지막 이벤트: [(다음 이벤트, 가중치), ...]}
TRANSITION_WEIGHTS = {
    "대학생": [("졸업", 5), ("휴학", 1), ("창업", 1)],
    "휴학": [("졸업", 3), ("창업", 1), ("취업", 1)],
    "졸업": [("취업", 5), ("창업", 2), ("퇴직", 0.1)],
    "취업": [("독립(월세)", 3), ("이직", 3), ("독립(전세)", 1), ("결혼", 2), ("퇴직", 1)],
    "이직": [("독립(월세)", 2), ("독립(전세)", 2), ("결혼", 2), ("이직", 1), ("퇴직", 1)],
    "퇴직": [("취업", 4), ("창업", 2)],
    "창업": [("취업", 1), ("독립(월세)", 2), ("결혼", 2)],
    "독립(월세)": [("독립(전세)", 3), ("결혼", 3), ("이직", 2), ("내집마련", 1)],
    "독립(전세)": [("결혼", 3), ("내집마련", 2), ("이직", 1)],
    "내집마련": [("결혼", 3), ("출산", 1)],
    "결혼": [("출산", 5), ("내집마련", 2), ("이직", 1)],
    "출산": [("이직", 1), ("내집마련", 1)],  # 종결에 가깝지만 소수 케이스 허용
}

START_EVENTS = ["대학생", "취업", "이직", "창업", "졸업"]
START_WEIGHTS = [5, 3, 1, 1, 1]  # 대부분 대학생부터 시작하되, 소수는 중간부터 시작(경력 편입 등 표현)


def generate_one_sequence(min_len=2, max_len=5):
    length = random.randint(min_len, max_len + 1)  # +1개는 next_event로 쓸 것
    start = random.choices(START_EVENTS, weights=START_WEIGHTS, k=1)[0]
    seq = [start]

    while len(seq) < length:
        last = seq[-1]
        options = TRANSITION_WEIGHTS.get(last)
        if not options:
            break
        # 10% 확률로 가중치 무시하고 완전 랜덤 이벤트 삽입 (개인차/이례적 케이스 표현)
        if random.random() < 0.10:
            next_event = random.choice(EVENT_POOL)
        else:
            events, weights = zip(*options)
            next_event = random.choices(events, weights=weights, k=1)[0]
        seq.append(next_event)

    if len(seq) < 2:
        return None

    history = seq[:-1]
    next_event = seq[-1]
    return {"history": history, "next_event": next_event}


# ---------------------------------------------------------------------------
# State / Transaction feature / 전이간격 / 필요자금 합성
# ---------------------------------------------------------------------------

# 시작 이벤트별 대략적인 시작 연령(세) — 나이 시뮬레이션의 기준점. 구체 통계보다는
# "대학생=20대 초반, 취업/이직으로 시작=20대 중후반 경력 편입"이라는 상식적 근사치다.
START_AGE_YEARS = {"대학생": 20, "졸업": 25, "취업": 25, "이직": 27, "창업": 26}
DEFAULT_START_AGE_YEARS = 24

# 이벤트 전이 간격(개월) 평균/표준편차. 통계 근거가 있는 쌍은 그대로 인용하고,
# 근거가 없는 쌍은 DEFAULT_INTERVAL_MONTHS로 처리한다(값 자체가 근사치임을 이 상수명으로 명시).
EVENT_INTERVAL_MONTHS = {
    ("졸업", "취업"): (11.2, 4.0),   # 통계청 경제활동인구조사 청년층 부가조사
    ("취업", "이직"): (20.8, 8.0),   # 첫 직장 평균 근속기간 1년 6.8개월(위와 동일 자료)
    ("취업", "결혼"): (60.0, 24.0),  # 평균 초혼연령(남 33.9/여 31.6)과 취업 시작 연령의 차이를 근사
}
DEFAULT_INTERVAL_MONTHS = (18.0, 9.0)  # 구체 통계가 없는 전이 쌍의 근사치 (발표 시 근사임을 명시)

# 이벤트 실현에 필요한 자금(원) 평균/표준편차/출처. 이 표에 없는 next_event(취업/이직/퇴직/
# 졸업/휴학/대학생)는 "필요자금" 개념 자체가 성립하지 않으므로 cash_need를 None으로 둔다.
CASH_NEED_KRW = {
    "독립(전세)": (
        110_000_000, 30_000_000,
        "청년버팀목전세자금대출 한도(최대 2억원)·대상 보증금 상한(3억원 이하) 참고 근사치",
    ),
    "독립(월세)": (
        10_000_000, 4_000_000,
        "서울시 청년월세지원 자격기준(보증금 8천만원 이하) 참고, 초기 보증금+이사비용 근사치",
    ),
    "결혼": (
        68_000_000, 20_000_000,
        "결혼정보회사 듀오 2025년 결혼비용 보고서 기준 예식+혼수 평균(주거비 제외)",
    ),
    "내집마련": (
        300_000_000, 80_000_000,
        "구체 통계 부재 - 수도권 소형 아파트 시세 근사치 (발표 시 근사임을 명시할 것)",
    ),
    "출산": (
        3_000_000, 1_000_000,
        "출산 준비물·병원비 근사치 - 구체 통계 부재",
    ),
    "창업": (
        30_000_000, 15_000_000,
        "청년창업자금 대출 한도대 참고 근사치 - 구체 통계 부재",
    ),
}

INCOME_RANGE_KRW = {  # 고용형태별 월소득 평균/표준편차(원) 근사치
    "정규직": (3_200_000, 700_000),
    "계약직": (2_600_000, 600_000),
    "프리랜서": (2_400_000, 900_000),
    "자영업": (2_800_000, 1_200_000),
    "무직": (300_000, 200_000),   # 실업급여 등 근사
    "학생": (350_000, 250_000),   # 용돈/아르바이트 근사
}

# next_event별 저축 추세 편향 — "목돈을 준비 중인 사람일수록 저축이 늘어난다"는 상식을 반영.
# 편향이 없는(=필요자금 개념이 약한) 이벤트는 DEFAULT_SAVING_GROWTH를 쓴다.
SAVING_GROWTH_BIAS = {
    "결혼": (0.15, 0.15),
    "내집마련": (0.12, 0.15),
    "독립(전세)": (0.10, 0.15),
}
DEFAULT_SAVING_GROWTH = (0.0, 0.15)

# next_event별 "가구·가전 구매 증가" 신호 발생 확률 — 독립을 앞둔 코호트일수록 높게 설정.
FURNITURE_SIGNAL_PROB = {"독립(월세)": 0.6, "독립(전세)": 0.6}
DEFAULT_FURNITURE_PROB = 0.05


def _derive_employment_type(history):
    """history 마지막 이벤트로부터 현재 고용형태를 근사한다."""
    if not history:
        return "학생"
    last = history[-1]
    if last == "퇴직":
        return "무직"
    if last == "창업":
        return "자영업"
    if last in ("취업", "이직") or any(e in history for e in ("취업", "이직", "창업")):
        return random.choices(["정규직", "계약직", "프리랜서"], weights=[6, 2, 2], k=1)[0]
    return "학생"


def _derive_housing_type(history):
    if "내집마련" in history:
        return "자가"
    if "독립(전세)" in history:
        return "전세"
    if "독립(월세)" in history:
        return "월세"
    if "대학생" in history and len(history) <= 2:
        return random.choices(["부모동거", "기숙사"], weights=[7, 3], k=1)[0]
    return "부모동거"


def _derive_marital_status(history):
    return "기혼" if "결혼" in history else "미혼"


def _sample_interval_months(from_event, to_event):
    mean, std = EVENT_INTERVAL_MONTHS.get((from_event, to_event), DEFAULT_INTERVAL_MONTHS)
    return max(1.0, random.gauss(mean, std))


def _simulate_age(history):
    """history를 시작부터 순서대로 훑으며 전이 간격을 누적해 현재 나이를 근사한다."""
    start = history[0]
    total_months = START_AGE_YEARS.get(start, DEFAULT_START_AGE_YEARS) * 12
    for i in range(len(history) - 1):
        total_months += _sample_interval_months(history[i], history[i + 1])
    return max(19, round(total_months / 12))


def _sample_income(employment_type):
    mean, std = INCOME_RANGE_KRW.get(employment_type, (2_500_000, 800_000))
    return max(0, round(random.gauss(mean, std) / 10_000) * 10_000)  # 만원 단위 근사


def _sample_credit_score(employment_type, has_loan):
    base = 720
    if employment_type in ("무직", "프리랜서"):
        base -= 40
    if has_loan:
        base -= 15
    return max(400, min(950, round(random.gauss(base, 45))))


def _sample_loan_portfolio(history, housing_type):
    """history/housing_type과 개연성 있는 대출 보유 현황을 합성한다.

    실제 상환구조 시뮬레이션(원리금균등 등)은 pipeline/portfolio.py가 담당하는 영역이라,
    여기서는 State 문장/DSR 근사에 필요한 수준(잔액·월상환액)까지만 만든다.
    """
    loans = []
    if ("대학생" in history or "졸업" in history) and random.random() < 0.4:
        principal = random.randint(8_000_000, 20_000_000)
        loans.append({
            "product_type": "학자금대출", "balance": principal,
            "interest_rate": 2.3, "monthly_payment": round(principal / 60),
        })
    if housing_type == "전세":
        mean, std, _ = CASH_NEED_KRW["독립(전세)"]
        principal = max(20_000_000, round(random.gauss(mean, std)))
        loans.append({
            "product_type": "전월세자금대출", "balance": principal,
            "interest_rate": 2.0, "monthly_payment": round(principal * 0.02 / 12),
        })
    elif housing_type == "월세" and random.random() < 0.3:
        principal = random.randint(3_000_000, 10_000_000)
        loans.append({
            "product_type": "청년전용보증부월세대출", "balance": principal,
            "interest_rate": 3.5, "monthly_payment": round(principal * 0.035 / 12),
        })
    if housing_type == "자가" and random.random() < 0.7:
        principal = random.randint(150_000_000, 300_000_000)
        loans.append({
            "product_type": "주택담보대출", "balance": principal,
            "interest_rate": 3.8, "monthly_payment": round(principal * 0.038 / 12),
        })
    return loans


def _sample_tx_features(next_event, employment_type, has_loan):
    """tx_features.py가 실제 유저 거래내역에서 계산하는 것과 같은 필드를 합성한다.

    소득 추세는 안정적 소득 자체가 없는 경우(학생/무직) None으로 둔다 —
    tx_features.py가 데이터 부족 시 None을 반환하는 것과 동일한 원칙.

    debt_growth도 같은 원칙: 대출 자체가 없으면 "대출상환" 거래 자체가 없어
    tx_features.py에서도 growth가 None으로 나오므로(_recent_prior_avg가 두 구간
    모두 비어 (None, None) 반환), has_loan=False면 여기서도 None으로 둔다.
    """
    income_growth = None if employment_type in ("학생", "무직") else round(random.gauss(0.01, 0.05), 3)
    expense_growth = round(random.gauss(0.0, 0.08), 3)
    saving_mean, saving_std = SAVING_GROWTH_BIAS.get(next_event, DEFAULT_SAVING_GROWTH)
    saving_growth = round(random.gauss(saving_mean, saving_std), 3)
    # 기존 대출이 있으면 상환액은 보통 일정하되(고정 원리금), 신규 대출 추가 실행이나
    # 변동금리 재산정 등으로 약간씩 늘어나는 방향의 완만한 편향을 준다.
    debt_growth = round(random.gauss(0.02, 0.06), 3) if has_loan else None
    cashflow_volatility = round(abs(random.gauss(0.15, 0.1)), 3)

    signals = []
    if random.random() < FURNITURE_SIGNAL_PROB.get(next_event, DEFAULT_FURNITURE_PROB):
        signals.append("가구·가전 구매 증가(독립 가능성)")

    return {
        "income_growth": income_growth,
        "expense_growth": expense_growth,
        "saving_growth": saving_growth,
        "debt_growth": debt_growth,
        "cashflow_volatility": cashflow_volatility,
        "signals": signals,
    }


def _sample_cash_need(next_event):
    if next_event not in CASH_NEED_KRW:
        return None, None
    mean, std, source = CASH_NEED_KRW[next_event]
    value = max(round(mean * 0.3), round(random.gauss(mean, std)))
    return value, source


def _sample_liquid_assets(monthly_income, employment_type):
    """여유자금(원) 근사치. 소득이 있을수록, 학생/무직이 아닐수록 여유자금이 쌓였을 가능성이 높다는
    상식을 반영 — 정확한 통계보다 State Embedding 문장 구조를 실제 UserState와 맞추는 목적이 크다."""
    if employment_type in ("학생", "무직"):
        mean = max(500_000, monthly_income * 2)
    else:
        mean = max(2_000_000, monthly_income * 3)
    return max(0, round(random.gauss(mean, mean * 0.5) / 10_000) * 10_000)


def attach_synthetic_state(seq: dict) -> dict:
    """history/next_event 하나에 State/거래 feature/전이간격/필요자금을 덧붙인다.

    실거래 시뮬레이션을 거치지 않고 값 자체를 next_event와 개연성 있게 직접 합성한다
    (파일 상단 "State/Transaction feature 확장" 절 참고). 확장 필드가 next_event와
    무관한 노이즈가 되지 않도록, employment_type/housing_type/marital_status는 history에서
    결정론적으로 파생시키고, tx_features/cash_need만 next_event에 편향된 분포에서 샘플링한다.
    """
    history = seq["history"]
    next_event = seq["next_event"]

    employment_type = _derive_employment_type(history)
    housing_type = _derive_housing_type(history)
    marital_status = _derive_marital_status(history)
    age = _simulate_age(history)
    monthly_income = _sample_income(employment_type)
    liquid_assets_krw = _sample_liquid_assets(monthly_income, employment_type)
    loans = _sample_loan_portfolio(history, housing_type)
    credit_score = _sample_credit_score(employment_type, has_loan=bool(loans))

    monthly_payment_total = sum(loan["monthly_payment"] for loan in loans)
    annual_income = monthly_income * 12
    # 대출이 없으면 DSR 자체가 정의되지 않으므로 0.0이 아니라 None
    # (pipeline/state_builder.py의 _current_dsr_pct()와 동일한 원칙).
    dsr_pct = round(monthly_payment_total * 12 / annual_income * 100, 1) if loans and annual_income > 0 else None

    tx_features = _sample_tx_features(next_event, employment_type, has_loan=bool(loans))
    event_interval_months = round(_sample_interval_months(history[-1], next_event))
    cash_need_krw, cash_need_source = _sample_cash_need(next_event)

    out = dict(seq)
    out["state"] = {
        "age": age,
        "employment_type": employment_type,
        "monthly_income": monthly_income,
        "housing_type": housing_type,
        "marital_status": marital_status,
        "credit_score": credit_score,
        "liquid_assets_krw": liquid_assets_krw,
        "loan_portfolio": loans,
        "dsr_pct": dsr_pct,
    }
    out["tx_features"] = tx_features
    out["event_interval_months"] = event_interval_months
    out["cash_need_krw"] = cash_need_krw
    out["cash_need_source"] = cash_need_source
    return out



def generate_all(n=300):
    sequences = []
    seen = set()
    attempts = 0
    while len(sequences) < n and attempts < n * 10:
        attempts += 1
        s = generate_one_sequence()
        if s is None:
            continue
        key = (tuple(s["history"]), s["next_event"])
        # 완전 중복은 스킵하되, 같은 history에 다른 next_event가 나오는 건 의도적으로 허용
        # (이게 바로 "순서가 정해져 있지 않다"를 데이터로 보여주는 부분)
        if key in seen:
            continue
        seen.add(key)
        sequences.append(attach_synthetic_state(s))
    return sequences


def save_as_python_module(sequences, path="data/cohort_sequences_300.py"):
    """생성 결과를 .py 모듈로 저장한다.

    json.dumps()가 아니라 pprint.pformat()을 쓴다 — None/True/False가 섞인 데이터를
    json.dumps()로 저장하면 JSON의 null/true/false가 그대로 텍스트에 들어가는데, 이건
    파이썬 문법이 아니라서(None/True/False가 맞음) 이 파일을 import하는 순간
    NameError로 깨진다. State/tx_features 확장으로 None 값이 등장하면서 실제로 이 문제가
    발생해 pprint 기반으로 교체했다.
    """
    from pprint import pformat

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            '"""가중치 기반 랜덤 생성(random.choices) 합성 코호트 300개.\n'
            "생성 로직: generate_cohorts.py (전이 가중치는 통계청/국가데이터처 공식 조사로\n"
            "방향성 검증됨, State/거래 feature/전이간격/필요자금 확장은 파일 상단 docstring의\n"
            '"State/Transaction feature 확장" 절 참고).\n"""\n\n'
        )
        f.write("COHORT_SEQUENCES_300 = ")
        f.write(pformat(sequences, indent=2, width=100, sort_dicts=False))
        f.write("\n")


if __name__ == "__main__":
    seqs = generate_all(300)
    save_as_python_module(seqs)
    print(f"생성 완료: {len(seqs)}개 시퀀스 -> data/cohort_sequences_300.py")

    # 간단 통계 출력
    from collections import Counter
    import statistics

    lengths = Counter(len(s["history"]) for s in seqs)
    print(f"히스토리 길이 분포: {dict(sorted(lengths.items()))}")
    starts = Counter(s["history"][0] for s in seqs)
    print(f"시작 이벤트 분포: {dict(starts)}")

    # State/feature 확장이 next_event와 개연성 있게 붙었는지 확인하는 검증용 통계
    print("\n--- State/feature 확장 검증 ---")
    for event in ("독립(월세)", "독립(전세)", "결혼", "내집마련"):
        subset = [s for s in seqs if s["next_event"] == event]
        if not subset:
            continue
        cash_needs = [s["cash_need_krw"] for s in subset if s["cash_need_krw"] is not None]
        furniture_rate = sum(1 for s in subset if s["tx_features"]["signals"]) / len(subset)
        saving_growths = [s["tx_features"]["saving_growth"] for s in subset]
        print(
            f"next_event='{event}' (n={len(subset)}): "
            f"cash_need 평균 {round(statistics.mean(cash_needs)):,}원, "
            f"가구가전 신호 비율 {furniture_rate:.0%}, "
            f"저축추세 평균 {statistics.mean(saving_growths):+.3f}"
        )