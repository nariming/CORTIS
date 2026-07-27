"""
합성 코호트 이벤트 시퀀스 더미 데이터.

실제로는 개발자1이 LLM으로 생성한 300개 시퀀스를 MySQL에 적재하지만,
개발자2(AI 파이프라인)는 그 전까지 이 더미셋으로 개발/테스트를 진행한다.

각 시퀀스: 확정된 이벤트 히스토리(순서 있는 리스트) + 그 다음 실제 발생한 이벤트.
이벤트 타입은 기획서에서 규칙 감지 가능한 것으로 추린 8종만 사용:
  대학생, 졸업, 취업, 이직, 퇴직, 독립(전세), 독립(월세), 결혼, 출산, 창업, 내집마련
  (문서에서 "감지 가능" 그룹인 신규이체처/이체중단/금액변화 신호로 잡히는 것들)
"""

COHORT_SEQUENCES = [
    {"history": ["대학생", "졸업", "취업"], "next_event": "독립(월세)"},
    {"history": ["대학생", "졸업", "취업"], "next_event": "이직"},
    {"history": ["대학생", "졸업", "취업", "독립(월세)"], "next_event": "결혼"},
    {"history": ["대학생", "졸업", "취업", "이직"], "next_event": "독립(전세)"},
    {"history": ["취업"], "next_event": "독립(월세)"},
    {"history": ["취업"], "next_event": "이직"},
    {"history": ["취업"], "next_event": "결혼"},
    {"history": ["이직", "취업"], "next_event": "결혼"},
    {"history": ["이직", "취업"], "next_event": "독립(전세)"},
    {"history": ["취업", "독립(전세)"], "next_event": "결혼"},
    {"history": ["취업", "독립(전세)"], "next_event": "내집마련"},
    {"history": ["취업", "결혼"], "next_event": "출산"},
    {"history": ["취업", "결혼"], "next_event": "내집마련"},
    {"history": ["창업"], "next_event": "결혼"},
    {"history": ["창업", "취업"], "next_event": "독립(월세)"},
    {"history": ["퇴직", "취업"], "next_event": "이직"},
    {"history": ["대학생", "졸업", "취업", "결혼"], "next_event": "출산"},
    {"history": ["대학생", "졸업", "창업"], "next_event": "취업"},
    {"history": ["취업", "독립(월세)", "이직"], "next_event": "독립(전세)"},
    {"history": ["취업", "독립(월세)"], "next_event": "이직"},
]

# 데모 대비용: 히스토리가 서로 다른 유저 A/B가 같은 "취업" 이벤트를 겪었을 때
# 검색 결과(따라서 예측)가 달라지는 걸 보여주기 위한 페르소나 두 명
DEMO_USER_A = {
    "user_id": "demo_user_A",
    "confirmed_history": ["대학생", "졸업", "취업"],  # 갓 졸업 후 첫 취업
}

DEMO_USER_B = {
    "user_id": "demo_user_B",
    "confirmed_history": ["이직", "취업"],  # 이직 경험 있는 재취업
}
