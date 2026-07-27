-- =====================================================================
-- CORTIS MySQL 스키마 (Source of Truth)
--
-- 설계 원칙
--   1. MySQL이 진실의 원천. 유저/대출/정책/거래내역/확정 이벤트 로그를 모두 여기에 정규화해 보관한다.
--   2. 코호트 임베딩 벡터도 MySQL에 JSON 컬럼으로 함께 저장한다.
--      (코호트 300건 규모에서는 전용 벡터DB 없이 numpy 코사인 유사도로 충분 — C파트 similarity.py)
--      => MySQL = 저장/원천, numpy = 검색 인덱스 라는 역할 분리.
--   3. life_events(확정 이벤트 로그)가 C엔진 입력의 원천이고,
--      predictions(예측 결과) / policy_match_results(A파트 산출물)가 그 출력의 원천이다.
--      이 셋이 시간순으로 쌓이는 것 자체가 "한 번 쓰고 끝나는 챗봇이 아니라 계속 재판단하는 에이전트"의 증거가 된다.
--
-- 실행: mysql -u root -p < schema.sql
--       (또는 python -m backend.db.seed.run_all 이 자동으로 실행)
-- =====================================================================

CREATE DATABASE IF NOT EXISTS cortis
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE cortis;

-- 개발 편의를 위한 재생성 순서 (FK 역순)
DROP TABLE IF EXISTS policy_match_results;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS life_events;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS user_loans;
DROP TABLE IF EXISTS cohort_sequences;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS loan_products;
DROP TABLE IF EXISTS users;


-- ---------------------------------------------------------------------
-- 1. users : 유저 프로필
--    정책 자격 판정(A파트)에 바로 쓰이도록 자격요건과 대응되는 컬럼을 정형으로 갖는다.
-- ---------------------------------------------------------------------
CREATE TABLE users (
    user_id            VARCHAR(32)  NOT NULL,
    name               VARCHAR(50)  NOT NULL,
    birth_year         SMALLINT     NOT NULL,
    -- 변동소득 청년이 핵심 타깃이므로 고용형태를 정형값으로 관리
    employment_type    ENUM('정규직','계약직','프리랜서','플랫폼노동','자영업','무직','학생') NOT NULL,
    monthly_income_avg INT          NOT NULL DEFAULT 0 COMMENT '최근 6개월 평균 월소득(원)',
    income_volatility  DECIMAL(5,3) NOT NULL DEFAULT 0 COMMENT 'MAD/중앙값 = 소득 변동성 지표(B파트 입력)',
    marital_status     ENUM('미혼','기혼') NOT NULL DEFAULT '미혼',
    housing_type       ENUM('부모동거','월세','전세','자가','기숙사') NOT NULL DEFAULT '부모동거',
    region_code        VARCHAR(10)  NOT NULL DEFAULT '11' COMMENT '법정동 시도코드 (11=서울)',
    credit_score       SMALLINT     NULL COMMENT 'KCB 기준 점수',
    created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 2. loan_products : KB 대출 상품 카탈로그
-- ---------------------------------------------------------------------
CREATE TABLE loan_products (
    product_id    VARCHAR(32)  NOT NULL,
    product_name  VARCHAR(100) NOT NULL,
    product_type  ENUM('전월세자금','신용대출','학자금','사업자','마이너스통장') NOT NULL,
    min_rate      DECIMAL(4,2) NOT NULL,
    max_rate      DECIMAL(4,2) NOT NULL,
    max_amount    BIGINT       NOT NULL,
    target_desc   VARCHAR(255) NULL,
    PRIMARY KEY (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 3. user_loans : 유저 보유 대출 (B파트 상환계획 재설계의 입력)
-- ---------------------------------------------------------------------
CREATE TABLE user_loans (
    loan_id         VARCHAR(32)  NOT NULL,
    user_id         VARCHAR(32)  NOT NULL,
    product_id      VARCHAR(32)  NOT NULL,
    principal       BIGINT       NOT NULL COMMENT '최초 대출원금',
    balance         BIGINT       NOT NULL COMMENT '현재 잔액',
    interest_rate   DECIMAL(4,2) NOT NULL,
    monthly_payment INT          NOT NULL,
    due_day         TINYINT      NOT NULL COMMENT '매월 상환일(1~28)',
    started_at      DATE         NOT NULL,
    maturity_at     DATE         NOT NULL,
    status          ENUM('정상','연체','완제') NOT NULL DEFAULT '정상',
    PRIMARY KEY (loan_id),
    KEY idx_user_loans_user (user_id),
    CONSTRAINT fk_loans_user    FOREIGN KEY (user_id)    REFERENCES users(user_id)         ON DELETE CASCADE,
    CONSTRAINT fk_loans_product FOREIGN KEY (product_id) REFERENCES loan_products(product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 4. policies : 정책금융/청년정책 DB + 자격요건 정형화
--    자격요건을 자연어 설명이 아니라 컬럼으로 쪼개 두는 게 핵심.
--    그래야 A파트가 RAG 없이 SQL WHERE 절만으로 "자격 있음/없음"을 결정론적으로 판정할 수 있다.
--    trigger_events 는 "이 이벤트가 확정되면 이 정책 자격을 재판단하라"는 역인덱스 역할.
-- ---------------------------------------------------------------------
CREATE TABLE policies (
    policy_id           VARCHAR(32)  NOT NULL,
    policy_name         VARCHAR(150) NOT NULL,
    provider            VARCHAR(50)  NOT NULL COMMENT '주택도시기금 / 서울시 / 고용노동부 등',
    category            ENUM('주거','고용','창업','결혼출산','금융','교육') NOT NULL,
    benefit_summary     VARCHAR(500) NOT NULL,
    apply_url           VARCHAR(255) NULL,
    -- --- 정형 자격요건 (NULL = 해당 조건 없음) ---
    min_age             TINYINT      NULL,
    max_age             TINYINT      NULL,
    max_annual_income   BIGINT       NULL COMMENT '연소득 상한(원)',
    allowed_employment  JSON         NULL COMMENT '["프리랜서","플랫폼노동"] 형태. NULL이면 고용형태 무관',
    allowed_housing     JSON         NULL COMMENT '["월세","전세"] 형태',
    allowed_marital     JSON         NULL COMMENT '["미혼","기혼"] 형태',
    region_code         VARCHAR(10)  NULL COMMENT 'NULL이면 전국',
    -- --- C엔진 연동 ---
    trigger_events      JSON         NOT NULL COMMENT '["독립(월세)","결혼"] — 이 이벤트 확정/예측 시 자격 재판단 대상',
    priority            TINYINT      NOT NULL DEFAULT 5 COMMENT '동점일 때 노출 우선순위(1이 높음)',
    PRIMARY KEY (policy_id),
    KEY idx_policies_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 5. transactions : 합성 거래내역 (규칙기반 이벤트 감지의 입력)
--    amount 음수 = 출금, 양수 = 입금
-- ---------------------------------------------------------------------
CREATE TABLE transactions (
    tx_id       BIGINT       NOT NULL AUTO_INCREMENT,
    user_id     VARCHAR(32)  NOT NULL,
    tx_date     DATE         NOT NULL,
    amount      BIGINT       NOT NULL COMMENT '음수=출금, 양수=입금',
    counterparty VARCHAR(100) NOT NULL COMMENT '이체처/가맹점명',
    category    VARCHAR(30)  NOT NULL COMMENT '급여/월세/보험/카드/의료/기타',
    memo        VARCHAR(255) NULL,
    PRIMARY KEY (tx_id),
    KEY idx_tx_user_date (user_id, tx_date),
    CONSTRAINT fk_tx_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 6. life_events : 확정된 생애주기 이벤트 로그 (C엔진 입력의 원천)
--    기획 캡처의 [이벤트명, 발생 시점(월 offset), 이전 이벤트까지의 간격] 구조를 그대로 컬럼화.
--    status로 감지(detected) / 확정(confirmed) / 기각(rejected)을 구분해서,
--    "규칙기반 감지 → 사용자 확인 → 확정" 흐름이 데이터로 남게 한다.
-- ---------------------------------------------------------------------
CREATE TABLE life_events (
    event_id        BIGINT       NOT NULL AUTO_INCREMENT,
    user_id         VARCHAR(32)  NOT NULL,
    event_type      VARCHAR(30)  NOT NULL COMMENT '취업/이직/독립(월세)/결혼 등',
    occurred_at     DATE         NOT NULL,
    offset_month    SMALLINT     NOT NULL DEFAULT 0 COMMENT '유저 첫 이벤트 기준 경과 개월',
    prev_gap_month  SMALLINT     NULL     COMMENT '직전 이벤트까지의 간격(개월). 첫 이벤트면 NULL',
    status          ENUM('detected','confirmed','rejected') NOT NULL DEFAULT 'detected',
    detected_by     ENUM('rule','user','seed') NOT NULL DEFAULT 'rule',
    confidence      DECIMAL(3,2) NULL COMMENT '규칙기반 감지 신뢰도 0~1',
    evidence        JSON         NULL COMMENT '감지 근거 거래내역 tx_id 목록 등',
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id),
    KEY idx_events_user_status (user_id, status, occurred_at),
    CONSTRAINT fk_events_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 7. cohort_sequences : 합성 코호트 300개 + 사전 임베딩 벡터
--    C파트 CohortIndex.load_from_mysql_rows() 가 이 테이블을 그대로 읽는다.
--    embedding_vector를 미리 저장해 두므로 서버 기동 때마다 재임베딩할 필요가 없다.
-- ---------------------------------------------------------------------
CREATE TABLE cohort_sequences (
    cohort_id           INT          NOT NULL AUTO_INCREMENT,
    history_json        JSON         NOT NULL COMMENT '["대학생","졸업","취업"]',
    event_history_text  VARCHAR(500) NOT NULL COMMENT '임베딩에 실제로 넣은 문장',
    next_event          VARCHAR(30)  NOT NULL,
    history_length      TINYINT      NOT NULL,
    embedding_vector    JSON         NOT NULL COMMENT 'float 배열. numpy로 로드해 코사인 유사도 계산',
    embedding_model     VARCHAR(50)  NOT NULL COMMENT 'offline-hash-64 / text-embedding-3-small',
    embedding_dim       SMALLINT     NOT NULL,
    created_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cohort_id),
    KEY idx_cohort_next (next_event),
    KEY idx_cohort_model (embedding_model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 8. predictions : C엔진 예측 결과 로그
--    trigger_event_id = "어떤 이벤트가 확정돼서 이 재예측이 돌았는가" (agentic 순환의 추적 기록)
--    matched_cohorts_json = 근거로 쓴 유사 사례 top-k 원본 → "이 숫자 어디서 나왔냐"에 즉답 가능
-- ---------------------------------------------------------------------
CREATE TABLE predictions (
    prediction_id        BIGINT       NOT NULL AUTO_INCREMENT,
    user_id              VARCHAR(32)  NOT NULL,
    trigger_event_id     BIGINT       NULL,
    input_history_json   JSON         NOT NULL COMMENT '예측 시점의 확정 히스토리 스냅샷',
    predictions_json     JSON         NOT NULL COMMENT '[{event, evidence_count, reasoning}]',
    confidence_level     ENUM('high','medium','low') NOT NULL,
    confidence_note      VARCHAR(500) NULL,
    matched_cohorts_json JSON         NULL COMMENT '검색된 top-k 코호트 + 유사도',
    created_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_id),
    KEY idx_pred_user (user_id, created_at),
    CONSTRAINT fk_pred_user  FOREIGN KEY (user_id)          REFERENCES users(user_id)      ON DELETE CASCADE,
    CONSTRAINT fk_pred_event FOREIGN KEY (trigger_event_id) REFERENCES life_events(event_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------
-- 9. policy_match_results : A파트 산출물
--    "예측된 이벤트 때문에 새로 자격이 생긴 정책"을 기록 → 데모에서 유저 A/B 대비를 보여주는 근거
-- ---------------------------------------------------------------------
CREATE TABLE policy_match_results (
    match_id      BIGINT      NOT NULL AUTO_INCREMENT,
    user_id       VARCHAR(32) NOT NULL,
    prediction_id BIGINT      NULL,
    policy_id     VARCHAR(32) NOT NULL,
    basis_event   VARCHAR(30) NOT NULL COMMENT '이 매칭을 유발한 이벤트',
    status        ENUM('newly_eligible','eligible','lost','not_eligible') NOT NULL,
    reason        VARCHAR(500) NOT NULL COMMENT '어떤 조건을 충족/불충족했는지 사람이 읽는 설명',
    created_at    DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (match_id),
    KEY idx_match_user (user_id, created_at),
    CONSTRAINT fk_match_user   FOREIGN KEY (user_id)       REFERENCES users(user_id)          ON DELETE CASCADE,
    CONSTRAINT fk_match_policy FOREIGN KEY (policy_id)     REFERENCES policies(policy_id),
    CONSTRAINT fk_match_pred   FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
