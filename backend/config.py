"""
백엔드 설정. .env 파일 또는 환경변수에서 읽는다.

.env.example 을 복사해서 .env 로 만들고 로컬 MySQL 비밀번호만 채우면 바로 돈다.
(.env 는 .gitignore 로 커밋되지 않으므로 각자 로컬 값 사용)
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = BASE_DIR.parent                       # 저장소 루트

load_dotenv(BASE_DIR / ".env")


MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "cortis")

# 임베딩 백엔드: offline(해시, API키 불필요) | openai
# 시드 스크립트가 코호트를 임베딩할 때, 그리고 C파트가 검색할 때 같은 값을 써야 벡터가 호환된다.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "offline")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))

# 유사 코호트 검색 top-k
COHORT_TOP_K = int(os.getenv("COHORT_TOP_K", "5"))

# 히스토리가 이 개수 미만이면 콜드스타트로 보고 신뢰도를 낮춰 표시
COLD_START_THRESHOLD = int(os.getenv("COLD_START_THRESHOLD", "2"))


def database_url(with_db: bool = True) -> str:
    """SQLAlchemy 접속 URL.

    with_db=False 는 아직 cortis 데이터베이스가 없을 때(최초 스키마 생성) 사용한다.
    비밀번호에 @, # 같은 특수문자가 있어도 깨지지 않게 quote_plus 처리.
    """
    auth = MYSQL_USER
    if MYSQL_PASSWORD:
        auth += f":{quote_plus(MYSQL_PASSWORD)}"
    base = f"mysql+pymysql://{auth}@{MYSQL_HOST}:{MYSQL_PORT}"
    return f"{base}/{MYSQL_DB}?charset=utf8mb4" if with_db else f"{base}/?charset=utf8mb4"
