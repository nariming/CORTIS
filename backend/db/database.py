"""SQLAlchemy 엔진 / 세션 / FastAPI 의존성."""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend import config

engine = create_engine(
    config.database_url(),
    pool_pre_ping=True,   # OneDrive/노트북 절전 등으로 커넥션이 끊겨도 자동 복구
    pool_recycle=3600,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def get_db():
    """FastAPI 라우터용 의존성."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    """스크립트(시드 등)용 컨텍스트 매니저."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
