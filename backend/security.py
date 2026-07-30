"""
API 키 기반 접근 통제.

왜 JWT/OAuth가 아니라 API 키인가:
  JWT/OAuth는 "개별 유저 로그인"을 전제로 한다 (아이디/비밀번호 또는 소셜 로그인 ->
  토큰 발급 -> 세션 유지). 근데 CORTIS는 아직 회원가입/로그인 시스템 자체가 없고,
  user_id는 미리 심어둔 데모 페르소나다. 없는 로그인 체계를 API 인증 하나 때문에
  새로 만드는 건 배보다 배꼽이 큰 일이라, "이 API를 호출할 자격이 있는 클라이언트인가"만
  확인하는 API 키 방식을 택했다.

  실사용자 온보딩 단계(KB 마이데이터 연동 등)에서는 이 디펜던시를 JWT 검증으로
  교체하면 되고, 라우터 쪽 코드는 안 건드려도 된다 (Depends(verify_api_key) 자리만
  Depends(verify_jwt) 등으로 바꾸면 됨 - 확장 지점을 여기 하나로 모아둔 이유).
"""

from fastapi import Header, HTTPException, status

from backend import config


def verify_api_key(x_api_key: str = Header(default="", alias="X-API-Key")):
    """라우터에 Depends(verify_api_key)로 붙여서 쓴다.

    config.API_KEY가 비어있으면(.env에 설정 안 함) 로컬 개발 편의상 통과시킨다 -
    단, 이 경우 서버 시작 시 경고를 남겨서 "인증이 꺼져있다"를 놓치지 않게 한다 (main.py 참고).
    """
    if not config.API_KEY:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 API 키입니다. X-API-Key 헤더를 확인하세요.",
        )