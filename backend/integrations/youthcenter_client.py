"""
온통청년(youthcenter.go.kr) 청년정책 오픈API 클라이언트.

⚠️ 중요한 한계: 이 코드를 작성한 개발 환경(샌드박스)에서는 이 API의 실제 서비스 포트
(www.youthcenter.go.kr:8080)로 아웃바운드 연결 자체가 막혀 있어(타임아웃), 실제 응답을
한 번도 받아보지 못한 채 공개 문서만 보고 작성했다. 즉:
  - 요청 URL/파라미터명(openApiVlak, display, pageIndex, query)은 공식 가이드
    (https://www.youthcenter.go.kr/cmnFooter/openapiIntro/oaiDoc)에서 확인됨
  - 응답 XML의 구체적 태그명은 검증되지 않음 → parser.py 에서 여러 후보 태그명을
    동시에 탐색하는 방어적 파싱을 쓴다
  - 각자 로컬 PC(일반 인터넷 환경)에서 아래 fetch_policies() 를 한 번 실행해
    실제 태그명을 확인하고, 안 맞으면 parser.py의 FIELD_CANDIDATES 만 고치면 된다

실행 예시:
    python -m backend.integrations.youthcenter_client  # 3건 받아서 원본 그대로 출력
"""

import os
import sys
from typing import List, Optional
from xml.etree import ElementTree

import requests

BASE_URL = "https://www.youthcenter.go.kr/opi/youthPlcyList.do"
DEFAULT_TIMEOUT = 15


class YouthCenterAPIError(RuntimeError):
    """네트워크 실패, HTTP 에러, 파싱 실패를 모두 이 예외로 통일해서
    호출부(seed 스크립트)가 '동작 안 하면 합성 데이터로 폴백'을 한 곳에서 처리하게 한다."""


def _get_api_key() -> str:
    key = os.getenv("YOUTHCENTER_API_KEY", "").strip()
    if not key:
        raise YouthCenterAPIError(
            "YOUTHCENTER_API_KEY 가 backend/.env 에 없습니다. "
            "온통청년 마이페이지 > 오픈(OPEN) API 에서 발급받은 키를 넣어주세요."
        )
    return key


def fetch_policies_raw(
    query: str = "",
    display: int = 100,
    page_index: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """API를 호출해 원본 XML 문자열을 그대로 반환. 실패하면 YouthCenterAPIError."""
    params = {
        "openApiVlak": _get_api_key(),
        "display": display,
        "pageIndex": page_index,
    }
    if query:
        params["query"] = query

    try:
        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise YouthCenterAPIError(f"온통청년 API 호출 실패: {exc}") from exc

    return resp.text


def _element_to_dict(elem: ElementTree.Element) -> dict:
    return {child.tag: (child.text or "").strip() for child in elem}


def _find_repeating_records(root: ElementTree.Element) -> List[ElementTree.Element]:
    """루트 태그명을 가정하지 않고, '같은 태그가 여러 번 반복되는 지점'을 찾아
    그걸 레코드(정책 1건)로 취급한다. 실제 태그명을 검증 못 한 상태라
    구조를 하드코딩하는 대신 이렇게 짜는 게 더 안전하다.
    """
    tag_counts: dict = {}
    parent_of: dict = {}
    for parent in root.iter():
        for child in parent:
            tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
            parent_of[child.tag] = parent

    repeating = [tag for tag, count in tag_counts.items() if count >= 2]
    if not repeating:
        return []

    # 가장 많이 반복되는 태그를 레코드로 채택 (정책 목록이 다른 어떤 반복 요소보다 많을 것)
    best_tag = max(repeating, key=lambda t: tag_counts[t])
    return list(parent_of[best_tag].findall(best_tag))


def fetch_policies(
    query: str = "",
    display: int = 100,
    page_index: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[dict]:
    """정책 목록을 [{태그명: 값}, ...] 형태의 원본(raw) dict 리스트로 반환.

    필드명을 우리 스키마에 매핑하는 건 youthcenter_mapper.py 가 담당한다.
    (에러 응답도 XML로 오는 API가 많아, 루트에 반복 요소가 없으면 원문을 그대로
    예외 메시지에 실어서 — 인증키 오류 등을 바로 알아볼 수 있게 한다)
    """
    xml_text = fetch_policies_raw(query, display, page_index, timeout)
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise YouthCenterAPIError(
            f"XML 파싱 실패 (JSON으로 응답했을 수 있음): {exc}\n원문 앞부분: {xml_text[:300]}"
        ) from exc

    records = _find_repeating_records(root)
    if not records:
        raise YouthCenterAPIError(
            f"응답에서 반복되는 정책 레코드를 찾지 못했습니다 (에러 응답 가능성). 원문 앞부분: {xml_text[:500]}"
        )
    return [_element_to_dict(r) for r in records]


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        rows = fetch_policies(display=3)
        print(f"{len(rows)}건 수신. 첫 번째 레코드의 필드명:")
        print(list(rows[0].keys()) if rows else "(없음)")
        print(rows[0] if rows else "")
    except YouthCenterAPIError as e:
        print(f"실패: {e}", file=sys.stderr)
        sys.exit(1)
