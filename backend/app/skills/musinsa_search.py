# 무신사 상품 검색 — 옷 이름으로 검색해서 첫 상품의 썸네일 URL을 반환
#
# 옛날 wardrobe_db.json에 있던 64벌은 image_url이 비어있어서 추천 결과에
# 썸네일을 못 띄움. 이걸 일괄로 채우기 위해 무신사 내부 search-api를 호출.
#
# 무신사는 공식 검색 API 문서를 공개하지 않아서 네트워크 탭을 보고 추론한
# 엔드포인트를 사용. 스펙이 바뀌면 _SEARCH_ENDPOINTS 리스트 위쪽부터 차례로
# 시도하다가 다 실패하면 None을 반환 → 호출부가 빈 결과로 graceful 처리.

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# 무신사 내부 검색 API 후보 — 위에서부터 시도.
# (1) 최근 SPA 버전: search-api.musinsa.com 도메인
# (2) 구버전: api.musinsa.com 도메인
# 둘 다 막히면 검색 페이지 HTML을 정규식으로 긁어서 첫 이미지 URL 추출.
_SEARCH_ENDPOINTS: list[dict] = [
    {
        "url": "https://search-api.musinsa.com/api2/v3/items",
        "params_builder": lambda kw: {"keyword": kw, "size": 5, "type": "GOODS"},
        "item_path": ("data", "items"),
        "image_field": "imageUrl",
    },
    {
        "url": "https://api.musinsa.com/api2/dpv/v1/musinsa/dp-goods",
        "params_builder": lambda kw: {"keyword": kw, "size": 5, "page": 1},
        "item_path": ("data", "list"),
        "image_field": "thumbnailImageUrl",
    },
]

# 무신사 검색 페이지 URL — HTML 파싱 폴백용
_SEARCH_PAGE_URL = "https://www.musinsa.com/search/musinsa/goods"

# 공통 헤더 — 봇 차단 회피용 User-Agent
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": "https://www.musinsa.com/",
}


def _walk(obj, path: tuple):
    """obj를 path대로 따라 들어가서 결과를 반환. 중간에 None이면 None."""
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if 0 <= key < len(current) else None
        else:
            return None
    return current


def _build_search_query(item_name: str) -> str:
    """검색어 정제 — 너무 긴 한국어 상품명은 무신사 검색이 잘 못 찾음.

    예: "큐티에잇 가먼츠 올드 트랙 셋업 블랙레드"
        → 브랜드 + 첫 키워드 정도만 남기는 게 매칭률 ↑
    """
    if not item_name:
        return ""

    # 양 끝 공백 제거 + 과한 공백 압축
    cleaned = re.sub(r"\s+", " ", item_name).strip()

    # 6개 단어 넘으면 앞 4개만 (브랜드 + 핵심 키워드)
    words = cleaned.split(" ")
    if len(words) > 6:
        cleaned = " ".join(words[:4])
    return cleaned


def _try_search_api(endpoint: dict, keyword: str) -> Optional[str]:
    """단일 search-api 엔드포인트 호출. 성공 시 첫 상품 이미지 URL 반환."""
    try:
        response = requests.get(
            endpoint["url"],
            params=endpoint["params_builder"](keyword),
            headers=_REQUEST_HEADERS,
            timeout=6,
        )
        # 무신사 API는 가끔 200 OK인데 빈 응답을 줌 — 본문 확인
        if response.status_code != 200:
            logger.debug("무신사 검색 %s → HTTP %s", endpoint["url"], response.status_code)
            return None
        data = response.json()

        items = _walk(data, endpoint["item_path"]) or []
        if not items:
            return None
        first = items[0]
        # 후보 이미지 필드들 — 무신사는 imageUrl, thumbnailImageUrl, image 등 변종이 많음
        for field in (
            endpoint["image_field"],
            "imageUrl",
            "thumbnailImageUrl",
            "image",
            "thumbnail",
        ):
            url = first.get(field) if isinstance(first, dict) else None
            if url and isinstance(url, str) and url.startswith("http"):
                return url
        return None

    except requests.Timeout:
        logger.debug("무신사 검색 API 타임아웃: %s", endpoint["url"])
        return None
    except requests.RequestException as e:
        logger.debug("무신사 검색 API 호출 실패 (%s): %s", endpoint["url"], e)
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.debug("무신사 검색 응답 파싱 실패: %s", e)
        return None


def _try_search_page_scrape(keyword: str) -> Optional[str]:
    """무신사 검색 페이지 HTML에서 첫 상품 이미지 URL 정규식 추출.

    JSON API가 모두 실패할 때 마지막 폴백. img src 또는 srcset에서
    image.msscdn.net 도메인의 첫 이미지를 가져옴.
    """
    try:
        response = requests.get(
            _SEARCH_PAGE_URL,
            params={"q": keyword},
            headers=_REQUEST_HEADERS,
            timeout=6,
        )
        if response.status_code != 200:
            return None
        html = response.text
        # 무신사 상품 이미지는 image.msscdn.net 도메인을 씀
        # 첫 매칭만 사용 — 검색 결과 첫 상품일 확률이 높음
        match = re.search(
            r'https?://image\.msscdn\.net/[^"\s\'>]+\.(?:jpg|jpeg|png|webp)',
            html,
            re.IGNORECASE,
        )
        if match:
            return match.group(0)
        return None
    except requests.RequestException as e:
        logger.debug("무신사 검색 페이지 스크랩 실패: %s", e)
        return None


def find_musinsa_product_image(item_name: str) -> Optional[dict]:
    """옷 이름으로 무신사를 검색해서 첫 상품의 이미지 URL을 찾아옵니다.

    여러 엔드포인트를 순서대로 시도 (가장 신뢰성 높은 것부터):
        1. search-api.musinsa.com /api2/v3/items
        2. api.musinsa.com /api2/dpv/v1/musinsa/dp-goods
        3. 검색 페이지 HTML 정규식 파싱 (최후 폴백)

    Returns:
        {
            "image_url": "https://image.msscdn.net/...",
            "source": "search-api" | "dp-goods" | "search-page",
            "query": "검색에 실제로 사용된 키워드",
        }
        모두 실패하면 None
    """
    keyword = _build_search_query(item_name)
    if not keyword:
        return None

    # JSON API들 차례로 시도
    for endpoint in _SEARCH_ENDPOINTS:
        image_url = _try_search_api(endpoint, keyword)
        if image_url:
            source = "search-api" if "search-api" in endpoint["url"] else "dp-goods"
            logger.info("무신사 매칭 성공 [%s] %s → %s", source, keyword, image_url[:60])
            return {"image_url": image_url, "source": source, "query": keyword}

    # 마지막 폴백: HTML 정규식
    image_url = _try_search_page_scrape(keyword)
    if image_url:
        logger.info("무신사 매칭 성공 [search-page] %s → %s", keyword, image_url[:60])
        return {"image_url": image_url, "source": "search-page", "query": keyword}

    logger.info("무신사 매칭 실패: %s", keyword)
    return None


def download_image_to_bytes(url: str, timeout: int = 8) -> Optional[bytes]:
    """이미지 URL을 다운로드해서 raw bytes로 반환. CLIP 임베딩 입력용.

    무신사 CDN(image.msscdn.net)은 Referer 검사가 있을 수 있어서
    공통 헤더를 같이 보냄.
    """
    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        # Content-Type이 이미지인지 확인
        ctype = response.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            logger.warning("응답이 이미지가 아님 (%s) → 무시: %s", ctype, url[:60])
            return None
        return response.content
    except requests.RequestException as e:
        logger.warning("이미지 다운로드 실패 (%s): %s", url[:60], e)
        return None
