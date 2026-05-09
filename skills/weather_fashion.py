# 날씨 데이터 + 무신사 스냅 + 내 옷장 임베딩을 결합해서 코디를 추천
# FAISS 인덱스를 데이터 규모에 따라 동적으로 선택
import json
import logging
import os
from typing import Optional

import faiss
import numpy as np
import requests
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# API 키는 환경 변수에서 읽어옴
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 카테고리 키워드 분류 테이블
_CATEGORY_KEYWORD_MAP: dict[str, list[str]] = {
    "아우터": ["아우터", "자켓", "재킷", "코트", "패딩", "가디건", "점퍼", "블레이저"],
    "상의": ["상의", "티셔츠", "니트", "셔츠", "맨투맨", "블라우스", "원피스", "후디"],
    "하의": ["하의", "팬츠", "바지", "치마", "스커트", "데님", "슬랙스", "반바지"],
    "신발": ["신발", "스니커즈", "구두", "운동화", "부츠", "샌들", "로퍼", "슬리퍼"],
}

# 무신사 시즌 코드 — API 스펙이 바뀌면 여기만 수정
_MUSINSA_SEASON_CODES = {
    "spring": "1",
    "summer": "2",
    "autumn": "3",
    "winter": "4",
}


def _resolve_outfit_category(raw_category: str) -> Optional[str]:
    """
    AI가 반환한 카테고리 문자열을 표준 표기로 정규화합니다.
    예) '맨투맨/후디' -> '상의', '스니커즈화' -> '신발'
    """
    if not raw_category:
        return None
    for standard_name, keywords in _CATEGORY_KEYWORD_MAP.items():
        if any(keyword in raw_category for keyword in keywords):
            return standard_name
    return None


def _get_season_code_from_temperature(temp_celsius: int) -> str:
    """기온(섭씨)을 무신사 시즌 코드로 변환합니다."""
    if temp_celsius >= 25:
        return _MUSINSA_SEASON_CODES["summer"]
    if temp_celsius >= 15:
        return _MUSINSA_SEASON_CODES["spring"]
    if temp_celsius >= 5:
        return _MUSINSA_SEASON_CODES["autumn"]
    return _MUSINSA_SEASON_CODES["winter"]


def _get_season_code_from_month(month: int) -> str:
    """날씨 API 실패 시 월(Month)을 기반으로 시즌 코드를 반환합니다 (Fallback)."""
    if month in [3, 4, 5]:
        return _MUSINSA_SEASON_CODES["spring"]
    if month in [6, 7, 8]:
        return _MUSINSA_SEASON_CODES["summer"]
    if month in [9, 10, 11]:
        return _MUSINSA_SEASON_CODES["autumn"]
    return _MUSINSA_SEASON_CODES["winter"]


def get_current_weather_wttr(location_name: str) -> Optional[dict]:
    """
    wttr.in 오픈 API로 실시간 날씨를 가져옵니다.

    응답 구조가 생각보다 중첩이 깊어서 .get()을 여러 번 쓰는 게 맞습니다.
    API가 다운되거나 지역명을 못 찾으면 None을 반환하고, 호출부에서 Fallback 처리합니다.
    """
    encoded_location = location_name.replace(" ", "+")
    url = f"https://wttr.in/{encoded_location}?format=j1"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        current = data.get("current_condition", [{}])[0]
        temp_celsius = current.get("temp_C")
        condition_english = current.get("weatherDesc", [{}])[0].get("value", "Clear")

        # 영어 날씨 설명을 한국어 요약으로 변환
        # "비/눈/흐림/맑음" 4단계 분류
        if any(kw in condition_english for kw in ["Rain", "rain", "Drizzle", "Shower"]):
            condition_korean = "비"
        elif any(kw in condition_english for kw in ["Snow", "snow", "Blizzard"]):
            condition_korean = "눈"
        elif any(kw in condition_english for kw in ["Cloud", "Overcast", "Mist", "Fog"]):
            condition_korean = "흐림"
        else:
            condition_korean = "맑음"

        return {"temp": temp_celsius, "condition": condition_korean}

    except requests.Timeout:
        logger.warning("wttr.in API 타임아웃 — Fallback 로직으로 전환합니다.")
        return None
    except requests.RequestException as e:
        logger.warning("wttr.in API 호출 실패: %s", e)
        return None
    except (KeyError, IndexError, ValueError) as e:
        # 응답 JSON 구조가 예상과 다를 때 — wttr.in이 포맷을 바꿨을 가능성
        logger.error("wttr.in 응답 파싱 실패 (API 스펙 변경?): %s", e)
        return None


def get_musinsa_snaps(user_gender: str = "여성", season_code: str = "4") -> list[str]:
    """
    무신사 스냅 API에서 해당 시즌 인기 코디 썸네일 URL 목록을 가져옵니다.

    무신사 API는 공식 문서가 없어서 내부 API를 리버스 엔지니어링한 거라
    언제든 스펙이 바뀔 수 있습니다. 에러 시 빈 리스트를 반환해서 앱이 멈추지 않도록 합니다.
    """
    gender_code = "WOMEN" if user_gender in ["여성", "여자", "WOMEN", "F"] else "MEN"
    api_url = (
        f"https://content.musinsa.com/api2/content/snap/ui/v2/modules/discovery/sections"
        f"?formatTypes=POST&genders={gender_code}&sort=POPULAR&seasonLabels={season_code}&size=10"
    )

    try:
        response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()

        # 응답 구조: data.list[0].contents[].thumbnailUrl
        # 이 경로가 깊어서 중간에 비어있으면 빈 리스트로 안전하게 처리
        sections = response.json().get("data", {}).get("list", [])
        if not sections:
            logger.warning("무신사 스냅 API 응답에 'list' 데이터가 비어 있습니다.")
            return []

        contents = sections[0].get("contents", [])
        snap_urls = [snap["thumbnailUrl"] for snap in contents if snap.get("thumbnailUrl")]
        logger.info("무신사 스냅 %d개 수집 완료 (시즌: %s, 성별: %s)", len(snap_urls), season_code, gender_code)
        return snap_urls

    except requests.Timeout:
        logger.warning("무신사 스냅 API 타임아웃")
        return []
    except requests.RequestException as e:
        logger.error("무신사 스냅 API 호출 실패: %s", e)
        return []
    except (KeyError, IndexError) as e:
        logger.error("무신사 스냅 응답 파싱 실패 (API 구조 변경?): %s", e)
        return []


def _download_image_bytes(url: str) -> Optional[bytes]:
    """URL에서 이미지를 다운로드해 바이트 데이터로 반환합니다."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        response.raise_for_status()
        # Content-Length가 없는 경우를 대비해 빈 바이트 체크
        if not response.content:
            logger.warning("이미지 다운로드는 됐지만 바이트가 비어 있습니다: %s", url)
            return None
        return response.content
    except requests.RequestException as e:
        logger.warning("이미지 다운로드 실패 (%s): %s", url, e)
        return None


def get_image_embedding_vector(image_bytes: bytes) -> Optional[list[float]]:
    """
    [트랙 1: 시각적 유사도] 의류 이미지를 고차원 벡터로 변환합니다.

    Gemini Embedding 모델을 사용합니다.
    이 벡터를 FAISS에 넣어서 "생김새가 비슷한 옷"을 찾는 게 핵심 아이디어입니다.
    """
    if not image_bytes:
        logger.warning("임베딩 요청에 빈 바이트가 들어왔습니다. 건너뜁니다.")
        return None
    try:
        response = _gemini_client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
        )
        return response.embeddings[0].values
    except Exception as e:
        
        logger.error("Gemini 임베딩 생성 실패: %s", e)
        return None


def analyze_outfit_image_vlm(image_bytes: bytes) -> dict:
    """
    [트랙 2: 의미적 분류] 의류 이미지에서 카테고리와 색상을 텍스트로 추출합니다.

    임베딩(숫자)만으로는 "이 옷이 상의인지 하의인지" 알 수 없어서
    VLM으로 카테고리를 별도로 판단합니다. 두 트랙을 합치는 게 hybrid_search_clothes의 핵심입니다.
    """
    if not image_bytes:
        return {"category": "", "color": ""}

    system_prompt = """
    사진 속 메인 의상의 카테고리(상의/하의/아우터/원피스 중 택 1)와 핵심 색상을 추출해.
    오직 JSON만 출력해. 예: {"category": "아우터", "color": "하늘색"}
    """
    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "이 의상을 분석해줘.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except json.JSONDecodeError:
        logger.warning("VLM 카테고리 분석 JSON 파싱 실패 — 빈 분류로 처리합니다.")
        return {"category": "", "color": ""}
    except Exception as e:
        logger.error("VLM 카테고리 분석 실패: %s", e)
        return {"category": "", "color": ""}


def _build_faiss_index(db_matrix: np.ndarray) -> tuple:
    """
    데이터 규모에 따라 FAISS 인덱스 종류를 동적으로 선택합니다.

    - 50개 미만: IndexFlatIP (정확도 100%, 브루트포스 KNN)
    - 50개 이상: IndexIVFFlat (근사치 검색 ANN, 속도 우선)

    옷장이 충분히 커지면 자동으로 빠른 엔진으로 전환됩니다.
    임계값 50은 IVFFlat의 최소 학습 데이터 요구사항을 고려한 값입니다.

    Returns:
        (index, engine_type_label)
    """
    num_data, dimension = db_matrix.shape

    if num_data < 50:
        index = faiss.IndexFlatIP(dimension)
        index.add(db_matrix)
        return index, "KNN (전수조사)"

    # nlist: 데이터를 나눌 클러스터 수
    nlist = int(np.sqrt(num_data))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(db_matrix)
    index.add(db_matrix)

    # nprobe: 검색 시 열어볼 클러스터 수. 높을수록 정확하지만 느려짐
    index.nprobe = min(5, nlist)
    return index, "ANN (근사치 검색)"


def recommend_fashion_for_weather(
    target_date, region: str, gender: str, wardrobe_data: list[dict]
) -> dict:
    """
    날씨 데이터와 무신사 스냅을 기반으로 내 옷장 코디를 추천합니다.

    전체 흐름:
    1. wttr.in으로 실시간 날씨 조회 (실패 시 날짜 기반 Fallback)
    2. 날씨에 맞는 무신사 인기 스냅 수집
    3. 스냅 이미지를 임베딩해서 FAISS로 옷장과 매칭
    4. 카테고리별 최적 아이템 조합 반환
    """
    if not wardrobe_data:
        return {"success": False, "comment": "옷장에 등록된 옷이 없습니다."}

    # 임베딩이 없는 아이템은 벡터 검색에 쓸 수 없으므로 필터링
    embeddable_items = [item for item in wardrobe_data if item.get("embedding")]
    if not embeddable_items:
        return {
            "success": False,
            "comment": "임베딩 데이터가 없습니다. 옷을 다시 등록(이미지 업로드)해 주세요.",
        }

    # --- 날씨 조회 + Fallback ---
    live_weather = get_current_weather_wttr(region)
    if live_weather:
        temp_str = live_weather["temp"]
        condition = live_weather["condition"]
        weather_summary = f"{region}은(는) 현재 {temp_str}°C, '{condition}' 날씨입니다."
        season_code = _get_season_code_from_temperature(int(temp_str))
    else:
        # wttr.in 실패 — 날짜의 월을 기준으로 시즌 추정
        month = target_date.month
        weather_summary = f"{region}의 {month}월 날씨 기준 추천입니다. (실시간 날씨 조회 실패)"
        season_code = _get_season_code_from_month(month)
        logger.warning("날씨 API 실패 — %d월 기반 Fallback 시즌 코드: %s", month, season_code)

    # --- 무신사 스냅 수집 ---
    snap_urls = get_musinsa_snaps(user_gender=gender, season_code=season_code)
    outfit_snaps = []
    for idx, url in enumerate(snap_urls):
        img_bytes = _download_image_bytes(url)
        if img_bytes:
            outfit_snaps.append({"snap_id": idx, "image_bytes": img_bytes, "url": url})

    if not outfit_snaps:
        return {
            "success": False,
            "comment": "무신사 스냅 이미지를 불러오지 못했습니다. 네트워크 상태를 확인해 주세요.",
        }

    # --- FAISS 인덱스 구성 ---
    embedding_dimension = len(embeddable_items[0]["embedding"])
    db_matrix = np.array([item["embedding"] for item in embeddable_items]).astype("float32")
    faiss.normalize_L2(db_matrix)

    faiss_index, engine_label = _build_faiss_index(db_matrix)

    # --- 스냅 순회: 내 옷장과 가장 잘 매칭되는 스냅 선택 ---
    best_snap = None
    best_category_matches: dict[str, Optional[dict]] = {
        "아우터": None, "상의": None, "하의": None, "신발": None
    }
    highest_match_score = -1.0

    num_items = len(embeddable_items)
    search_k = min(20, num_items)

    for snap in outfit_snaps:
        snap_vector = get_image_embedding_vector(snap["image_bytes"])
        if not snap_vector:
            continue  # 임베딩 실패한 스냅은 건너뛰기

        snap_matrix = np.array([snap_vector]).astype("float32")
        faiss.normalize_L2(snap_matrix)

        distances, indices = faiss_index.search(snap_matrix, search_k)
        top_match_score = float(distances[0][0]) * 100

        if top_match_score > highest_match_score:
            highest_match_score = top_match_score
            best_snap = snap

            # 카테고리별로 첫 번째로 매칭되는 아이템을 대표 추천으로 선택
            candidate_matches: dict[str, Optional[dict]] = {
                "아우터": None, "상의": None, "하의": None, "신발": None
            }
            found_categories: set[str] = set()

            for rank in range(search_k):
                item_idx = indices[0][rank]
                if item_idx == -1:
                    continue

                candidate_item = embeddable_items[item_idx]
                normalized_category = _resolve_outfit_category(
                    candidate_item.get("category", "")
                )

                if normalized_category and normalized_category not in found_categories:
                    candidate_matches[normalized_category] = candidate_item
                    found_categories.add(normalized_category)

            best_category_matches = candidate_matches

    if not best_snap:
        return {
            "success": False,
            "comment": "스냅 이미지 임베딩에 모두 실패했습니다. 잠시 후 다시 시도해 주세요.",
        }

    return {
        "success": True,
        "weather_context": weather_summary,
        "best_snap": best_snap,
        "recommendations": best_category_matches,
        "engine_used": engine_label,
    }


def hybrid_search_clothes(
    reference_image_bytes: bytes,
    wardrobe_data: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    이미지 임베딩(시각적 유사도) + VLM 카테고리 분석을 결합한 하이브리드 검색.

    두 AI 트랙을 쓰는 이유:
    - 임베딩만 쓰면 색감이나 실루엣이 비슷한데 카테고리가 다른 옷이 상위에 오는 경우가 있습니다.
    - VLM으로 카테고리를 별도 판단해서, 카테고리가 다르면 점수에 페널티를 줍니다.
    """
    if not wardrobe_data:
        return []

    if not reference_image_bytes:
        logger.warning("하이브리드 검색에 빈 이미지가 들어왔습니다.")
        return []

    # 두 AI 트랙을 동시에 호출 (추후 asyncio로 병렬화 가능)
    reference_vector = get_image_embedding_vector(reference_image_bytes)
    vlm_analysis = analyze_outfit_image_vlm(reference_image_bytes)

    if not reference_vector:
        logger.error("레퍼런스 이미지 임베딩 실패 — 검색 불가")
        return []

    embeddable_items = [item for item in wardrobe_data if item.get("embedding")]
    if not embeddable_items:
        return []

    # FAISS 검색
    db_matrix = np.array([item["embedding"] for item in embeddable_items]).astype("float32")
    faiss.normalize_L2(db_matrix)

    reference_matrix = np.array([reference_vector]).astype("float32")
    faiss.normalize_L2(reference_matrix)

    faiss_index, _ = _build_faiss_index(db_matrix)

    search_k = min(top_n * 2, len(embeddable_items))
    distances, indices = faiss_index.search(reference_matrix, search_k)

    target_category = vlm_analysis.get("category", "")
    scored_results: list[dict] = []

    for rank in range(search_k):
        item_idx = indices[0][rank]
        if item_idx == -1:
            continue

        candidate_item = embeddable_items[item_idx]
        visual_score = float(distances[0][rank]) * 100
        candidate_category = candidate_item.get("category", "")

        # 카테고리 불일치 페널티 (-30점)
        # 완전히 다른 카테고리(예: 신발 vs 상의)가 최상위에 오는 걸 방지
        if target_category and candidate_category:
            if (
                target_category not in candidate_category
                and candidate_category not in target_category
            ):
                visual_score -= 30

        scored_results.append({
            "item_name": candidate_item.get("item_name", "이름 없음"),
            "category": candidate_category,
            "color": candidate_item.get("color", "알 수 없음"),
            "fit_info": candidate_item.get("fit_info", "알 수 없음"),
            "final_score": round(visual_score, 1),
        })

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results[:top_n]
