# 날씨 데이터 + 무신사 스냅 + 내 옷장 임베딩을 결합해서 코디를 추천 (v2: OWM 통합)
# FAISS 인덱스를 데이터 규모에 따라 동적으로 선택
#
# backend/app/routers/wardrobe.py의 GET /recommend 엔드포인트가 Claude로 텍스트 추천을 하는데,
# 이 모듈의 recommend_fashion_text() 함수가 그 패턴을 그대로 재현합니다 — 임베딩 검색이
# 불가능한 상황(임베딩 미보유, 무신사 API 실패)에서 폴백으로 가동됩니다.
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

# API 키는 환경 변수에서 읽어옴 — 없으면 lazy하게 처리해서 import 단계에 안 죽게
_gemini_client = None
if os.environ.get("GEMINI_API_KEY"):
    try:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        logger.warning("Gemini 클라이언트 초기화 실패 — 임베딩/VLM 기능 비활성화: %s", e)
else:
    logger.warning("GEMINI_API_KEY 환경변수가 비어 있습니다 — 이미지 임베딩/VLM 기능 비활성화.")

# Claude SDK는 선택적 의존성 — 없으면 OpenAI로 폴백
try:
    from anthropic import Anthropic  # type: ignore
    _anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:
    _anthropic_client = None
    logger.info("anthropic SDK 미설치 — 텍스트 추천은 OpenAI로 폴백됩니다.")

try:
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if os.environ.get("OPENAI_API_KEY") else None
except ImportError:
    _openai_client = None

# CLIP 보조 임베딩 — 선택적 의존성 (PDF2 캡스톤 실험에서 검증된 백본).
# 무거운 의존성이라 graceful fallback. 없으면 메인 트랙(Gemini)만 동작.
_clip_model = None
_clip_preprocess = None
_clip_device = None
try:
    import torch  # type: ignore
    import clip  # type: ignore  # openai/CLIP
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
    _clip_model.eval()
    logger.info("CLIP ViT-B/32 보조 트랙 활성화 (device=%s)", _clip_device)
except Exception as e:
    logger.info("CLIP 비활성화 (의존성 미설치 또는 로딩 실패) — Gemini 단독 모드: %s", e)


# 유사도 의미 임계값 — PDF2 (캡스톤 CLIP 실험) 기준
# 0.9+ : 거의 동일 카테고리
# 0.7~0.9 : 관련 있음 (추천 가능)
# 0.5 이하 : 무관
_SIMILARITY_LABELS = (
    (0.9, "거의 동일"),
    (0.7, "추천 가능"),
    (0.5, "약한 관련"),
)


def _classify_similarity_label(score_percent: float) -> str:
    """
    코사인 유사도(0~100% 스케일)를 사람이 읽기 좋은 의미 라벨로 변환.

    PDF2의 임계값 가이드(0.9/0.7/0.5)를 직접 채택. UI가 단순 % 대신
    "거의 동일/추천 가능/약한 관련/무관" 같은 의미를 함께 보여줄 수 있게 합니다.
    """
    score = score_percent / 100.0
    for threshold, label in _SIMILARITY_LABELS:
        if score >= threshold:
            return label
    return "무관"

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


# 옷차림 단계 — 팀원 weather_service.py에서 가져온 6단계 분류.
# UI에 "선선함" "쌀쌀함" 같은 한국어 라벨로 직접 띄우기 좋습니다.
# 임계값은 한국 기후 기준으로 튜닝된 값.
_OUTFIT_SEASON_THRESHOLDS = (
    (28, "매우 더움"),
    (23, "더움"),
    (17, "따뜻함"),
    (12, "선선함"),
    (5, "쌀쌀함"),
)


def get_outfit_season(temp_celsius: float) -> str:
    """기온(℃)을 6단계 옷차림 라벨로 변환합니다.

    매우 더움(28+)/더움(23+)/따뜻함(17+)/선선함(12+)/쌀쌀함(5+)/매우 추움(<5)
    """
    try:
        t = int(round(float(temp_celsius)))
    except (TypeError, ValueError):
        return "선선함"
    for threshold, label in _OUTFIT_SEASON_THRESHOLDS:
        if t >= threshold:
            return label
    return "매우 추움"


# 옷차림 단계별 태그 키워드 — 팀원 wardrobe.py에서 가져옴.
# 두 용도로 쓰임:
#   (1) LLM 프롬프트에 "이 단계엔 이런 태그가 적합" 힌트로 추가
#   (2) AI 백본이 모두 실패했을 때 마지막 폴백으로 태그 매칭 추천
_OUTFIT_SEASON_KEYWORDS: dict[str, list[str]] = {
    "매우 더움": ["여름", "반팔", "반바지", "얇은", "린넨", "쇼츠"],
    "더움":     ["여름", "반팔", "얇은", "린넨"],
    "따뜻함":   ["봄", "가을", "긴팔", "가디건", "셔츠"],
    "선선함":   ["봄", "가을", "자켓", "가디건", "후디", "맨투맨"],
    "쌀쌀함":   ["겨울", "코트", "니트", "두꺼운", "기모"],
    "매우 추움": ["겨울", "패딩", "두꺼운", "방한", "기모", "양털"],
}


def recommend_by_tags(
    wardrobe_data: list[dict],
    outfit_season: str,
    limit_per_category: int = 3,
) -> dict[str, list[dict]]:
    """[최후 폴백] 옷차림 단계 키워드로 옷장에서 태그 매칭만 해서 추천.

    임베딩/LLM 둘 다 실패한 비상 상황에 호출됩니다. 카테고리별로 최대
    limit_per_category개씩 반환합니다.
    """
    keywords = _OUTFIT_SEASON_KEYWORDS.get(outfit_season, [])
    if not keywords:
        return {}

    grouped: dict[str, list[dict]] = {"상의": [], "하의": [], "아우터": [], "신발": [], "기타": []}
    for item in wardrobe_data:
        tags = str(item.get("tags", ""))
        if not any(kw in tags for kw in keywords):
            continue
        category = _resolve_outfit_category(item.get("category", "")) or "기타"
        bucket = grouped.setdefault(category, [])
        if len(bucket) < limit_per_category:
            bucket.append(item)
    # 빈 카테고리 제거
    return {k: v for k, v in grouped.items() if v}


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


def _safe_int(value, default=None):
    """문자열/숫자/None을 안전하게 int로. 실패하면 default."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _english_condition_to_korean(condition_english: str) -> str:
    """영어 날씨 설명을 한국어 4단계 분류(비/눈/흐림/맑음)로."""
    if any(kw in condition_english for kw in ["Rain", "rain", "Drizzle", "Shower"]):
        return "비"
    if any(kw in condition_english for kw in ["Snow", "snow", "Blizzard"]):
        return "눈"
    if any(kw in condition_english for kw in ["Cloud", "Overcast", "Mist", "Fog"]):
        return "흐림"
    return "맑음"


def get_current_weather_wttr(location_name: str) -> Optional[dict]:
    """wttr.in 오픈 API로 실시간 날씨를 가져옵니다 (API 키 불필요).

    팀원 weather_service.py 머지: 그동안 temp/condition만 뽑던 걸
    feels_like / humidity / wind_speed / temp_min / temp_max 까지 확장.
    wttr.in의 weather[0] 블록에 일일 최저/최고가 들어있어서 그것도 활용.

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
        temp_celsius = _safe_int(current.get("temp_C"))
        feels_like = _safe_int(current.get("FeelsLikeC"), default=temp_celsius)
        humidity = _safe_int(current.get("humidity"))
        # wttr는 풍속이 kmph 단위 — OpenWeatherMap은 m/s 단위.
        # 통일성을 위해 m/s로 변환 (kmph * 0.2778 = m/s)
        wind_kmph = _safe_float(current.get("windspeedKmph"))
        wind_speed = round(wind_kmph * 0.2778, 1) if wind_kmph is not None else None

        condition_english = current.get("weatherDesc", [{}])[0].get("value", "Clear")
        condition_korean = _english_condition_to_korean(condition_english)

        # 오늘 일일 최저/최고 — weather[0]에 있음
        today_block = data.get("weather", [{}])[0]
        temp_min = _safe_int(today_block.get("mintempC"))
        temp_max = _safe_int(today_block.get("maxtempC"))

        return {
            "source": "wttr.in",
            "city": location_name,
            "temp": temp_celsius,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "condition": condition_korean,
            "description": condition_english,
            "outfit_season": get_outfit_season(temp_celsius) if temp_celsius is not None else None,
        }

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


# 한국 주요 도시 → OpenWeatherMap 영어 표기 매핑.
# "서울" 그대로 넣어도 OWM이 받긴 하는데 매번 검색해야 해서 미리 영어로 보내는 게 빠름.
_OWM_CITY_ALIASES = {
    "서울": "Seoul,KR",
    "부산": "Busan,KR",
    "대구": "Daegu,KR",
    "인천": "Incheon,KR",
    "광주": "Gwangju,KR",
    "대전": "Daejeon,KR",
    "울산": "Ulsan,KR",
    "수원": "Suwon,KR",
    "제주": "Jeju,KR",
    "춘천": "Chuncheon,KR",
}


def get_current_weather_owm(location_name: str) -> Optional[dict]:
    """OpenWeatherMap API로 실시간 날씨를 가져옵니다 (WEATHER_API_KEY 필요).

    팀원 weather_service.py의 OWM 호출을 sync(requests)로 옮긴 버전.
    WEATHER_API_KEY 환경변수가 없으면 None을 반환해서 호출부가 wttr.in으로 폴백.
    """
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return None

    city_query = _OWM_CITY_ALIASES.get(location_name, location_name)
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_query,
        "appid": api_key,
        "units": "metric",   # 섭씨
        "lang": "kr",        # 한국어 설명
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = _safe_int(data["main"]["temp"])
        feels_like = _safe_int(data["main"].get("feels_like"), default=temp)
        condition_english = data["weather"][0].get("main", "Clear")

        return {
            "source": "openweathermap",
            "city": data.get("name", location_name),
            "temp": temp,
            "feels_like": feels_like,
            "humidity": _safe_int(data["main"].get("humidity")),
            "wind_speed": _safe_float(data.get("wind", {}).get("speed")),
            "temp_min": _safe_int(data["main"].get("temp_min")),
            "temp_max": _safe_int(data["main"].get("temp_max")),
            "condition": _english_condition_to_korean(condition_english),
            "description": data["weather"][0].get("description", condition_english),
            "icon": data["weather"][0].get("icon"),  # OWM 전용 — 아이콘 코드
            "outfit_season": get_outfit_season(temp) if temp is not None else None,
        }

    except requests.Timeout:
        logger.warning("OpenWeatherMap API 타임아웃")
        return None
    except requests.RequestException as e:
        logger.warning("OpenWeatherMap API 호출 실패: %s", e)
        return None
    except (KeyError, IndexError, ValueError) as e:
        logger.error("OpenWeatherMap 응답 파싱 실패: %s", e)
        return None


def get_current_weather(location_name: str) -> Optional[dict]:
    """현재 날씨 통합 진입점.

    우선순위:
      1. WEATHER_API_KEY가 있으면 OpenWeatherMap (정확도/신뢰도 우위)
      2. 실패하면 wttr.in (무료, 키 불필요)

    어느 쪽이든 같은 스키마(temp, feels_like, humidity, wind_speed, condition,
    outfit_season 등)를 반환하므로 호출부는 source만 보고 출처를 구분하면 됩니다.
    """
    if os.environ.get("WEATHER_API_KEY"):
        result = get_current_weather_owm(location_name)
        if result:
            return result
        logger.info("OWM 실패 — wttr.in으로 폴백")
    return get_current_weather_wttr(location_name)


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

    백본 우선순위:
      1. CLIP ViT-B/32 (메인) — 로컬 모델, API 비용 0, 무료 티어 제약 없음
      2. Gemini multimodal embedding (폴백) — 일부 환경에서만 사용 가능

    Gemini가 막혀있어도 CLIP만 설치돼 있으면 정상 동작합니다.
    CLIP과 Gemini는 차원이 달라서 같은 옷장 안에서는 한 백본만 사용해야 합니다.

    NOTE: 의류 영역 추출(배경 제거 + 멀티 분리)은
    skills.preprocess.prepare_images_for_embedding을 호출부에서 먼저 적용하세요.
    """
    if not image_bytes:
        logger.warning("임베딩 요청에 빈 바이트가 들어왔습니다. 건너뜁니다.")
        return None

    # 1순위: CLIP (메인)
    clip_vec = get_clip_embedding_vector(image_bytes)
    if clip_vec is not None:
        return clip_vec

    # 2순위: Gemini (CLIP 미설치 시에만)
    if _gemini_client is None:
        logger.warning("CLIP과 Gemini 둘 다 사용 불가 — 임베딩 건너뜁니다.")
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


def get_clip_text_embedding(text: str) -> Optional[list[float]]:
    """
    CLIP 텍스트 인코더로 텍스트 → 벡터 변환.

    CLIP의 강점 중 하나 — 이미지와 텍스트가 같은 임베딩 공간에 매핑됩니다.
    그래서 "이미지가 없는 옷장 아이템(텍스트 메타정보만 있음)"도
    같은 공간에서 무신사 스냅 이미지와 매칭 가능해집니다.

    사용 예:
      "상의 검정색 오버핏 후드" → 벡터
      이 벡터를 무신사 스냅 이미지 벡터와 같은 FAISS 인덱스에 넣을 수 있음.
    """
    if _clip_model is None:
        return None
    if not text or not text.strip():
        return None

    try:
        import torch as _torch  # type: ignore
        import clip as _clip  # type: ignore

        # CLIP의 토크나이저는 입력 길이 77 token 제한 — 너무 길면 자름
        tokens = _clip.tokenize([text[:200]], truncate=True).to(_clip_device)
        with _torch.no_grad():
            features = _clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.warning("CLIP 텍스트 임베딩 실패: %s", e)
        return None


def build_item_text_for_embedding(item: dict) -> str:
    """
    옷장 아이템의 텍스트 메타를 CLIP 텍스트 인코더용 자연어로 합칩니다.

    "검정색 오버핏 후드 상의" 같은 짧은 영문/한글 문장을 만들면 CLIP이 잘 받아들입니다.
    """
    parts: list[str] = []
    if item.get("category"):
        parts.append(str(item["category"]))
    if item.get("color"):
        parts.append(str(item["color"]))
    if item.get("fit_info"):
        parts.append(str(item["fit_info"]))
    if item.get("tags"):
        # 쉼표 구분 tags의 첫 1~2개만
        tag_words = [t.strip() for t in str(item["tags"]).split(",") if t.strip()][:2]
        parts.extend(tag_words)
    if item.get("name"):
        parts.append(str(item["name"]))
    return " ".join(parts).strip()


def get_clip_embedding_vector(image_bytes: bytes) -> Optional[list[float]]:
    """
    [보조 트랙: CLIP] PDF2의 캡스톤 실험에서 사용한 OpenAI CLIP ViT-B/32로
    이미지를 임베딩합니다. Gemini 임베딩과 합의도가 높을수록 매칭 신뢰도가 높습니다.

    CLIP 의존성이 없으면 None을 반환해서 호출부가 graceful하게 처리.
    """
    if _clip_model is None or _clip_preprocess is None:
        return None
    if not image_bytes:
        return None

    try:
        from PIL import Image  # type: ignore
        import io as _io
        import torch as _torch  # local alias

        image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        tensor = _clip_preprocess(image).unsqueeze(0).to(_clip_device)
        with _torch.no_grad():
            features = _clip_model.encode_image(tensor)
            # L2 normalize for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.warning("CLIP 임베딩 실패: %s", e)
        return None


def get_dual_embedding(image_bytes: bytes) -> dict:
    """
    Gemini(메인) + CLIP(보조)을 동시에 시도해서 두 벡터를 한 번에 얻습니다.

    Returns:
        {
            "embedding": [...],          # Gemini, 메인 (FAISS 검색용)
            "clip_embedding": [...] | None,  # CLIP, 앙상블용
        }
    """
    return {
        "embedding": get_image_embedding_vector(image_bytes),
        "clip_embedding": get_clip_embedding_vector(image_bytes),
    }


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """순수 numpy 코사인 유사도 (CLIP 점수 계산용)."""
    a = np.array(vec_a, dtype="float32")
    b = np.array(vec_b, dtype="float32")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _filter_embeddings_by_majority_dim(items: list[dict]) -> tuple[list[dict], int, int]:
    """
    옷장에 차원이 다른 임베딩이 섞여있을 때 (예: Gemini 3072 + CLIP 512),
    가장 많이 등장하는 차원만 선별해 통일성을 확보합니다.

    PDF1·PDF2 인사이트와 별개로, 백본 전환(Gemini → CLIP) 과정에서 발생할 수 있는
    실무 이슈를 처리합니다.

    Returns:
        (필터링된 아이템 리스트, 채택 차원, 제외된 개수)
    """
    embeddable = [it for it in items if it.get("embedding")]
    if not embeddable:
        return [], 0, 0

    # 각 임베딩의 차원 카운트
    dim_counter: dict[int, int] = {}
    for item in embeddable:
        emb = item["embedding"]
        if isinstance(emb, list):
            dim = len(emb)
            dim_counter[dim] = dim_counter.get(dim, 0) + 1

    if not dim_counter:
        return [], 0, 0

    # 다수파 차원 선택
    majority_dim = max(dim_counter, key=lambda d: dim_counter[d])
    filtered = [it for it in embeddable if isinstance(it.get("embedding"), list) and len(it["embedding"]) == majority_dim]
    excluded = len(embeddable) - len(filtered)

    if excluded > 0:
        logger.warning(
            "임베딩 차원 불일치 — %d차원 %d개를 채택, 다른 차원 %d개 제외 (재임베딩 권장)",
            majority_dim, len(filtered), excluded,
        )
    return filtered, majority_dim, excluded


def analyze_outfit_image_vlm(image_bytes: bytes) -> dict:
    """
    [트랙 2: 의미적 분류] 의류 이미지에서 카테고리와 색상을 텍스트로 추출합니다.

    임베딩(숫자)만으로는 "이 옷이 상의인지 하의인지" 알 수 없어서
    VLM으로 카테고리를 별도로 판단합니다. 두 트랙을 합치는 게 hybrid_search_clothes의 핵심입니다.
    """
    if not image_bytes:
        return {"category": "", "color": ""}
    if _gemini_client is None:
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


def _format_wardrobe_for_llm(items: list[dict]) -> str:
    """
    backend/app/routers/wardrobe.py의 get_recommendation()과 동일한 포맷으로
    옷장을 텍스트화합니다. 두 코드베이스가 같은 프롬프트 컨벤션을 공유하도록 정렬.
    """
    return "\n".join(
        f"- {item.get('name', '이름 없음')} "
        f"({item.get('category', '?')}, {item.get('color', '?')}, "
        f"태그: {item.get('tags', '')})"
        for item in items
    )


def build_daily_outfit_notification(
    region: str,
    wardrobe_data: list[dict],
    gender: str = "여성",
    max_length: int = 120,
) -> dict:
    """
    매일 아침 푸시 알림용 짧은 코디 추천 메시지를 생성합니다.

    Streamlit UI와 Cloud Functions(매일 오전 8시 트리거) 양쪽에서 재사용 가능하도록
    독립적인 함수로 분리. 외부 의존: wttr.in (날씨), Gemini/OpenAI (텍스트 폴백).

    Returns:
        {
            "title": "오늘의 코디",
            "body": "서울 22°C 맑음 ☀️ 베이지 오버핏 + 그레이 와이드 팬츠 어떠세요?",
            "weather": {"temp": 22, "condition": "맑음"},
            "items": [{"category": "상의", "name": "..."}, ...]
        }
        실패 시 {"title": ..., "body": "...폴백 메시지...", "items": []}
    """
    from datetime import date

    # 1) 날씨 조회 — 통합 진입점 사용 (OWM → wttr.in 폴백)
    live_weather = get_current_weather(region)
    if live_weather:
        temp = live_weather.get("temp") or 20
        feels_like = live_weather.get("feels_like") or temp
        condition = live_weather.get("condition", "맑음")
        outfit_season = live_weather.get("outfit_season") or get_outfit_season(temp)
    else:
        temp = 20  # 폴백 기본값
        feels_like = 20
        condition = "맑음"
        outfit_season = "선선함"

    # 2) 옷장 추천 (recommend_fashion_for_weather와 같은 로직 활용)
    items_summary: list[dict] = []
    recommendation = recommend_fashion_for_weather(
        target_date=date.today(),
        region=region,
        gender=gender,
        wardrobe_data=wardrobe_data,
    )

    if recommendation.get("success") and recommendation.get("recommendations"):
        # 카테고리별 추천 아이템 — 푸시 알림이라 핵심 2개만 (상의 + 하의)
        for cat in ["상의", "하의", "아우터"]:
            item = recommendation["recommendations"].get(cat)
            if item:
                items_summary.append({
                    "category": cat,
                    "name": item.get("name") or item.get("item_name", "이름 없음"),
                    "color": item.get("color", ""),
                    "fit_info": item.get("fit_info", ""),
                })
            if len(items_summary) >= 2:
                break

    # 3) 알림 본문 조립
    weather_emoji = {
        "맑음": "☀️", "흐림": "☁️", "비": "🌧️", "눈": "❄️",
    }.get(condition, "👕")

    # 체감온도가 실제 온도와 다르면 함께 표기 ("22°C(체감 19°C)")
    if feels_like is not None and abs(feels_like - temp) >= 2:
        temp_phrase = f"{temp}°C(체감 {feels_like}°C)"
    else:
        temp_phrase = f"{temp}°C"

    weather_phrase = f"{region} {temp_phrase} {condition} {weather_emoji} [{outfit_season}]"

    if items_summary:
        # "베이지 오버핏 셔츠 + 회색 와이드 팬츠 어떠세요?" 식
        item_phrases: list[str] = []
        for it in items_summary:
            parts = [it.get("color", ""), it.get("fit_info", ""), it["name"]]
            phrase = " ".join(p for p in parts if p).strip()
            item_phrases.append(phrase)
        outfit_text = " + ".join(item_phrases)
        body = f"{weather_phrase} {outfit_text} 어떠세요?"
    elif recommendation.get("fallback") and recommendation.get("text_recommendation"):
        # 텍스트 LLM 폴백 결과를 짧게 자르기
        body = recommendation["text_recommendation"].split("\n")[0]
        body = f"{weather_phrase} {body}"
    else:
        body = f"{weather_phrase} 오늘도 멋진 하루 보내세요!"

    # 푸시 알림은 짧아야 — max_length 컷
    if len(body) > max_length:
        body = body[:max_length - 1].rstrip() + "…"

    return {
        "title": "오늘의 코디 ✨",
        "body": body,
        "weather": {
            "temp": temp,
            "feels_like": feels_like,
            "condition": condition,
            "outfit_season": outfit_season,
        },
        "items": items_summary,
    }


def recommend_fashion_text(
    weather: str, temperature: int, wardrobe_data: list[dict]
) -> dict:
    """
    [폴백 트랙: 텍스트 LLM]
    backend/app/routers/wardrobe.py의 GET /recommend 엔드포인트와 동일한 의미를
    fashion-backend 쪽에서 재현합니다. 이미지 임베딩이 없거나 무신사 스냅 API가
    실패한 상황에서, 옷장을 텍스트로 요약해 LLM에게 자연어 코디를 받아옵니다.

    LLM 우선순위:
        1. ANTHROPIC_API_KEY가 있으면 Claude (backend와 동일한 백본)
        2. 없으면 OpenAI gpt-4o-mini
        3. 둘 다 없으면 명시적 실패 메시지

    Returns:
        {"success": True, "recommendation": "...", "engine_used": "Claude|OpenAI"}
        또는 {"success": False, "comment": "..."}
    """
    if not wardrobe_data:
        return {"success": False, "comment": "등록된 옷이 없습니다. 먼저 옷장에 옷을 추가해 주세요."}

    wardrobe_text = _format_wardrobe_for_llm(wardrobe_data)
    user_message = (
        f"오늘 날씨는 '{weather}'이고 기온은 {temperature}°C입니다.\n"
        f"내 옷장 목록:\n{wardrobe_text}\n\n"
        "위 옷들 중에서 오늘 날씨에 어울리는 코디를 추천해 주세요. "
        "추천 사유를 한국어로 친근하게 1~2문단으로 적어주세요."
    )

    # 1순위: Claude (backend와 동일한 LLM 백본)
    if _anthropic_client:
        try:
            response = _anthropic_client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=600,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text if response.content else ""
            return {"success": True, "recommendation": text, "engine_used": "Claude (텍스트 폴백)"}
        except Exception as e:
            logger.warning("Claude 텍스트 추천 실패 — OpenAI로 폴백: %s", e)

    # 2순위: OpenAI
    if _openai_client:
        try:
            response = _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_message}],
                temperature=0.5,
                timeout=15,
            )
            text = response.choices[0].message.content or ""
            return {"success": True, "recommendation": text, "engine_used": "OpenAI gpt-4o-mini (텍스트 폴백)"}
        except Exception as e:
            logger.error("OpenAI 텍스트 추천 실패: %s", e)
            return {"success": False, "comment": f"텍스트 추천 LLM 호출 실패: {e}"}

    return {
        "success": False,
        "comment": "텍스트 추천에 사용할 LLM 키(ANTHROPIC_API_KEY/OPENAI_API_KEY)가 설정되지 않았습니다.",
    }


def recommend_fashion_for_weather(
    target_date, region: str, gender: str, wardrobe_data: list[dict]
) -> dict:
    """
    날씨 데이터와 무신사 스냅을 기반으로 내 옷장 코디를 추천합니다.

    전체 흐름:
    1. wttr.in으로 실시간 날씨 조회 (실패 시 날짜 기반 Fallback)
    2. 날씨에 맞는 무신사 인기 스냅 수집
    3. 스냅 이미지를 임베딩해서 FAISS로 옷장과 매칭
    4. 실패 경로 (임베딩 없음 / 스냅 API 실패) → recommend_fashion_text()로 폴백
    5. 카테고리별 최적 아이템 조합 반환
    """
    if not wardrobe_data:
        return {"success": False, "comment": "옷장에 등록된 옷이 없습니다."}

    # 날씨 먼저 조회 — 폴백 텍스트 추천에서도 필요 (OWM → wttr.in 폴백)
    live_weather_for_fallback = get_current_weather(region)

    # 임베딩 필터링 — 다수파 차원만 통과 (Gemini/CLIP 혼재 시 강건성)
    embeddable_items, majority_dim, excluded = _filter_embeddings_by_majority_dim(wardrobe_data)
    if not embeddable_items:
        # 폴백: 텍스트 LLM 추천 (backend wardrobe.py 패턴)
        logger.info("임베딩 0개 — 텍스트 LLM 폴백 가동")
        weather_label = live_weather_for_fallback["condition"] if live_weather_for_fallback else "맑음"
        temperature = int(live_weather_for_fallback["temp"]) if live_weather_for_fallback else 20
        text_result = recommend_fashion_text(weather_label, temperature, wardrobe_data)
        if text_result.get("success"):
            return {
                "success": True,
                "fallback": True,
                "weather_context": f"{region} {weather_label} {temperature}°C (텍스트 폴백 모드)",
                "weather_data": live_weather_for_fallback,
                "text_recommendation": text_result["recommendation"],
                "engine_used": text_result["engine_used"],
            }
        return {
            "success": False,
            "comment": "임베딩이 없고 텍스트 폴백도 실패: " + text_result.get("comment", "원인 불명"),
            "weather_data": live_weather_for_fallback,
        }

    # --- 날씨 조회 + Fallback ---
    live_weather = live_weather_for_fallback  # 위에서 이미 한 번 조회함
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
        # 무신사 API 실패 — 텍스트 LLM 폴백 (backend wardrobe.py 패턴)
        logger.info("무신사 스냅 0건 — 텍스트 LLM 폴백 가동")
        weather_label = live_weather["condition"] if live_weather else "맑음"
        temperature = int(live_weather["temp"]) if live_weather else 20
        text_result = recommend_fashion_text(weather_label, temperature, wardrobe_data)
        if text_result.get("success"):
            return {
                "success": True,
                "fallback": True,
                "weather_context": weather_summary + " (스냅 API 실패 → 텍스트 폴백)",
                "weather_data": live_weather,
                "text_recommendation": text_result["recommendation"],
                "engine_used": text_result["engine_used"],
            }
        return {
            "success": False,
            "comment": "무신사 스냅 실패 + 텍스트 폴백도 실패: " + text_result.get("comment", "원인 불명"),
            "weather_data": live_weather,
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
        "weather_data": live_weather,
        "best_snap": best_snap,
        "recommendations": best_category_matches,
        "engine_used": engine_label,
    }


def hybrid_search_clothes(
    reference_image_bytes: bytes,
    wardrobe_data: list[dict],
    top_n: int = 3,
    use_preprocess: bool = True,
    ensemble_clip: bool = True,
) -> list[dict]:
    """
    이미지 임베딩(시각적 유사도) + VLM 카테고리 분석 + (옵션) CLIP 앙상블을 결합한
    하이브리드 손민수 검색.

    파이프라인 (PDF1·PDF2 인사이트 반영):
        1. (선택) skills.preprocess로 의류 영역만 crop → 배경 노이즈 제거
        2. Gemini 메인 임베딩 + CLIP 보조 임베딩
        3. FAISS로 옷장과 매칭 → 카테고리 페널티 적용
        4. CLIP 점수와 가중 평균 (Gemini 70%, CLIP 30%)
        5. PDF2 임계값 기준 의미 라벨 부여 (거의 동일/추천 가능/약한 관련/무관)

    Args:
        use_preprocess: True면 의류 영역만 추출 후 임베딩 (PDF2 배경 노이즈 해결)
        ensemble_clip: True면 CLIP 점수와 앙상블 (CLIP 미설치면 자동 무시)
    """
    if not wardrobe_data:
        return []

    if not reference_image_bytes:
        logger.warning("하이브리드 검색에 빈 이미지가 들어왔습니다.")
        return []

    # --- Step 1: 전처리 (의류 영역 추출, 배경 노이즈 제거) ---
    embedding_input = reference_image_bytes
    preprocess_meta: dict = {"preprocessed": False}
    if use_preprocess:
        # 손민수는 "닮은 옷 1개"를 찾는 거라서 단일 의류 모드 (가장 큰 박스)
        from .preprocess import prepare_images_for_embedding
        prepared = prepare_images_for_embedding(reference_image_bytes, multi_item_mode=False)
        if prepared:
            embedding_input = prepared[0]["image_bytes"]
            preprocess_meta = prepared[0]["meta"]
            logger.info("손민수 전처리: %s", preprocess_meta.get("source"))

    # --- Step 2: 메인(Gemini) + 보조(CLIP) 임베딩 동시 시도 ---
    reference_vector = get_image_embedding_vector(embedding_input)
    reference_clip = get_clip_embedding_vector(embedding_input) if ensemble_clip else None
    vlm_analysis = analyze_outfit_image_vlm(embedding_input)

    if not reference_vector:
        logger.error("레퍼런스 이미지 임베딩 실패 — 검색 불가")
        return []

    # 차원 다수파 필터링 — Gemini/CLIP 혼재 시에도 강건하게
    embeddable_items, majority_dim, _excluded = _filter_embeddings_by_majority_dim(wardrobe_data)
    if not embeddable_items:
        return []

    # 레퍼런스 벡터 차원이 옷장 다수파와 다르면 검색 불가 — 호출부에 알림
    if len(reference_vector) != majority_dim:
        logger.warning(
            "손민수: 레퍼런스 임베딩 차원(%d)이 옷장 다수파 차원(%d)과 달라요. "
            "옷장 관리 탭에서 'CLIP 일괄 재임베딩'을 돌려야 매칭됩니다.",
            len(reference_vector), majority_dim,
        )
        return []

    # --- Step 3: FAISS 검색 (메인 트랙) ---
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
        gemini_score = float(distances[0][rank]) * 100  # 0~100%
        candidate_category = candidate_item.get("category", "")

        # 카테고리 불일치 페널티 (-30점)
        # 완전히 다른 카테고리(예: 신발 vs 상의)가 최상위에 오는 걸 방지
        category_penalty = 0
        if target_category and candidate_category:
            if (
                target_category not in candidate_category
                and candidate_category not in target_category
            ):
                category_penalty = -30

        # --- Step 4: CLIP 앙상블 ---
        # 옷장 아이템에 clip_embedding이 저장돼 있고 레퍼런스 CLIP 벡터가 있으면 가중 평균
        clip_score: Optional[float] = None
        candidate_clip = candidate_item.get("clip_embedding")
        if reference_clip and candidate_clip:
            clip_score = _cosine_similarity(reference_clip, candidate_clip) * 100

        if clip_score is not None:
            # 두 모델 가중 평균 (Gemini 0.7, CLIP 0.3 — Gemini가 우리 메인 백본이므로)
            blended_score = gemini_score * 0.7 + clip_score * 0.3
        else:
            blended_score = gemini_score

        final_score = blended_score + category_penalty
        # backend ClothingItem 호환: name 우선, 구 데이터를 위해 item_name fallback
        scored_results.append({
            "id": candidate_item.get("id"),
            "name": candidate_item.get("name") or candidate_item.get("item_name", "이름 없음"),
            "category": candidate_category,
            "color": candidate_item.get("color", "알 수 없음"),
            "tags": candidate_item.get("tags", ""),
            "image_url": candidate_item.get("image_url", ""),
            "fit_info": candidate_item.get("fit_info", "알 수 없음"),
            "gemini_score": round(gemini_score, 1),
            "clip_score": round(clip_score, 1) if clip_score is not None else None,
            "category_penalty": category_penalty,
            "final_score": round(final_score, 1),
            # PDF2 임계값 기반 의미 라벨
            "similarity_label": _classify_similarity_label(final_score),
            # 디버그/UI용 — 어떤 전처리가 적용됐는지
            "_query_preprocess": preprocess_meta,
        })

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results[:top_n]


# 머지 마커: 팀원 weather_service.py 합본 완료 (v2)

