# 의류 이미지 전처리 모듈 — 임베딩 직전에 적용해서 두 가지 문제를 해결합니다.
#
# 1) 배경 노이즈 제거 (PDF2: 캡스톤 CLIP 실험에서 발견)
#    같은 카테고리 옷도 배경이 다르면 코사인 유사도가 망가짐.
#    예) 코트 vs 자켓 = 0.508 < 코트 vs 운동화 = 0.632 (배경 영향)
#    배경 통일 후: 코트 vs 자켓 = 0.932 > 코트 vs 운동화 = 0.749 (정상)
#
# 2) 멀티 의류 분리 (PDF1: 나민정님 보고서의 "해결할 부분")
#    한 사진에 상의·하의·신발이 같이 있으면 모두 동일 임베딩을 갖게 되는 버그.
#    객체 탐지 파이프라인을 임베딩 앞단에 두면 해결됨.
#
# 두 문제 모두 "이미지에서 의류 영역만 잘라내면" 한 번에 해결되므로,
# Gemini Vision에게 의류 bounding box를 받아서 PIL로 crop하는 방식을 채택했습니다.
# 추가 라이브러리(YOLO/SAM/rembg) 없이 이미 쓰고 있는 google-genai만으로 동작합니다.

import base64
import io
import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# 옷장 DB에 저장할 썸네일 사이즈 (data URL이라 너무 크면 JSON이 비대해짐)
_THUMBNAIL_MAX_SIZE = 320

logger = logging.getLogger(__name__)

_gemini_client = None
if os.environ.get("GEMINI_API_KEY"):
    try:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        logger.warning("preprocess: Gemini 클라이언트 초기화 실패 — 의류 검출 비활성화: %s", e)
else:
    logger.warning("preprocess: GEMINI_API_KEY 비어있음 — 의류 영역 자동 추출 비활성화.")


# Gemini가 반환하는 좌표는 [0, 1000] 정규화 좌표 (Vision API 표준).
# PIL의 픽셀 좌표로 변환할 때 사용합니다.
_GEMINI_BBOX_SCALE = 1000


def detect_clothing_regions(image_bytes: bytes) -> list[dict]:
    """
    Gemini Vision으로 이미지 안의 의류 객체들을 검출합니다.

    한 사진에 여러 의류가 있으면 각각의 bounding box를 반환합니다.
    PDF1의 "해결할 부분: 한 사진에 여러 옷이 있을 때 모두 동일 임베딩" 문제를
    이 함수가 직접 해결합니다.

    Returns:
        [
            {"category": "상의", "bbox": [y_min, x_min, y_max, x_max], "color": "검정"},
            {"category": "하의", "bbox": [...], "color": "베이지"},
            ...
        ]
        bbox는 Gemini 표준대로 [0, 1000] 범위 정규화 좌표.
        검출 실패 시 빈 리스트.
    """
    if not image_bytes:
        return []
    if _gemini_client is None:
        # 키 없을 때는 검출 스킵 → 호출부가 원본 이미지로 폴백 (graceful degradation)
        return []

    system_prompt = """
    너는 패션 이미지 분석가야. 사진 속의 모든 의류·신발·가방 아이템을 객체별로 검출해서
    각각의 카테고리와 bounding box를 반환해.

    bounding box는 [y_min, x_min, y_max, x_max] 형식의 정수 배열이고,
    좌표 범위는 0~1000 (이미지 크기 기준 정규화).

    배경 사람·풍경은 무시하고 의류 객체만 골라내.
    같은 카테고리 안에서도 시각적으로 다른 객체면 별도 박스로 분리해 (예: 상의1, 상의2).

    오직 JSON 배열만 출력해. 마크다운 펜스 없이.
    """

    schema_hint = '{"items": [{"category": "상의|하의|아우터|신발|가방", "bbox": [y_min, x_min, y_max, x_max], "color": "주요 색상"}]}'

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                f"이 사진의 모든 의류 객체를 검출해서 다음 형식으로 답해: {schema_hint}",
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            logger.warning("의류 검출 응답 형식 이상 — 빈 리스트로 처리")
            return []

        # 좌표 sanity check
        valid = []
        for item in items:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            y_min, x_min, y_max, x_max = bbox
            if y_min >= y_max or x_min >= x_max:
                continue  # 잘못된 박스
            valid.append(item)
        return valid

    except json.JSONDecodeError as e:
        logger.warning("의류 검출 JSON 파싱 실패: %s", e)
        return []
    except Exception as e:
        logger.error("Gemini 의류 검출 실패: %s", e)
        return []


def crop_clothing_region(image_bytes: bytes, bbox: list[int], padding_ratio: float = 0.05) -> Optional[bytes]:
    """
    bbox에 해당하는 영역만 잘라낸 이미지 바이트를 반환합니다.

    PDF2 실험에서 본 "배경 노이즈" 문제를 줄이기 위해, 의류 영역만 추출해
    후속 임베딩 모델이 배경에 휘둘리지 않도록 합니다.

    Args:
        image_bytes: 원본 이미지 바이트
        bbox: [y_min, x_min, y_max, x_max] (0~1000 정규화)
        padding_ratio: bbox를 살짝 키워서 잘라낼 비율 (의류 가장자리가 잘리는 것 방지)

    Returns:
        잘린 이미지의 JPEG 바이트, 실패 시 None
    """
    if not _PIL_AVAILABLE:
        logger.warning("PIL 미설치 — crop 불가, 원본 바이트 그대로 반환")
        return image_bytes

    if not image_bytes or not bbox or len(bbox) != 4:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        y_min, x_min, y_max, x_max = bbox
        # 정규화 좌표 → 픽셀 좌표
        px_x_min = int(x_min / _GEMINI_BBOX_SCALE * width)
        px_y_min = int(y_min / _GEMINI_BBOX_SCALE * height)
        px_x_max = int(x_max / _GEMINI_BBOX_SCALE * width)
        px_y_max = int(y_max / _GEMINI_BBOX_SCALE * height)

        # 패딩 (의류 가장자리가 잘리지 않도록)
        pad_x = int((px_x_max - px_x_min) * padding_ratio)
        pad_y = int((px_y_max - px_y_min) * padding_ratio)
        px_x_min = max(0, px_x_min - pad_x)
        px_y_min = max(0, px_y_min - pad_y)
        px_x_max = min(width, px_x_max + pad_x)
        px_y_max = min(height, px_y_max + pad_y)

        cropped = image.crop((px_x_min, px_y_min, px_x_max, px_y_max))

        buffer = io.BytesIO()
        cropped.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    except Exception as e:
        logger.error("이미지 crop 실패: %s", e)
        return None


def prepare_images_for_embedding(image_bytes: bytes, multi_item_mode: bool = True) -> list[dict]:
    """
    임베딩 모델에 넣기 직전의 통합 전처리 진입점입니다.

    - multi_item_mode=True (기본): 한 사진에서 여러 의류를 검출하면 각각 분리 반환.
      검출 실패 시 원본 이미지 그대로 1개 반환 (graceful degradation).
    - multi_item_mode=False: 검출은 시도하되 가장 큰 박스 1개만 잘라서 반환 (단일 의류 모드).

    Returns:
        [
            {
                "image_bytes": <전처리된 이미지 바이트>,
                "meta": {
                    "preprocessed": True/False,  # crop 적용됐는지
                    "detected_category": "상의" | None,
                    "detected_color": "검정" | None,
                    "bbox": [...] | None,
                    "source": "cropped" | "original_fallback",
                }
            },
            ...
        ]
    """
    detected = detect_clothing_regions(image_bytes)

    if not detected:
        # 객체 검출 실패 → 원본 그대로 (배경 노이즈는 못 줄이지만 동작은 보장)
        logger.info("의류 객체 검출 0건 — 원본 이미지로 폴백")
        return [{
            "image_bytes": image_bytes,
            "meta": {
                "preprocessed": False,
                "detected_category": None,
                "detected_color": None,
                "bbox": None,
                "source": "original_fallback",
                # 원본도 옷장에 표시되도록 썸네일 인코딩
                "thumbnail_data_url": image_bytes_to_thumbnail_data_url(image_bytes),
            },
        }]

    # 단일 의류 모드: 가장 큰 박스 하나만
    if not multi_item_mode:
        largest = max(detected, key=lambda d: _bbox_area(d.get("bbox", [0, 0, 0, 0])))
        cropped = crop_clothing_region(image_bytes, largest["bbox"])
        if cropped is None:
            return [{"image_bytes": image_bytes, "meta": {"preprocessed": False, "source": "crop_failed_fallback"}}]
        return [{
            "image_bytes": cropped,
            "meta": {
                "preprocessed": True,
                "detected_category": largest.get("category"),
                "detected_color": largest.get("color"),
                "bbox": largest["bbox"],
                "source": "cropped",
                "thumbnail_data_url": image_bytes_to_thumbnail_data_url(cropped),
            },
        }]

    # 멀티 의류 모드: 모든 박스를 잘라서 반환
    results: list[dict] = []
    for item in detected:
        cropped = crop_clothing_region(image_bytes, item["bbox"])
        if cropped is None:
            continue
        results.append({
            "image_bytes": cropped,
            "meta": {
                "preprocessed": True,
                "detected_category": item.get("category"),
                "detected_color": item.get("color"),
                "bbox": item["bbox"],
                "source": "cropped",
                # 각 분리된 영역을 옷장 썸네일로 보존 (시연용 핵심)
                "thumbnail_data_url": image_bytes_to_thumbnail_data_url(cropped),
            },
        })

    if not results:
        # 모든 crop 실패 → 원본 폴백
        return [{
            "image_bytes": image_bytes,
            "meta": {
                "preprocessed": False,
                "source": "all_crops_failed_fallback",
                "thumbnail_data_url": image_bytes_to_thumbnail_data_url(image_bytes),
            },
        }]

    return results


def _bbox_area(bbox: list[int]) -> int:
    """bbox 면적 계산 (가장 큰 객체 선택용)."""
    if not bbox or len(bbox) != 4:
        return 0
    y_min, x_min, y_max, x_max = bbox
    return max(0, y_max - y_min) * max(0, x_max - x_min)


def image_bytes_to_thumbnail_data_url(image_bytes: bytes, max_size: int = _THUMBNAIL_MAX_SIZE) -> Optional[str]:
    """
    잘린 의류 이미지를 썸네일로 다운샘플링한 뒤 base64 data URL로 변환합니다.

    옷장 DB(JSON)에 image_url 필드로 그대로 박아두면, Streamlit data_editor의
    ImageColumn이 추가 라이브러리 없이도 렌더링해줍니다 — "내 옷장의 이미지들이
    나와야 좋다"는 시연 요구사항을 가벼운 방식으로 만족시키는 트릭.

    너무 큰 원본을 그대로 저장하면 wardrobe_db.json이 비대해지므로
    한 변 320px 정도로 줄여서 인코딩합니다.
    """
    if not _PIL_AVAILABLE or not image_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logger.warning("썸네일 data URL 생성 실패: %s", e)
        return None
