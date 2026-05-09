# 사용자가 올린 쇼핑몰 스크린샷이나 텍스트에서 패션 아이템 데이터를 뽑아내는 파서 모듈.
# OpenAI GPT-4o의 Vision 기능(VLM)을 활용해 색상, 카테고리, 핏 정보 구조화
import json
import base64
import logging
import os
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

from dotenv import load_dotenv  

load_dotenv()

logger = logging.getLogger(__name__)

_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


_OUTFIT_ANALYSIS_SCHEMA = """
{
    "purchased_items": [
        {
            "item_name": "상품명 (텍스트 추출)",
            "category": "카테고리 (예: 상의, 하의, 아우터, 신발)",
            "color": "색상 (사진을 보고 판단, 알 수 없으면 '알 수 없음')",
            "fit_info": "핏/실루엣 특징 (예: 와이드, 슬림, 크롭, 오버핏. 사진으로 유추)"
        }
    ]
}
"""


def _strip_markdown_fences(raw_text: str) -> str:
    """AI 응답에서 ```json ... ``` 같은 마크다운 펜스를 걷어냅니다."""
    return raw_text.replace("```json", "").replace("```", "").strip()


def _encode_image_to_base64(image_file) -> str:
    """Streamlit UploadedFile 객체를 GPT-4o Vision API가 읽을 수 있는 base64 문자열로 변환합니다."""
    raw_bytes = image_file.read()
    if not raw_bytes:
        raise ValueError(f"이미지 파일({getattr(image_file, 'name', '알 수 없음')})이 비어 있습니다.")
    return base64.b64encode(raw_bytes).decode("utf-8")


def parse_shopping_screenshot(image_file) -> dict:
    """
    [방식 A] 쇼핑몰 구매 내역 스크린샷을 GPT-4o Vision으로 분석합니다.

    텍스트 OCR과 상품 이미지 분석을 동시에 수행해서
    색상·카테고리·핏 정보를 포함한 구조화된 데이터를 반환합니다.

    Returns:
        {"status": "success", "data": {...}} 또는
        {"status": "error", "message": "..."}
    """
    logger.info("스크린샷 분석 시작: %s", getattr(image_file, "name", "알 수 없음"))

    try:
        base64_image = _encode_image_to_base64(image_file)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    system_prompt = f"""
    너는 쇼핑몰 결제 내역 스크린샷을 분석하는 패션 데이터 전문가야.
    상품명 텍스트를 OCR로 추출하고, 상품 사진을 시각적으로 분석해서 색상과 핏을 판단해.
    마크다운 기호(```) 없이 오직 순수한 JSON으로만 답해.

    [출력 JSON 스키마]
    {_OUTFIT_ANALYSIS_SCHEMA}
    """

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 스크린샷의 구매 내역을 분석해 줘. 사진 속 옷 색상을 특히 신경 써줘."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            temperature=0.1,  # 창의적인 답변보다 일관된 JSON 구조가 중요하므로 낮게 설정
            timeout=30,        # 30초보다 길어지면 타임아웃
        )

        raw_output = _strip_markdown_fences(response.choices[0].message.content)
        outfit_data = json.loads(raw_output)

        if "purchased_items" not in outfit_data:
            raise ValueError("API 응답에 'purchased_items' 키가 없습니다.")

        return {"status": "success", "data": outfit_data}

    except APITimeoutError:
        logger.error("GPT-4o API 타임아웃 — 네트워크 상태나 이미지 크기를 확인하세요.")
        return {"status": "error", "message": "API 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."}
    except RateLimitError:
        logger.error("OpenAI API 요청 한도 초과")
        return {"status": "error", "message": "API 사용량 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."}
    except APIConnectionError:
        logger.error("OpenAI API 연결 실패")
        return {"status": "error", "message": "API 서버에 연결할 수 없습니다. 인터넷 연결을 확인해 주세요."}
    except json.JSONDecodeError as e:
        logger.error("AI 응답 JSON 파싱 실패: %s", e)
        return {"status": "error", "message": "AI가 올바른 형식으로 응답하지 않았습니다. 다시 시도해 주세요."}
    except Exception as e:
        logger.exception("예상치 못한 에러 발생")
        return {"status": "error", "message": str(e)}


def parse_shopping_history(text_input: str) -> dict:
    """
    [방식 B] 사용자가 직접 붙여넣은 구매 내역 텍스트를 분석합니다.

    이미지 없이 텍스트만 있어서 색상이 '알 수 없음'으로 나올 수 있지만,
    그게 정직한 결과입니다 — 없는 정보를 꾸며내는 것보다 낫습니다.
    """
    logger.info("텍스트 구매 내역 분석 시작")

    if not text_input or not text_input.strip():
        return {"status": "error", "message": "분석할 텍스트가 비어 있습니다."}

    system_prompt = """
    너는 쇼핑몰 구매 내역 텍스트를 분석하는 AI야.
    텍스트에서 패션 아이템을 모두 찾아서 JSON으로 답해.
    색상 정보가 없으면 '알 수 없음(직접 입력)'으로 표기해.
    마크다운 기호(```) 없이 순수한 JSON만 출력해.

    출력 형식:
    {"purchased_items": [{"item_name": "...", "category": "...", "color": "...", "fit_info": "..."}]}
    """

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 텍스트에서 구매 내역을 모두 추출해:\n\n{text_input}"},
            ],
            temperature=0.1,
            timeout=15,
        )

        raw_output = _strip_markdown_fences(response.choices[0].message.content)
        outfit_data = json.loads(raw_output)

        if "purchased_items" not in outfit_data:
            raise ValueError("API 응답에 'purchased_items' 키가 없습니다.")

        return {"status": "success", "data": outfit_data}

    except APITimeoutError:
        return {"status": "error", "message": "API 응답 시간이 초과되었습니다."}
    except RateLimitError:
        return {"status": "error", "message": "API 사용량 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."}
    except json.JSONDecodeError as e:
        logger.error("텍스트 파서 JSON 파싱 실패: %s", e)
        return {"status": "error", "message": "AI 응답을 파싱하지 못했습니다."}
    except Exception as e:
        logger.exception("텍스트 파서 예상치 못한 에러")
        return {"status": "error", "message": str(e)}
