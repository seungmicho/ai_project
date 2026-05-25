import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 모델 초기화
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "당신은 사용자의 개인 AI 비서입니다. "
        "일정 관리, 날씨 기반 코디 추천, 일상 대화를 친절하고 간결하게 도와주세요. "
        "항상 한국어로 답변하세요."
    ),
)


def build_gemini_history(db_messages: list) -> list[dict]:
    """
    DB에서 꺼낸 ChatMessage 목록을 Gemini 형식으로 변환.
    Gemini는 role이 'user' | 'model' 이어야 하고,
    user/model이 반드시 번갈아 나와야 해요.
    """
    history = []
    for msg in db_messages:
        role = "model" if msg.role == "assistant" else "user"
        history.append({"role": role, "parts": [msg.content]})
    return history


async def chat_with_gemini(message: str, history: list[dict] = None) -> str:
    """
    Gemini와 대화.
    history: build_gemini_history()로 변환된 이전 대화 목록
    """
    try:
        chat_session = model.start_chat(history=history or [])
        response = await chat_session.send_message_async(message)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API 오류: {e}")


async def generate_briefing(schedules: list[dict], weather: dict = None) -> str:
    """오늘 일정 + 날씨를 받아 아침 브리핑 생성"""
    weather_text = ""
    if weather:
        weather_text = (
            f"오늘 날씨: {weather['city']} {weather['temp']}℃, "
            f"{weather['description']}, 체감 {weather['feels_like']}℃\n"
        )

    if not schedules:
        prompt = (
            f"{weather_text}"
            "오늘 등록된 일정이 없습니다. "
            "날씨 정보를 포함해서 여유로운 하루가 될 것 같다고 친근하게 브리핑해주세요."
        )
    else:
        schedule_text = "\n".join([
            f"- {s['title']} ({s['start_time']}): {s.get('description', '')}"
            for s in schedules
        ])
        prompt = (
            f"{weather_text}"
            f"오늘의 일정:\n{schedule_text}\n\n"
            "날씨와 일정을 바탕으로 친근하고 간결한 아침 브리핑을 2~3문장으로 작성해주세요."
        )

    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"브리핑 생성 오류: {e}")


async def generate_outfit_recommendation(weather: dict, clothing_items: list[dict]) -> str:
    """날씨 + 보유 옷 목록으로 코디 추천"""
    items_text = (
        "\n".join([
            f"- {item['name']} ({item['category']}) 태그: {item.get('tags', '없음')}"
            for item in clothing_items
        ])
        if clothing_items else "등록된 옷이 없습니다."
    )

    prompt = (
        f"현재 날씨: {weather['city']} {weather['temp']}℃, {weather['description']}\n"
        f"보유 옷 목록:\n{items_text}\n\n"
        "이 날씨에 맞는 코디를 보유 옷 중에서 구체적으로 추천해주세요. "
        "없으면 어떤 종류의 옷이 필요한지 알려주세요. 2~3문장으로 간결하게 답하세요."
    )

    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"코디 추천 오류: {e}")
