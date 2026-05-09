import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

SYSTEM_PROMPT = """당신은 친절하고 유능한 한국어 개인비서입니다.
사용자의 일정 관리, 날씨 기반 코디 추천, 일상적인 질문에 도움을 드립니다.
항상 한국어로 답변하고, 간결하면서도 도움이 되는 답변을 제공하세요."""


def chat_with_gemini(messages: list[dict]) -> str:
    # 첫 메시지에 시스템 프롬프트 주입
    contents = []
    for i, msg in enumerate(messages):
        role = "user" if msg["role"] == "user" else "model"
        text = msg["content"]
        if i == 0 and role == "user":
            text = f"{SYSTEM_PROMPT}\n\n{text}"
        contents.append({"role": role, "parts": [{"text": text}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
    }

    response = httpx.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]