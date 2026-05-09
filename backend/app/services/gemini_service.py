import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """당신은 친절하고 유능한 한국어 개인비서입니다.
사용자의 일정 관리, 날씨 기반 코디 추천, 일상적인 질문에 도움을 드립니다.
항상 한국어로 답변하고, 간결하면서도 도움이 되는 답변을 제공하세요."""


def chat_with_gemini(messages: list[dict]) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text
