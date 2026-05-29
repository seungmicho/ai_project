from fastapi import APIRouter
from pydantic import BaseModel
import os, logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=500,
                system="당신은 AI 개인 패션 비서입니다. 날씨, 옷, 코디 관련 질문에 친근하게 답하세요.",
                messages=[{"role": "user", "content": message}]
            )
            return ChatResponse(reply=response.content[0].text)
        except Exception as e:
            logger.warning("Claude 실패: %s", e)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 AI 개인 패션 비서입니다."},
                    {"role": "user", "content": message}
                ]
            )
            return ChatResponse(reply=response.choices[0].message.content)
        except Exception as e:
            logger.error("OpenAI 실패: %s", e)

    return ChatResponse(reply="AI 키가 설정되지 않았어요. .env 파일을 확인해주세요.")