from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.history import ChatMessage
from app.services.gemini_service import chat_with_gemini, build_gemini_history

router = APIRouter()

# 히스토리 몇 개까지 Gemini에 보낼지 (너무 많으면 토큰 낭비)
HISTORY_WINDOW = 20


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    timestamp: str


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    메시지 전송 → DB에서 이전 대화 로드 → Gemini 호출 → 응답 저장
    """
    try:
        # 1. 이전 대화 히스토리 DB에서 로드 (최근 N개)
        past_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == request.user_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(HISTORY_WINDOW)
            .all()
        )

        # 2. Gemini 형식으로 변환
        gemini_history = build_gemini_history(past_messages)

        # 3. Gemini API 호출
        reply = await chat_with_gemini(
            message=request.message,
            history=gemini_history,
        )

        # 4. 유저 메시지 DB 저장
        db.add(ChatMessage(
            role="user",
            user_id=request.user_id,
            content=request.message,
        ))

        # 5. AI 응답 DB 저장
        db.add(ChatMessage(
            role="assistant",
            user_id=request.user_id,
            content=reply,
        ))
        db.commit()

        return ChatResponse(reply=reply, timestamp=datetime.now().isoformat())

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")


@router.get("/history")
async def get_history(user_id: str = "default", limit: int = 50, db: Session = Depends(get_db)):
    """대화 히스토리 조회 (오래된 순)"""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return {
        "history": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
        "total": len(messages),
    }


@router.delete("/history")
async def clear_history(user_id: str = "default", db: Session = Depends(get_db)):
    """대화 히스토리 전체 삭제"""
    db.query(ChatMessage).filter(ChatMessage.user_id == user_id).delete()
    db.commit()
    return {"message": "대화 히스토리가 초기화됐어요."}
