from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.history import ChatMessage

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    timestamp: str


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """메시지 전송 → DB 저장 → AI 응답"""
    try:
        # 1. 유저 메시지 DB 저장
        user_msg = ChatMessage(
            role="user",
            user_id=request.user_id,
            content=request.message,
        )
        db.add(user_msg)

        # 2. AI 응답 (Gemini API 연동 예정)
        reply = f"'{request.message}'에 대한 AI 응답입니다. (Gemini 연동 예정)"

        # 3. AI 응답 DB 저장
        ai_msg = ChatMessage(
            role="assistant",
            user_id=request.user_id,
            content=reply,
        )
        db.add(ai_msg)
        db.commit()

        return ChatResponse(reply=reply, timestamp=datetime.now().isoformat())

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


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
