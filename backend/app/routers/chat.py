from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.chat import ChatMessage
from app.schemas import MessageRequest, MessageResponse
from app.servives.claude_servive import chat_with_claude

router = APIRouter()


@router.post("/message", response_model=MessageResponse)
async def send_message(req: MessageRequest, db: Session = Depends(get_db)):
    # DB에서 최근 20개 대화 불러오기
    history = (
        db.query(ChatMessage)
        .order_by(ChatMessage.created_at.asc())
        .limit(20)
        .all()
    )
    messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})

    reply = chat_with_claude(messages)

    # 사용자 메시지 & AI 응답 저장
    db.add(ChatMessage(role="user", content=req.message))
    db.add(ChatMessage(role="assistant", content=reply))
    db.commit()

    return {"reply": reply}


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return {
        "history": [
            {"role": msg.role, "content": msg.content, "created_at": msg.created_at}
            for msg in messages
        ]
    }


@router.delete("/history")
async def clear_history(db: Session = Depends(get_db)):
    db.query(ChatMessage).delete()
    db.commit()
    return {"message": "대화 기록이 삭제되었습니다."}
