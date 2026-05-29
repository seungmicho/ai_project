"""
routers/chat.py

챗봇 엔드포인트.
- POST /chat/message  : 메시지 전송 → 인텐트 감지 → 일정/옷/경로/일반 응답
- GET  /chat/history  : 대화 히스토리 조회
- DELETE /chat/history: 대화 히스토리 초기화
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.history import ChatMessage
from app.services.gemini_service import build_gemini_history
from app.services.intent_service import chat_with_intent

router = APIRouter()

# 이전 대화를 Gemini에 전달할 최대 개수 (많을수록 토큰 소비 증가)
HISTORY_WINDOW = 20


# ──────────────────────────────────────────────
# 요청 / 응답 스키마
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    timestamp: str


# ──────────────────────────────────────────────
# 메시지 전송
# ──────────────────────────────────────────────

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    사용자 메시지를 받아 Gemini Function Calling으로 인텐트를 파악하고
    일정 / 옷 추천 / 경로 / 일반 대화 중 적절한 응답을 반환합니다.

    동작 흐름:
      1. DB에서 최근 N개 대화 히스토리 로드
      2. Gemini에 메시지 전송 (Function Calling 활성화)
      3. 필요 시 일정/옷/경로 함수 자동 실행
      4. 최종 응답을 DB에 저장 후 반환
    """
    try:
        # 1. 이전 대화 히스토리 DB에서 로드
        past_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == request.user_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(max(0,
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == request.user_id)
                .count() - HISTORY_WINDOW
            ))
            .all()
        )

        # 2. Gemini 형식으로 변환
        gemini_history = build_gemini_history(past_messages)

        # 3. 인텐트 서비스 호출 (Function Calling 포함)
        reply = await chat_with_intent(
            message=request.message,
            db=db,
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


# ──────────────────────────────────────────────
# 대화 히스토리 조회
# ──────────────────────────────────────────────

@router.get("/history")
async def get_history(
    user_id: str = "default",
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """대화 히스토리를 오래된 순으로 조회합니다."""
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


# ──────────────────────────────────────────────
# 대화 히스토리 전체 삭제
# ──────────────────────────────────────────────

@router.delete("/history")
async def clear_history(user_id: str = "default", db: Session = Depends(get_db)):
    """특정 유저의 대화 히스토리를 전부 삭제합니다."""
    db.query(ChatMessage).filter(ChatMessage.user_id == user_id).delete()
    db.commit()
    return {"message": "대화 히스토리가 초기화됐어요."}
