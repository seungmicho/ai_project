from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import get_db
from app.models.device import DeviceToken

router = APIRouter()


class TokenRequest(BaseModel):
    token: str
    user_id: str = "default"


@router.post("/token")
async def save_token(req: TokenRequest, db: Session = Depends(get_db)):
    """앱 실행 시 FCM 토큰을 저장합니다."""
    existing = db.query(DeviceToken).filter(DeviceToken.user_id == req.user_id).first()

    if existing:
        existing.token = req.token
        existing.updated_at = datetime.utcnow()
    else:
        db.add(DeviceToken(token=req.token, user_id=req.user_id))

    db.commit()
    return {"message": "토큰이 저장됐어요."}


@router.get("/token")
async def get_token(user_id: str = "default", db: Session = Depends(get_db)):
    """push 서버에서 FCM 토큰을 가져갑니다."""
    device = db.query(DeviceToken).filter(DeviceToken.user_id == user_id).first()
    if not device:
        return {"token": None}
    return {"token": device.token, "updated_at": device.updated_at}
