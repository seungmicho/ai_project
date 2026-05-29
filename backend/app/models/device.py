from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base
from datetime import datetime


class DeviceToken(Base):
    """FCM 디바이스 토큰 저장"""
    __tablename__ = "device_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, nullable=False)
    user_id = Column(String, default="default")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
