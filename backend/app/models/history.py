from sqlalchemy import Column, Integer, String, DateTime, Float
from app.database import Base
from datetime import datetime


class ChatMessage(Base):
    """챗봇 대화 히스토리"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False)        # "user" | "assistant"
    user_id = Column(String, default="default")  # 사용자 구분
    content = Column(String, nullable=False)     # 메시지 내용
    created_at = Column(DateTime, default=datetime.utcnow)


class OutfitHistory(Base):
    """코디 추천 히스토리"""
    __tablename__ = "outfit_history"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, default="Seoul")
    temp = Column(Float)                         # 기온 (℃)
    feels_like = Column(Float)                   # 체감 온도
    description = Column(String, default="")     # 날씨 설명 (예: 맑음)
    outfit_season = Column(String, default="")   # 옷차림 단계
    recommended_ids = Column(String, default="") # 추천된 옷 id (쉼표 구분)
    created_at = Column(DateTime, default=datetime.utcnow)
