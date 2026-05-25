from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base
from datetime import datetime


class Schedule(Base):
    """일정 테이블"""
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, default="")
    date = Column(DateTime, nullable=False)          # 일정 시작 시각
    created_at = Column(DateTime, default=datetime.utcnow)
