from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base
from datetime import datetime


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, default="")
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
