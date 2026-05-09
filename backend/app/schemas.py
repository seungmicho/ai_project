from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    reply: str


class ScheduleCreate(BaseModel):
    title: str
    description: str = ""
    start_time: datetime
    end_time: Optional[datetime] = None


class ScheduleResponse(BaseModel):
    id: int
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class ClothingCreate(BaseModel):
    name: str
    category: str
    color: str = ""
    tags: str = ""
    image_url: str = ""


class ClothingResponse(BaseModel):
    id: int
    name: str
    category: str
    color: str
    tags: str
    image_url: str

    class Config:
        from_attributes = True
