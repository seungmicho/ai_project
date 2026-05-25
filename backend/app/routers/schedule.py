from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, date

from app.database import get_db
from backend.app.models.calendar_agent import Schedule
from app.schemas import ScheduleCreate, ScheduleResponse
from app.services.gemini_service import chat_with_gemini

router = APIRouter()


@router.get("/", response_model=list[ScheduleResponse])
async def get_schedules(db: Session = Depends(get_db)):
    return db.query(Schedule).order_by(Schedule.date.asc()).all()


@router.post("/", response_model=ScheduleResponse)
async def create_schedule(data: ScheduleCreate, db: Session = Depends(get_db)):
    schedule = Schedule(
        title=data.title,
        description=data.description,
        date=data.date,
    )
    db.add(schedule)
    db.commit()
    db.refresh(schedule)
    return schedule


@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다.")
    db.delete(schedule)
    db.commit()
    return {"message": f"일정 '{schedule.title}'이(가) 삭제되었습니다."}


@router.get("/briefing")
async def get_briefing(db: Session = Depends(get_db)):
    today = date.today()
    schedules = (
        db.query(Schedule)
        .filter(Schedule.date >= datetime(today.year, today.month, today.day))
        .filter(Schedule.date < datetime(today.year, today.month, today.day + 1))
        .order_by(Schedule.date.asc())
        .all()
    )

    if not schedules:
        schedule_text = "오늘 등록된 일정이 없습니다."
    else:
        items = "\n".join(
            f"- {s.date.strftime('%H:%M')} {s.title}"
            + (f" ({s.description})" if s.description else "")
            for s in schedules
        )
        schedule_text = f"오늘 일정:\n{items}"

    messages = [
        {
            "role": "user",
            "content": f"오늘 날짜는 {today}입니다. 다음 일정을 바탕으로 친절하게 하루 브리핑을 해주세요.\n{schedule_text}",
        }
    ]
    briefing = chat_with_gemini(messages)
    return {"briefing": briefing}