from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.clothing import ClothingItem
from app.schemas import ClothingCreate, ClothingResponse
from app.services.gemini_service import chat_with_gemini

router = APIRouter()


@router.get("/", response_model=list[ClothingResponse])
async def get_wardrobe(db: Session = Depends(get_db)):
    return db.query(ClothingItem).all()


@router.post("/", response_model=ClothingResponse)
async def add_clothing(data: ClothingCreate, db: Session = Depends(get_db)):
    item = ClothingItem(
        name=data.name,
        category=data.category,
        color=data.color,
        tags=data.tags,
        image_url=data.image_url,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}")
async def delete_clothing(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ClothingItem).filter(ClothingItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="옷을 찾을 수 없습니다.")
    db.delete(item)
    db.commit()
    return {"message": f"'{item.name}'이(가) 삭제되었습니다."}


@router.get("/recommend")
async def get_recommendation(weather: str = "맑음", temperature: int = 20, db: Session = Depends(get_db)):
    items = db.query(ClothingItem).all()

    if not items:
        return {"recommendation": "등록된 옷이 없습니다. 먼저 옷장에 옷을 추가해 주세요."}

    wardrobe_text = "\n".join(
        f"- {item.name} ({item.category}, {item.color}, 태그: {item.tags})"
        for item in items
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"오늘 날씨는 '{weather}'이고 기온은 {temperature}°C입니다.\n"
                f"내 옷장 목록:\n{wardrobe_text}\n\n"
                "위 옷들 중에서 오늘 날씨에 어울리는 코디를 추천해 주세요."
            ),
        }
    ]
    recommendation = chat_with_gemini(messages)
    return {"recommendation": recommendation}
