from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.history import OutfitHistory
from app.services.weather_service import get_current_weather, get_outfit_season

router = APIRouter()


class ClothingItemCreate(BaseModel):
    name: str
    category: str   # 상의 | 하의 | 아우터 | 신발 | 기타
    color: str = ""
    tags: str = ""  # 쉼표 구분 (예: "캐주얼,봄,얇은")
    image_url: str = ""


# 임시 인메모리 옷장 (추후 DB로 교체)
wardrobe_db: list[dict] = []
next_id = 1


@router.get("/")
async def get_wardrobe():
    """등록된 옷 목록 전체 조회"""
    return {"items": wardrobe_db, "total": len(wardrobe_db)}


@router.post("/")
async def add_clothing(item: ClothingItemCreate):
    """옷 등록"""
    global next_id
    new_item = {"id": next_id, **item.model_dump()}
    wardrobe_db.append(new_item)
    next_id += 1
    return {"message": "옷이 등록됐어요!", "item": new_item}


@router.delete("/{item_id}")
async def delete_clothing(item_id: int):
    """옷 삭제"""
    global wardrobe_db
    before = len(wardrobe_db)
    wardrobe_db = [i for i in wardrobe_db if i["id"] != item_id]
    if len(wardrobe_db) == before:
        raise HTTPException(status_code=404, detail="해당 아이템을 찾을 수 없어요.")
    return {"message": f"아이템 {item_id}이 삭제됐어요."}


@router.get("/recommend")
async def get_recommendation(city: str = "Seoul", db: Session = Depends(get_db)):
    """날씨 기반 코디 추천 + 히스토리 저장"""
    # 1. 날씨 조회
    try:
        weather = await get_current_weather(city)
    except Exception:
        raise HTTPException(status_code=502, detail="날씨 정보를 불러오지 못했어요. API 키를 확인해주세요.")

    # 2. 옷차림 단계 분류
    season = get_outfit_season(weather["temp"])

    # 3. 태그 기반 추천
    season_keywords = {
        "매우 더움": ["여름", "반팔", "반바지", "얇은"],
        "더움":     ["여름", "반팔", "얇은"],
        "따뜻함":   ["봄", "가을", "긴팔", "가디건"],
        "선선함":   ["봄", "가을", "자켓", "가디건"],
        "쌀쌀함":   ["겨울", "코트", "니트", "두꺼운"],
        "매우 추움": ["겨울", "패딩", "두꺼운", "방한"],
    }
    keywords = season_keywords.get(season, [])
    recommended = [
        item for item in wardrobe_db
        if any(kw in item.get("tags", "").split(",") for kw in keywords)
    ]

    # 4. 추천 히스토리 DB 저장 ✅
    history_entry = OutfitHistory(
        city=city,
        temp=weather["temp"],
        feels_like=weather["feels_like"],
        description=weather["description"],
        outfit_season=season,
        recommended_ids=",".join(str(i["id"]) for i in recommended),
    )
    db.add(history_entry)
    db.commit()

    return {
        "weather": weather,
        "outfit_season": season,
        "recommended_items": recommended,
        "tip": f"오늘 {weather['city']}은 {weather['temp']}℃ ({season})이에요. "
               f"체감온도는 {weather['feels_like']}℃예요.",
    }


@router.get("/history")
async def get_outfit_history(limit: int = 20, db: Session = Depends(get_db)):
    """코디 추천 히스토리 조회"""
    records = (
        db.query(OutfitHistory)
        .order_by(OutfitHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "history": [
            {
                "id": r.id,
                "city": r.city,
                "temp": r.temp,
                "feels_like": r.feels_like,
                "description": r.description,
                "outfit_season": r.outfit_season,
                "recommended_ids": r.recommended_ids.split(",") if r.recommended_ids else [],
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
        "total": len(records),
    }
