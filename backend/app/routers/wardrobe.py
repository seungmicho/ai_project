from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.history import OutfitHistory
from app.services.weather_service import get_current_weather, get_outfit_season
import base64 as b64_module

router = APIRouter()


class ClothingItemCreate(BaseModel):
    name: str
    category: str
    color: str = ""
    tags: str = ""
    image_url: str = ""
    fit_info: str = ""


# 임시 인메모리 옷장
wardrobe_db: list[dict] = []
next_id = 1


@router.get("/")
async def get_wardrobe():
    return {"items": wardrobe_db, "total": len(wardrobe_db)}


@router.post("/")
async def add_clothing(item: ClothingItemCreate):
    global next_id
    new_item = {"id": next_id, **item.model_dump()}
    wardrobe_db.append(new_item)
    next_id += 1
    return {"message": "옷이 등록됐어요!", "item": new_item}


@router.put("/{item_id}")
async def update_clothing(item_id: int, item: ClothingItemCreate):
    for i, existing in enumerate(wardrobe_db):
        if existing["id"] == item_id:
            wardrobe_db[i] = {"id": item_id, **item.model_dump()}
            return {"message": "수정됐어요!", "item": wardrobe_db[i]}
    raise HTTPException(status_code=404, detail="해당 아이템을 찾을 수 없어요.")


@router.delete("/{item_id}")
async def delete_clothing(item_id: int):
    global wardrobe_db
    before = len(wardrobe_db)
    wardrobe_db = [i for i in wardrobe_db if i["id"] != item_id]
    if len(wardrobe_db) == before:
        raise HTTPException(status_code=404, detail="해당 아이템을 찾을 수 없어요.")
    return {"message": f"아이템 {item_id}이 삭제됐어요."}


@router.post("/upload-image")
async def upload_clothing_image(file: UploadFile = File(...)):
    global next_id
    try:
        from app.skills.shopping_parser import parse_shopping_screenshot

        contents = await file.read()

        # 이미지를 base64 data URL로 변환해서 저장
        content_type = file.content_type or "image/jpeg"
        b64_image = b64_module.b64encode(contents).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64_image}"

        class FakeFile:
            def __init__(self, data, name):
                self._data = data
                self.name = name
            def read(self):
                return self._data
            def seek(self, pos):
                pass

        fake_file = FakeFile(contents, file.filename or "upload.jpg")
        result = parse_shopping_screenshot(fake_file)

        if result["status"] == "success":
            items = result["data"].get("purchased_items", [])
            added_items = []
            for item in items:
                new_item = {"id": next_id, **item, "image_url": data_url}
                wardrobe_db.append(new_item)
                next_id += 1
                added_items.append(new_item)
            return {"message": f"{len(added_items)}개 옷이 등록됐어요!", "items": added_items}
        else:
            return {"message": result.get("message", "분석 실패")}

    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}


@router.get("/recommend")
async def get_recommendation(city: str = "Seoul", db: Session = Depends(get_db)):
    try:
        weather = await get_current_weather(city)
    except Exception:
        raise HTTPException(status_code=502, detail="날씨 정보를 불러오지 못했어요.")

    season = get_outfit_season(weather["temp"])
    season_keywords = {
        "매우 더움": ["여름", "반팔", "반바지", "얇은"],
        "더움": ["여름", "반팔", "얇은"],
        "따뜻함": ["봄", "가을", "긴팔", "가디건"],
        "선선함": ["봄", "가을", "자켓", "가디건"],
        "쌀쌀함": ["겨울", "코트", "니트", "두꺼운"],
        "매우 추움": ["겨울", "패딩", "두꺼운", "방한"],
    }
    keywords = season_keywords.get(season, [])
    recommended = [
        item for item in wardrobe_db
        if any(kw in item.get("tags", "").split(",") for kw in keywords)
    ]
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
        "tip": f"오늘 {weather['city']}은 {weather['temp']}℃ ({season})이에요.",
    }


@router.get("/history")
async def get_outfit_history(limit: int = 20, db: Session = Depends(get_db)):
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