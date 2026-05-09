from sqlalchemy import Column, Integer, String
from app.database import Base


class ClothingItem(Base):
    __tablename__ = "clothing_items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)   # 상의, 하의, 아우터, 신발 등
    color = Column(String, default="")
    tags = Column(String, default="")           # 쉼표 구분 태그 (예: "캐주얼,봄")
    image_url = Column(String, default="")
