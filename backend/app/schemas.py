from pydantic import BaseModel


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
