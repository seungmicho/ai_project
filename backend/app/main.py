from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# .env 절대 경로로 지정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from app.routers import wardrobe
from app.routers import chat
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✅ 데이터베이스 초기화 완료")
    yield
    print("👋 서버 종료")


app = FastAPI(
    title="패션 AI 어시스턴트 API",
    description="날씨 기반 코디 추천, 옷장 관리 백엔드 서버",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wardrobe.router, prefix="/wardrobe", tags=["옷장"])
app.include_router(chat.router, tags=["채팅"])