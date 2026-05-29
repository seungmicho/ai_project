from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.routers import chat, schedule, wardrobe, route, device
from app.database import init_db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✅ 데이터베이스 초기화 완료")
    yield
    print("👋 서버 종료")


app = FastAPI(
    title="AI 개인비서 API",
    description="일정 관리, 날씨 코디 추천, AI 챗봇 백엔드 서버",
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

app.include_router(chat.router, prefix="/chat", tags=["챗봇"])
app.include_router(schedule.router, prefix="/schedules", tags=["일정"])
app.include_router(wardrobe.router, prefix="/wardrobe", tags=["옷장"])
app.include_router(route.router, prefix="/route", tags=["경로"])
app.include_router(device.router, prefix="/device", tags=["디바이스"])
