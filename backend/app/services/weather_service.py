import httpx
import os
from typing import Optional

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"


async def get_current_weather(city: str = "Seoul") -> dict:
    """현재 날씨 조회"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/weather",
            params={
                "q": city,
                "appid": WEATHER_API_KEY,
                "units": "metric",   # 섭씨
                "lang": "kr",        # 한국어 설명
            },
        )
        response.raise_for_status()
        data = response.json()

    return {
        "city": data["name"],
        "temp": round(data["main"]["temp"]),           # 현재 기온 (℃)
        "feels_like": round(data["main"]["feels_like"]),  # 체감 온도
        "temp_min": round(data["main"]["temp_min"]),   # 최저 기온
        "temp_max": round(data["main"]["temp_max"]),   # 최고 기온
        "humidity": data["main"]["humidity"],           # 습도 (%)
        "description": data["weather"][0]["description"],  # 날씨 설명
        "icon": data["weather"][0]["icon"],             # 날씨 아이콘 코드
        "wind_speed": data["wind"]["speed"],            # 풍속 (m/s)
    }


def get_outfit_season(temp: int) -> str:
    """기온에 따른 계절/옷차림 단계 분류"""
    if temp >= 28:
        return "매우 더움"    # 반팔, 반바지
    elif temp >= 23:
        return "더움"         # 반팔
    elif temp >= 17:
        return "따뜻함"       # 긴팔, 얇은 가디건
    elif temp >= 12:
        return "선선함"       # 자켓, 가디건
    elif temp >= 5:
        return "쌀쌀함"       # 코트, 두꺼운 니트
    else:
        return "매우 추움"    # 패딩, 두꺼운 코트
