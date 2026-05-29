"""
intent_service.py

새 google-genai SDK(1.x)를 사용해 챗봇 메시지의 의도를 파악하고
일정 / 옷 추천 / 경로 기능을 자동으로 실행합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, date, timedelta
from typing import Any

import httpx
from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from app.services.gemini_service import build_gemini_history

# ─────────────────────────────────────────────
# 구글 캘린더 서비스 초기화
# ─────────────────────────────────────────────
def _get_calendar_service():
    """구글 캘린더 API 서비스를 반환합니다."""
    from pathlib import Path
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    base_dir = Path(__file__).parent.parent.parent  # backend/
    token_file = base_dir / "token.json"
    credentials_file = base_dir / "credentials.json"

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_file.exists():
                raise FileNotFoundError("credentials.json 파일이 없습니다.")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=creds)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 클라이언트 & 모델 설정
# ─────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "당신은 사용자의 개인 AI 비서입니다. "
    "일정 관리, 날씨 기반 코디 추천, 경로 안내, 일상 대화를 친절하고 간결하게 도와주세요. "
    "항상 한국어로 자연스럽게 답변하세요. "
    "함수 호출 결과를 받으면 그 데이터를 바탕으로 친근하게 정리해서 알려주세요."
)

# ─────────────────────────────────────────────
# Function Declarations (Tool 정의)
# ─────────────────────────────────────────────
_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_schedules",
            description="사용자의 일정을 날짜별로 조회합니다. '오늘 일정', '내일 뭐 있어' 같은 요청에 사용합니다.",
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "조회할 날짜 (YYYY-MM-DD). 없으면 오늘 날짜 사용."
                    }
                }
            }
        ),
        types.FunctionDeclaration(
            name="create_schedule",
            description="새로운 일정을 등록합니다. '오전 9시에 회의 추가해줘', '매일 오전 10시에 약먹기' 같은 요청에 사용합니다. 반복 일정도 등록 가능합니다.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "일정 제목"},
                    "date":  {"type": "string", "description": "날짜 (YYYY-MM-DD). 반복 일정이면 시작 날짜."},
                    "time":  {"type": "string", "description": "시작 시간 (HH:MM, 24시간제)"},
                    "description": {"type": "string", "description": "일정 상세 설명 (선택)"},
                    "recurrence": {"type": "string", "description": "반복 규칙. 매일=DAILY, 매주=WEEKLY, 매월=MONTHLY, 없으면 빈 문자열"},
                },
                "required": ["title", "date", "time"]
            }
        ),
        types.FunctionDeclaration(
            name="delete_schedule",
            description="일정을 ID로 삭제합니다. 먼저 get_schedules로 id를 확인 후 삭제하세요.",
            parameters={
                "type": "object",
                "properties": {
                    "schedule_id": {"type": "string", "description": "삭제할 일정의 Google Calendar 이벤트 ID"}
                },
                "required": ["schedule_id"]
            }
        ),
        types.FunctionDeclaration(
            name="get_outfit_recommendation",
            description="현재 날씨를 기반으로 오늘 입을 옷을 추천합니다. '오늘 뭐 입어?', '코디 추천해줘' 같은 요청에 사용합니다.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "날씨를 조회할 도시명 (기본값: 서울)"}
                }
            }
        ),
        types.FunctionDeclaration(
            name="get_route",
            description="출발지에서 목적지까지의 경로와 추천 출발 시간을 안내합니다. '강남역에서 홍대까지 10시까지 가려면?' 같은 요청에 사용합니다.",
            parameters={
                "type": "object",
                "properties": {
                    "start":        {"type": "string", "description": "출발지 장소명"},
                    "end":          {"type": "string", "description": "목적지 장소명"},
                    "arrival_time": {"type": "string", "description": "도착 목표 시간 (HH:MM)"},
                    "transport":    {"type": "string", "description": "'car'(자동차) 또는 'transit'(대중교통)"},
                },
                "required": ["start", "end", "arrival_time"]
            }
        ),
    ])
]

# ─────────────────────────────────────────────
# 날짜 파싱 헬퍼
# ─────────────────────────────────────────────
def _resolve_date(date_str: str | None) -> date:
    today = date.today()
    if not date_str or date_str.strip().lower() in ("today", "오늘", ""):
        return today
    if date_str.strip().lower() in ("tomorrow", "내일"):
        return today + timedelta(days=1)
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return today


# ─────────────────────────────────────────────
# Tool 실행 함수
# ─────────────────────────────────────────────
def _exec_get_schedules(args: dict, db=None) -> dict:
    target = _resolve_date(args.get("date"))
    time_min = datetime(target.year, target.month, target.day, 0, 0, 0).isoformat() + "+09:00"
    time_max = datetime(target.year, target.month, target.day, 23, 59, 59).isoformat() + "+09:00"

    try:
        service = _get_calendar_service()
        items = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        ).execute().get("items", [])

        if not items:
            return {"date": target.isoformat(), "count": 0, "schedules": [],
                    "message": f"{target.month}월 {target.day}일에 등록된 일정이 없습니다."}

        schedules = []
        for item in items:
            start = item["start"].get("dateTime", item["start"].get("date", ""))
            try:
                from dateutil import parser as dateparser
                dt = dateparser.parse(start)
                time_str = dt.strftime("%H:%M")
            except Exception:
                time_str = start
            schedules.append({
                "id": item["id"],
                "title": item.get("summary", "(제목 없음)"),
                "time": time_str,
                "description": item.get("description", ""),
            })

        return {"date": target.isoformat(), "count": len(schedules), "schedules": schedules}

    except Exception as e:
        logger.error("구글 캘린더 일정 조회 오류: %s", e)
        return {"success": False, "error": f"구글 캘린더 조회 실패: {e}"}


def _exec_create_schedule(args: dict, db=None) -> dict:
    title       = args.get("title", "새 일정")
    date_str    = args.get("date", date.today().isoformat())
    time_str    = args.get("time", "09:00")
    description = args.get("description", "")
    recurrence  = args.get("recurrence", "")

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return {"success": False, "error": f"날짜/시간 형식 오류: {date_str} {time_str}"}

    end_dt = dt + timedelta(hours=1)
    event_body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": dt.isoformat(), "timeZone": "Asia/Seoul"},
        "end":   {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Seoul"},
    }

    # 반복 규칙 추가
    recurrence_map = {
        "DAILY":   "RRULE:FREQ=DAILY",
        "WEEKLY":  "RRULE:FREQ=WEEKLY",
        "MONTHLY": "RRULE:FREQ=MONTHLY",
    }
    if recurrence and recurrence.upper() in recurrence_map:
        event_body["recurrence"] = [recurrence_map[recurrence.upper()]]
        recurrence_label = {"DAILY": "매일", "WEEKLY": "매주", "MONTHLY": "매월"}[recurrence.upper()]
    else:
        recurrence_label = None

    try:
        service = _get_calendar_service()
        event = service.events().insert(calendarId="primary", body=event_body).execute()

        if recurrence_label:
            msg = f"✅ '{title}' 일정이 {recurrence_label} {dt.strftime('%H:%M')}에 반복 등록됐어요!"
        else:
            msg = f"✅ '{title}' 일정이 {dt.strftime('%m월 %d일 %H:%M')}에 구글 캘린더에 등록됐어요!"

        return {
            "success": True,
            "id": event["id"],
            "title": title,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
            "recurrence": recurrence_label,
            "message": msg,
        }
    except Exception as e:
        logger.error("구글 캘린더 일정 생성 오류: %s", e)
        return {"success": False, "error": f"구글 캘린더 등록 실패: {e}"}


def _exec_delete_schedule(args: dict, db=None) -> dict:
    schedule_id = args.get("schedule_id")
    if not schedule_id:
        return {"success": False, "error": "schedule_id가 필요합니다."}

    try:
        service = _get_calendar_service()
        # 삭제 전 제목 확인
        event = service.events().get(calendarId="primary", eventId=schedule_id).execute()
        title = event.get("summary", "(제목 없음)")
        service.events().delete(calendarId="primary", eventId=schedule_id).execute()
        return {"success": True, "message": f"🗑️ '{title}' 일정을 구글 캘린더에서 삭제했어요."}
    except Exception as e:
        logger.error("구글 캘린더 일정 삭제 오류: %s", e)
        return {"success": False, "error": f"구글 캘린더 삭제 실패: {e}"}


def _exec_get_outfit_recommendation(args: dict) -> dict:
    city = args.get("city", "서울")
    try:
        import requests as _req
        city_enc = city.replace(" ", "+")
        resp = _req.get(f"https://wttr.in/{city_enc}?format=j1", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        current      = data.get("current_condition", [{}])[0]
        temp         = int(current.get("temp_C", 20))
        feels_like   = int(current.get("FeelsLikeC", temp))
        condition_en = current.get("weatherDesc", [{}])[0].get("value", "Clear")

        if any(k in condition_en for k in ["Rain", "Drizzle", "Shower"]): condition_ko = "비"
        elif any(k in condition_en for k in ["Snow", "Blizzard"]):        condition_ko = "눈"
        elif any(k in condition_en for k in ["Cloud", "Overcast", "Mist", "Fog"]): condition_ko = "흐림"
        else: condition_ko = "맑음"

        if temp >= 28:   outfit_season = "매우 더움"
        elif temp >= 23: outfit_season = "더움"
        elif temp >= 17: outfit_season = "따뜻함"
        elif temp >= 12: outfit_season = "선선함"
        elif temp >= 5:  outfit_season = "쌀쌀함"
        else:            outfit_season = "매우 추움"

        clothing_items = _get_wardrobe_items_sync()

        return {
            "success": True,
            "city": city, "temp": temp, "feels_like": feels_like,
            "condition": condition_ko, "outfit_season": outfit_season,
            "wardrobe_count": len(clothing_items),
            "wardrobe_items": clothing_items[:10],
        }
    except Exception as e:
        logger.error("옷 추천 데이터 수집 오류: %s", e)
        return {"success": False, "error": str(e)}


def _get_wardrobe_items_sync() -> list[dict]:
    # 1. SQLite DB에서 먼저 조회
    try:
        from app.database import SessionLocal
        import importlib
        ClothingItem = None
        for module_path in ("app.models.clothing", "app.models.wardrobe"):
            try:
                mod = importlib.import_module(module_path)
                ClothingItem = getattr(mod, "ClothingItem", None)
                if ClothingItem:
                    break
            except ImportError:
                continue
        if ClothingItem is not None:
            db = SessionLocal()
            try:
                items = db.query(ClothingItem).limit(30).all()
                if items:
                    return [{"name": getattr(i, "name", ""), "category": getattr(i, "category", ""),
                             "color": getattr(i, "color", ""), "tags": getattr(i, "tags", "")} for i in items]
            finally:
                db.close()
    except Exception as e:
        logger.warning("옷장 DB 조회 실패: %s", e)

    # 2. SQLite가 비어있으면 wardrobe_db.json 파일에서 읽기 (Streamlit 앱과 공유)
    try:
        import json as _json
        for json_path in (
            "wardrobe_db.json",
            os.path.join(os.path.dirname(__file__), "..", "wardrobe_db.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "wardrobe_db.json"),
            os.path.join(os.getcwd(), "wardrobe_db.json"),
            os.path.join(os.getcwd(), "app", "wardrobe_db.json"),
        ):
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                if data:
                    return [{"name": i.get("name", ""), "category": i.get("category", ""),
                             "color": i.get("color", ""), "tags": i.get("tags", "")} for i in data[:30]]
    except Exception as e:
        logger.warning("wardrobe_db.json 조회 실패: %s", e)

    return []


async def _exec_get_route(args: dict) -> dict:
    start        = args.get("start", "")
    end          = args.get("end", "")
    arrival_time = args.get("arrival_time", "09:00")
    transport    = args.get("transport", "car")

    if not start or not end:
        return {"success": False, "error": "출발지와 목적지를 모두 입력해 주세요."}

    route_server = os.getenv("ROUTE_SERVER_URL", "http://localhost:3000")
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            response = await c.post(
                f"{route_server}/calculate",
                json={"start": start, "end": end, "time": arrival_time, "transport": transport},
            )
            response.raise_for_status()
            data = response.json()

        if data.get("error"):
            return {"success": False, "error": data["error"]}

        return {
            "success": True,
            "transport": data.get("transportLabel", transport),
            "travel_time_text": data.get("travelTimeText", ""),
            "travel_time_min": data.get("travelTime"),
            "distance_km": data.get("distanceKm"),
            "departure_time": data.get("departure"),
            "arrival_time": arrival_time,
            "main_message": data.get("mainMessage", ""),
            "detail_message": data.get("detailMessage", ""),
            "start_place": data.get("startPlaceName", start),
            "end_place": data.get("endPlaceName", end),
        }
    except httpx.ConnectError:
        return {"success": False, "error": "경로 서버에 연결할 수 없습니다. backend-route 서버가 실행 중인지 확인해 주세요."}
    except Exception as e:
        return {"success": False, "error": f"경로 계산 오류: {e}"}


# ─────────────────────────────────────────────
# 메인 함수: 인텐트 기반 챗 처리
# ─────────────────────────────────────────────
async def chat_with_intent(
    message: str,
    db: Session,
    history: list | None = None,
) -> str:
    today_str = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    contents = list(history or [])
    contents.append({"role": "user", "parts": [{"text": f"[현재 시각: {today_str}]\n{message}"}]})

    # 일정/옷/경로 관련 키워드 감지 → 함수 강제 호출 모드
    schedule_keywords = ["일정", "등록", "추가", "넣어", "만들어", "삭제", "지워", "언제", "뭐 있"]
    outfit_keywords = ["코디", "옷", "입어", "추천"]
    route_keywords = ["경로", "가려면", "출발", "어떻게 가"]

    all_keywords = schedule_keywords + outfit_keywords + route_keywords
    force_tool = any(kw in message for kw in all_keywords)

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        tools=_TOOLS,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY" if force_tool else "AUTO"
            )
        ),
    )

    # ── 1단계: Gemini 호출 (429 시 자동 재시도) ──
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < 2:
                wait_sec = 60
                logger.warning("429 quota 초과, %d초 후 재시도 (%d/3)...", wait_sec, attempt + 1)
                await asyncio.sleep(wait_sec)
            else:
                raise RuntimeError(f"Gemini API 오류: {e}")

    # ── 2단계: Function Call 처리 (최대 3회) ──
    print(f"[DEBUG] Gemini 응답 parts: {[str(p)[:100] for p in response.candidates[0].content.parts]}")
    for _ in range(3):
        fn_call = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                fn_call = part.function_call
                break

        if fn_call is None:
            break

        fn_name = fn_call.name
        fn_args = dict(fn_call.args) if fn_call.args else {}
        logger.info("Function Call: %s(%s)", fn_name, fn_args)

        # 함수 실행
        print(f"[DEBUG] 함수 호출: {fn_name}, args: {fn_args}")
        try:
            if fn_name == "get_schedules":
                fn_result = _exec_get_schedules(fn_args, db)
            elif fn_name == "create_schedule":
                fn_result = _exec_create_schedule(fn_args, db)
            elif fn_name == "delete_schedule":
                fn_result = _exec_delete_schedule(fn_args, db)
            elif fn_name == "get_outfit_recommendation":
                fn_result = _exec_get_outfit_recommendation(fn_args)
            elif fn_name == "get_route":
                fn_result = await _exec_get_route(fn_args)
            else:
                fn_result = {"error": f"알 수 없는 함수: {fn_name}"}
        except Exception as e:
            fn_result = {"error": str(e)}
        print(f"[DEBUG] 함수 결과: {fn_result}")

        # 함수 결과를 대화에 추가 후 재호출
        contents.append({"role": "model", "parts": [{"function_call": {"name": fn_name, "args": fn_args}}]})
        contents.append({
            "role": "user",
            "parts": [{"function_response": {"name": fn_name, "response": {"result": json.dumps(fn_result, ensure_ascii=False)}}}]
        })

        # 함수 결과 받은 후엔 AUTO 모드로 텍스트 응답만 받기
        followup_config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            tools=_TOOLS,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            ),
        )
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=followup_config,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < 2:
                    await asyncio.sleep(60)
                else:
                    raise RuntimeError(f"Gemini 함수 응답 오류: {e}")

    # ── 3단계: 최종 텍스트 응답 ──
    try:
        return response.text
    except Exception:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text
        return "죄송해요, 응답을 생성하지 못했어요. 다시 시도해 주세요."
