import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

from dateutil import parser as dateparser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = Path("token.json")
CREDENTIALS_FILE = Path("credentials.json")
TIMEZONE = "Asia/Seoul"

def get_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError("credentials.json 파일이 없습니다.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=creds)

def normalize_text(text: str) -> str:
    return text.strip().replace("오전", "AM ").replace("오후", "PM ")

def parse_time(text: str):
    text = normalize_text(text)

    m = re.search(r"(AM|PM)\s*(\d{1,2})(?:시|:)(\d{2})?", text, re.IGNORECASE)
    if m:
        ampm = m.group(1).upper()
        hour = int(m.group(2))
        minute = int(m.group(3) or 0)

        if ampm == "PM" and hour != 12:
            hour += 12
        if ampm == "AM" and hour == 12:
            hour = 0

        return hour, minute

    m = re.search(r"(\d{1,2})[:시](\d{2})?", text)
    if m:
        return int(m.group(1)), int(m.group(2) or 0)

    if "아침" in text:
        return 8, 0
    if "점심" in text:
        return 13, 0
    if "저녁" in text:
        return 20, 0

    return 9, 0

def parse_date(text: str, now: datetime):
    if "오늘" in text:
        return now.date()
    if "내일" in text:
        return (now + timedelta(days=1)).date()
    if "모레" in text:
        return (now + timedelta(days=2)).date()

    m = re.search(r"(\d+)일\s*뒤", text)
    if m:
        return (now + timedelta(days=int(m.group(1)))).date()

    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()

    m = re.search(r"(\d{1,2})/(\d{1,2})", text)
    if m:
        return datetime(now.year, int(m.group(1)), int(m.group(2))).date()

    m = re.search(r"(\d{1,2})월\s*(\d{1,2})일", text)
    if m:
        return datetime(now.year, int(m.group(1)), int(m.group(2))).date()

    return now.date()

def parse_title(text: str):
    # 따옴표 안 텍스트가 있으면 우선 사용
    m = re.search(r'"([^"]+)"', text)
    if m:
        return m.group(1).strip()

    cleaned = text

    # 날짜 표현 제거
    cleaned = re.sub(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}월\s*\d{1,2}일", " ", cleaned)
    cleaned = re.sub(r"\d{4}-\d{1,2}-\d{1,2}", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}/\d{1,2}", " ", cleaned)
    cleaned = re.sub(r"(오늘|내일|모레|\d+일\s*뒤)", " ", cleaned)

    # 시간 표현 제거
    cleaned = re.sub(r"(오전|오후)\s*\d{1,2}시(?:\s*\d{1,2}분)?", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}시(?:\s*\d{1,2}분)?", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}:\d{2}", " ", cleaned)

    # 조사/명령어 제거
    cleaned = re.sub(
        r"(에서|부터|까지|에|등록해줘|추가해줘|넣어줘|만들어줘|삭제해줘|지워줘|없애줘|일정|캘린더)",
        " ",
        cleaned,
    )

    # 공백 정리
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned if cleaned else "새 일정"

def parse_recurrence(text: str):
    if "매일" in text:
        return ["RRULE:FREQ=DAILY"]
    if "매주" in text:
        return ["RRULE:FREQ=WEEKLY"]
    return None

def parse_event_from_text(text: str, now: datetime):
    date_value = parse_date(text, now)
    hour, minute = parse_time(text)
    title = parse_title(text)
    recurrence = parse_recurrence(text)

    start_dt = datetime(date_value.year, date_value.month, date_value.day, hour, minute)
    duration_minutes = 10 if "약" in text else 60
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    return {
        "summary": title,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": TIMEZONE,
        },
        "recurrence": recurrence,
    }

def create_event(service, calendar_id: str, event_body: dict):
    clean_body = {k: v for k, v in event_body.items() if v is not None}
    return service.events().insert(calendarId=calendar_id, body=clean_body).execute()

def brief_events(service, calendar_id: str, days: int = 1):
    now = datetime.now()
    time_min = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "+09:00"
    time_max = (now + timedelta(days=days)).replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + "+09:00"

    items = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
    ).execute().get("items", [])

    if not items:
        print("해당 기간 일정이 없습니다.")
        return

    print(f"[일정 브리핑] 앞으로 {days}일")
    for item in items:
        start = item["start"].get("dateTime", item["start"].get("date"))
        dt = dateparser.parse(start)
        print(f"- {dt.strftime('%Y-%m-%d %H:%M')} | {item.get('summary', '(제목 없음)')}")
def has_explicit_date(text: str):
    return bool(
        re.search(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일", text)
        or re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
        or re.search(r"\d{1,2}/\d{1,2}", text)
        or re.search(r"\d{1,2}월\s*\d{1,2}일", text)
        or re.search(r"(오늘|내일|모레|\d+일\s*뒤)", text)
    )

def has_explicit_time(text: str):
    return bool(
        re.search(r"(오전|오후)\s*\d{1,2}시(?:\s*\d{1,2}분)?", text)
        or re.search(r"\d{1,2}:\d{2}", text)
        or re.search(r"\d{1,2}시(?:\s*\d{1,2}분)?", text)
    )
def delete_events_by_title(service, calendar_id: str, title: str, original_text: str):
    now = datetime.now()
    time_min = (now - timedelta(days=3650)).isoformat() + "+09:00"
    time_max = (now + timedelta(days=3650)).isoformat() + "+09:00"

    items = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        q=title,
    ).execute().get("items", [])

    explicit_date = has_explicit_date(original_text)
    explicit_time = has_explicit_time(original_text)

    target_date = parse_date(original_text, now) if explicit_date else None
    target_hour, target_minute = None, None
    if explicit_time:
        target_hour, target_minute = parse_time(original_text)

    def matches_datetime(item):
        start = item["start"].get("dateTime", item["start"].get("date"))
        dt = dateparser.parse(start)

        if explicit_date and dt.date() != target_date:
            return False

        if explicit_time and (dt.hour != target_hour or dt.minute != target_minute):
            return False

        return True

    exact_matched = []
    contains_matched = []

    for item in items:
        summary = (item.get("summary") or "").strip()

        if summary == title:
            exact_matched.append(item)
        elif title in summary:
            contains_matched.append(item)

    # 1순위: 완전일치 + 날짜/시간 일치
    matched = [item for item in exact_matched if matches_datetime(item)]

    # 2순위: 완전일치
    if not matched:
        matched = exact_matched

    # 3순위: 포함일치 + 날짜/시간 일치
    if not matched:
        matched = [item for item in contains_matched if matches_datetime(item)]

    # 4순위: 포함일치
    if not matched:
        matched = contains_matched

    if not matched:
        print(f"'{title}' 와(과) 일치하는 일정이 없습니다.")
        return

    deleted_count = 0
    for item in matched:
        service.events().delete(calendarId=calendar_id, eventId=item["id"]).execute()
        deleted_count += 1

    print(f"삭제 완료: {deleted_count}건")
    for item in matched:
        start = item["start"].get("dateTime", item["start"].get("date"))
        print(f"- {item.get('summary')} | {start}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["add", "brief", "delete"])
    parser.add_argument("--text")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--calendar", default="primary")
    args = parser.parse_args()

    service = get_service()

    if args.mode == "add":
        if not args.text:
            raise ValueError("--text 값이 필요합니다.")
        event_body = parse_event_from_text(args.text, datetime.now())
        event = create_event(service, args.calendar, event_body)
        print("일정이 등록되었습니다.")
        print(f"- 제목: {event.get('summary')}")
        print(f"- 시작: {event['start'].get('dateTime', event['start'].get('date'))}")
        print(f"- 링크: {event.get('htmlLink')}")
    elif args.mode == "brief":
        brief_events(service, args.calendar, args.days)
    elif args.mode == "delete":
        if not args.text:
            raise ValueError("--text 값이 필요합니다.")
        title = parse_title(args.text)
        delete_events_by_title(service, args.calendar, title, args.text)
if __name__ == "__main__":
    main()