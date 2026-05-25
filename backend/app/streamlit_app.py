# Streamlit 기반 메인 진입점. UI 렌더링과 상태 관리만 담당,
# 실제 AI 로직은 모두 skills/ 하위 모듈에서 처리
import streamlit as st
import pandas as pd
import json
import os
import logging

from skills.shopping_parser import parse_shopping_screenshot, parse_shopping_history
from skills.weather_fashion import (
    recommend_fashion_for_weather,
    hybrid_search_clothes,
    get_image_embedding_vector,
    get_dual_embedding,
    get_clip_text_embedding,
    build_item_text_for_embedding,
    build_daily_outfit_notification,
)
from skills.preprocess import prepare_images_for_embedding

# --- 앱 설정 ---
st.set_page_config(page_title="AI 개인 패션 비서", page_icon="👗", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 알림창 커스텀 스타일 (민트 계열로 브랜드 무드 통일) ---
st.markdown("""
<style>
div[data-testid="stAlert"] {
    background-color: #e0f7fa;
    color: #006064;
    border-color: #b2ebf2;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 옷장 DB: 로컬 JSON 파일로 가볍게 관리
# 스키마는 backend/app/models/clothing.py의 ClothingItem과 1:1 매핑
#   - id (자동 부여), name, category, color, tags, image_url
#   - fit_info / embedding은 fashion-backend 고유 확장 필드
# 나중에 SQLAlchemy로 갈아끼울 때 ClothingItem ORM과 그대로 호환됩니다.
# ==========================================
WARDROBE_DB_PATH = "wardrobe_db.json"

# backend ClothingItem 컬럼 + Streamlit 확장 필드 + 이미지 유사도 고도화 필드
# embedding_meta: 전처리 정보 (bbox, 단일/멀티, 배경 제거 여부 등) — PDF1·PDF2 인사이트
# clip_embedding: CLIP 보조 임베딩 (앙상블용, 선택적)
_WARDROBE_SCHEMA_FIELDS = (
    "id", "name", "category", "color", "tags", "image_url",
    "fit_info", "embedding", "embedding_meta", "clip_embedding",
)


def _migrate_legacy_item(raw_item: dict, fallback_id: int) -> dict:
    """
    구 스키마 → 신 스키마 마이그레이션. 두 단계:
      1) item_name → name (backend ClothingItem 정렬)
      2) embedding_meta / clip_embedding 누락 시 기본값 부여 (이미지 유사도 고도화)

    기존 데이터의 임베딩은 그대로 보존됩니다 — 다만 embedding_meta가 비어 있어서
    "전처리 안 된 임베딩"으로 표시됩니다. UI에서 재임베딩 트리거 가능.
    """
    migrated: dict = {}

    # id: 없으면 인덱스 기반 자동 부여 (기존 데이터엔 id가 없었음)
    migrated["id"] = raw_item.get("id", fallback_id)

    # name: 신 스키마 우선, 없으면 구 스키마 item_name fallback
    migrated["name"] = raw_item.get("name") or raw_item.get("item_name") or "이름 없음"

    # 그 외 backend 컬럼들 (없으면 기본값)
    migrated["category"] = raw_item.get("category", "")
    migrated["color"] = raw_item.get("color", "")
    migrated["tags"] = raw_item.get("tags", "")
    migrated["image_url"] = raw_item.get("image_url", "")

    # fashion-backend 고유 확장 필드 보존
    migrated["fit_info"] = raw_item.get("fit_info", "")
    migrated["embedding"] = raw_item.get("embedding")

    # 이미지 유사도 고도화 필드 (PDF1·PDF2 인사이트)
    # 기존 데이터엔 없으니 마이그레이션 시 기본값 — UI에서 재임베딩하면 채워짐
    migrated["embedding_meta"] = raw_item.get("embedding_meta") or {
        "preprocessed": False,
        "source": "legacy_unprocessed",
    }
    migrated["clip_embedding"] = raw_item.get("clip_embedding")

    return migrated


def load_wardrobe() -> list[dict]:
    """저장된 옷장 데이터를 불러옵니다. 파일이 없거나 손상됐을 때도 빈 리스트로 안전하게 시작합니다.

    구 스키마 (item_name) 데이터가 섞여 있어도 자동으로 신 스키마 (name)로 정규화합니다.
    """
    if not os.path.exists(WARDROBE_DB_PATH):
        return []
    try:
        with open(WARDROBE_DB_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # DB 파일이 깨진 경우 — 앱을 멈추는 것보다 빈 상태로 시작하는 게 낫다고 판단
        logger.warning("옷장 DB 파일을 읽는 데 실패했습니다. 새 DB로 시작합니다. 원인: %s", e)
        return []

    # 모든 아이템을 backend ClothingItem 호환 스키마로 정규화
    return [_migrate_legacy_item(item, idx + 1) for idx, item in enumerate(raw_data)]


def save_wardrobe(outfit_data: list[dict]) -> None:
    """옷장 데이터를 저장합니다. 쓰기 도중 오류가 생겨도 기존 데이터를 보호합니다.

    스키마 정렬을 위해 알려진 필드만 골라서 저장합니다 (UI에서 들어온 noise 컬럼 차단).
    """
    try:
        # 알려진 컬럼만 통과시켜 깨끗하게 저장
        cleaned = [
            {k: item.get(k) for k in _WARDROBE_SCHEMA_FIELDS if k in item}
            for item in outfit_data
        ]
        with open(WARDROBE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error("옷장 DB 저장에 실패했습니다: %s", e)
        st.error("DB 저장 중 오류가 발생했습니다. 디스크 공간을 확인해 주세요.")


def delete_wardrobe_item(item_id: int) -> tuple[bool, str]:
    """
    backend wardrobe.py의 DELETE /{item_id} 엔드포인트와 동일한 의미를 로컬에서 수행합니다.

    Returns:
        (성공 여부, 메시지)
    """
    current = load_wardrobe()
    target = next((it for it in current if it.get("id") == item_id), None)
    if not target:
        return False, f"id={item_id} 아이템을 찾을 수 없습니다."

    remaining = [it for it in current if it.get("id") != item_id]
    save_wardrobe(remaining)
    return True, f"'{target.get('name', '이름 없음')}'이(가) 삭제되었습니다."


def next_wardrobe_id(existing: list[dict]) -> int:
    """다음 아이템 id를 발급합니다 (간단한 max+1 방식)."""
    if not existing:
        return 1
    return max((it.get("id", 0) or 0) for it in existing) + 1


# --- 세션 초기화: 탭 이동 시 분석 결과가 날아가지 않도록 ---
if "parsed_outfit_items" not in st.session_state:
    st.session_state["parsed_outfit_items"] = []

st.title("✨ AI 개인 패션 비서")
st.write("쇼핑몰 구매 내역을 등록하거나, 내 옷장 코디를 추천받아 보세요.")

tab_register, tab_wardrobe, tab_weather, tab_snap = st.tabs([
    "새로운 옷 등록",
    "내 옷장 관리 (DB)",
    "오늘 날씨 맞춤 추천",
    "코디 손민수",
])


# ==========================================
# 탭 1: 새로운 옷 등록
# ==========================================
with tab_register:
    st.subheader("구매 내역 업로드")
    sub_screenshot, sub_text = st.tabs(["스크린샷 올리기 (VLM)", "텍스트 붙여넣기"])

    # --- 방식 A: 이미지 업로드 ---
    with sub_screenshot:
        uploaded_files = st.file_uploader(
            "구매 내역 스크린샷 (png, jpg, jpeg)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        col_opt_multi, col_opt_clip = st.columns(2)
        with col_opt_multi:
            multi_item_mode = st.checkbox(
                "한 사진에 여러 옷이 있으면 자동 분리",
                value=True,
                help="PDF1 발견: 한 사진에 여러 옷이 있을 때 모두 동일 임베딩이 되는 버그를 해결합니다. "
                     "Gemini Vision으로 의류별 bounding box를 추출해 각각 임베딩합니다.",
            )
        with col_opt_clip:
            ensemble_clip_register = st.checkbox(
                "CLIP 보조 임베딩 같이 저장 (앙상블용)",
                value=True,
                help="CLIP 패키지가 설치돼 있으면 보조 벡터를 같이 저장합니다. "
                     "손민수 검색 시 Gemini와 가중 평균해 정확도가 올라갑니다.",
            )

        if st.button("이미지 분석 및 등록"):
            if not uploaded_files:
                st.warning("이미지를 먼저 첨부해 주세요.")
            else:
                with st.spinner("사진을 분석 중입니다..."):
                    collected_items: list[dict] = []
                    success_count = 0
                    fail_log: list[str] = []

                    for file in uploaded_files:
                        # Step 1: VLM으로 텍스트/카테고리/핏 추출
                        parse_result = parse_shopping_screenshot(file)

                        if parse_result["status"] != "success":
                            fail_log.append(
                                f"[{file.name}] 분석 실패 — {parse_result.get('message', '원인 불명')}"
                            )
                            continue

                        # Step 2: 원본 이미지 바이트 확보 (parse_screenshot이 read()를 이미 했으므로 seek(0))
                        file.seek(0)
                        image_bytes = file.read()

                        # Step 3 (신규): 의류 영역 추출 + 멀티 의류 분리
                        # PDF2 인사이트(배경 노이즈) + PDF1 인사이트(멀티 의류 동일 임베딩 버그)를 동시에 해결
                        prepared_images = prepare_images_for_embedding(
                            image_bytes, multi_item_mode=multi_item_mode
                        )

                        outfit_items = parse_result["data"].get("purchased_items", [])
                        if not outfit_items:
                            fail_log.append(f"[{file.name}] 분석은 됐지만 아이템을 찾지 못했어요.")
                            continue

                        # Step 4: 검출된 의류 영역 수와 VLM이 뽑은 아이템 수를 매칭
                        # - 검출 1개 / VLM 1개: 1:1
                        # - 검출 N개 / VLM 1개: VLM 정보를 N개 모두에 복사 (이름은 같음, 임베딩은 영역별)
                        # - 검출 1개 / VLM N개: 모두 같은 임베딩 (구버전과 동일 동작)
                        # - 검출 0개: 원본 이미지로 폴백 임베딩
                        if len(prepared_images) > 1 and len(outfit_items) == 1:
                            # 한 스크린샷에 여러 의류 검출 → 각각 별개 아이템으로 분기
                            base_item = outfit_items[0]
                            outfit_items = []
                            for prep in prepared_images:
                                detected_cat = prep["meta"].get("detected_category")
                                detected_color = prep["meta"].get("detected_color")
                                outfit_items.append({
                                    **base_item,
                                    # Gemini가 검출한 카테고리/색이 더 정확하면 덮어씌움
                                    "category": detected_cat or base_item.get("category", ""),
                                    "color": detected_color or base_item.get("color", ""),
                                })

                        # Step 5: 각 아이템에 임베딩 + 잘린 이미지 썸네일 부여
                        for idx, item in enumerate(outfit_items):
                            # 영역 매칭 — 검출 영역이 부족하면 첫 번째 영역 재사용
                            prep = prepared_images[min(idx, len(prepared_images) - 1)]
                            embed_input = prep["image_bytes"]

                            if ensemble_clip_register:
                                dual = get_dual_embedding(embed_input)
                                item["embedding"] = dual["embedding"]
                                item["clip_embedding"] = dual["clip_embedding"]
                            else:
                                item["embedding"] = get_image_embedding_vector(embed_input)
                                item["clip_embedding"] = None

                            item["embedding_meta"] = prep["meta"]

                            # 옷장 시각화: 잘린 영역 썸네일을 image_url에 data URL로 저장
                            # 사용자가 수동으로 image_url을 채워뒀으면 그대로 유지
                            thumb_url = prep["meta"].get("thumbnail_data_url")
                            if thumb_url and not item.get("image_url"):
                                item["image_url"] = thumb_url

                        collected_items.extend(outfit_items)
                        success_count += 1

                    # 결과 리포트
                    if success_count > 0:
                        st.session_state["parsed_outfit_items"] = collected_items
                        # 멀티 의류로 분리된 케이스가 있으면 사용자에게 알림
                        split_note = ""
                        split_cnt = sum(
                            1 for it in collected_items
                            if it.get("embedding_meta", {}).get("source") == "cropped"
                        )
                        if split_cnt > 0:
                            split_note = f" (그 중 {split_cnt}개 의류는 자동 영역 추출됨)"
                        st.success(
                            f"총 {len(uploaded_files)}장 중 {success_count}장 분석 완료, "
                            f"{len(collected_items)}개 의류 검출{split_note}! 아래에서 수정 후 저장하세요."
                        )
                    if fail_log:
                        for msg in fail_log:
                            st.error(msg)

    # --- 방식 B: 텍스트 붙여넣기 ---
    with sub_text:
        purchase_text = st.text_area("구매 내역 텍스트 붙여넣기:", height=150)
        if st.button("텍스트 분석 및 등록"):
            if not purchase_text.strip():
                st.warning("분석할 텍스트를 먼저 입력해 주세요.")
            else:
                with st.spinner("AI가 텍스트를 분석 중입니다..."):
                    parse_result = parse_shopping_history(purchase_text)
                    if parse_result["status"] == "success":
                        st.session_state["parsed_outfit_items"] = parse_result["data"].get(
                            "purchased_items", []
                        )
                        st.success("텍스트 분석 완료! 아래에서 수정 후 저장하세요.")
                    else:
                        st.error(f"분석 실패: {parse_result.get('message', '원인 불명')}")

    # --- 분석된 아이템 확인 및 저장 ---
    st.divider()
    st.subheader("내 옷장 등록")

    if st.session_state["parsed_outfit_items"]:
        # --- 자동 분리 결과 시각화 (PDF1 인사이트의 시연용) ---
        # Gemini Vision이 의류별로 잘라낸 영역을 사용자가 직접 눈으로 확인하는 단계.
        # "분류가 된 모습들도 보여줬으면 좋겠다"는 시연 요구사항.
        items_with_thumb = [
            it for it in st.session_state["parsed_outfit_items"]
            if it.get("image_url", "").startswith("data:image/")
        ]
        if items_with_thumb:
            st.markdown("##### 🔍 자동 분리된 의류 영역")
            st.caption(
                "Gemini Vision이 업로드한 사진에서 의류 객체별로 bounding box를 검출하고, "
                "그 영역만 잘라서 각각 임베딩한 결과입니다. 분리가 어색하면 아래 표에서 카테고리/색상을 직접 수정할 수 있어요."
            )
            preview_cols = st.columns(min(len(items_with_thumb), 4))
            for idx, item in enumerate(items_with_thumb):
                col = preview_cols[idx % len(preview_cols)]
                with col:
                    st.image(item["image_url"], use_container_width=True)
                    src = (item.get("embedding_meta") or {}).get("source", "")
                    badge = "✂️ crop" if src == "cropped" else "🖼️ 원본"
                    st.caption(
                        f"{badge} · **{item.get('category', '?')}** · {item.get('color', '?')}"
                    )

        outfit_df = pd.DataFrame(st.session_state["parsed_outfit_items"])
        edited_outfit_df = st.data_editor(
            outfit_df, use_container_width=True, num_rows="dynamic"
        )

        if st.button("내 옷장에 등록하기", type="primary"):
            current_wardrobe = load_wardrobe()
            new_items = edited_outfit_df.to_dict(orient="records")

            # backend ClothingItem과 호환되도록 id 자동 발급 (없는 아이템에만)
            issued_id = next_wardrobe_id(current_wardrobe)
            for item in new_items:
                if not item.get("id"):
                    item["id"] = issued_id
                    issued_id += 1

            current_wardrobe.extend(new_items)
            save_wardrobe(current_wardrobe)

            st.balloons()
            st.success("성공적으로 옷장에 저장되었습니다! [내 옷장 관리] 탭에서 확인해 보세요.")
            st.session_state["parsed_outfit_items"] = []
    else:
        st.info("위에서 구매 내역을 먼저 분석해 주세요.")


# ==========================================
# 탭 2: 내 옷장 관리
# (backend wardrobe.py의 GET / POST / DELETE 엔드포인트와 동등한 동작 제공)
# ==========================================
with tab_wardrobe:
    st.subheader("나의 옷장 데이터베이스")
    wardrobe_data = load_wardrobe()

    if not wardrobe_data:
        st.warning("아직 옷장에 등록된 옷이 없습니다.")
    else:
        st.write(f"현재 총 **{len(wardrobe_data)}벌**의 옷이 등록되어 있습니다.")

        # 임베딩/CLIP 같은 무거운 컬럼은 표에서 숨기되, 저장 시엔 원본 유지하기 위해 분리
        # 대신 "전처리 상태" 컬럼은 표시 — 사용자가 어떤 아이템이 재임베딩 필요한지 한눈에 보게
        def _meta_summary(meta: dict) -> str:
            if not meta or not isinstance(meta, dict):
                return "-"
            if meta.get("source") == "legacy_unprocessed":
                return "구버전 (재임베딩 권장)"
            if meta.get("source") == "cropped":
                cat = meta.get("detected_category", "")
                return f"영역 추출됨 ({cat})" if cat else "영역 추출됨"
            if meta.get("source") in ("original_fallback", "all_crops_failed_fallback", "crop_failed_fallback"):
                return "원본 사용"
            return meta.get("source", "-")

        display_df = pd.DataFrame([
            {
                **{k: v for k, v in item.items() if k not in ("embedding", "clip_embedding", "embedding_meta")},
                "전처리": _meta_summary(item.get("embedding_meta", {})),
                "CLIP": "✓" if item.get("clip_embedding") else "",
            }
            for item in wardrobe_data
        ])

        # image_url에 data:image/...; 형태의 썸네일이 들어있는 아이템들을 위해 ImageColumn 적용
        # data URL이 아닌 일반 http URL이어도 streamlit이 자동으로 잘 처리합니다.
        edited_wardrobe_df = st.data_editor(
            display_df, key="wardrobe_editor", use_container_width=True, num_rows="dynamic",
            column_order=("id", "image_url", "name", "category", "color", "tags", "fit_info", "전처리", "CLIP"),
            column_config={
                "image_url": st.column_config.ImageColumn(
                    "썸네일",
                    help="자동 분리된 의류 영역 (data URL) 또는 외부 이미지 URL",
                    width="small",
                ),
            },
            disabled=("전처리", "CLIP", "image_url"),  # 썸네일은 직접 편집 불가
        )

        col_save, col_delete = st.columns([2, 3])

        with col_save:
            if st.button("변경사항 DB에 업데이트"):
                # data_editor에서 임베딩 / CLIP / 메타가 빠졌으니, id 기준으로 원본을 다시 머지
                lookup = {item.get("id"): item for item in wardrobe_data}
                merged = edited_wardrobe_df.to_dict(orient="records")
                for row in merged:
                    original = lookup.get(row.get("id"), {})
                    row["embedding"] = original.get("embedding")
                    row["clip_embedding"] = original.get("clip_embedding")
                    row["embedding_meta"] = original.get("embedding_meta")
                    # 디스플레이용 컬럼 제거
                    row.pop("전처리", None)
                    row.pop("CLIP", None)
                save_wardrobe(merged)
                st.success("옷장 데이터가 업데이트되었습니다!")

        # backend wardrobe.py의 DELETE /{item_id} 엔드포인트와 동등한 UI
        with col_delete:
            with st.expander("🗑️ 특정 아이템 삭제 (id 지정)"):
                deletable_ids = [item.get("id") for item in wardrobe_data if item.get("id") is not None]
                if not deletable_ids:
                    st.caption("삭제할 수 있는 id가 없습니다.")
                else:
                    target_id = st.selectbox("삭제할 아이템 id", deletable_ids, key="delete_target_id")
                    if st.button("선택 아이템 삭제", type="secondary"):
                        ok, msg = delete_wardrobe_item(int(target_id))
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        # --- 매일 오전 8시 알림 미리보기 ---
        # Firebase Cloud Functions가 매일 호출할 함수를 Streamlit에서 미리 확인.
        # 이 본문 그대로 FCM 푸시로 사용자한테 보내질 예정.
        with st.expander("🔔 매일 오전 8시 푸시 알림 미리보기 (Firebase FCM 연동 대비)"):
            st.caption(
                "Cloud Functions이 매일 오전 8시에 호출할 build_daily_outfit_notification()의 결과를 "
                "여기서 미리 확인할 수 있어요. 실제 푸시 본문은 이 텍스트 그대로 발송됩니다."
            )
            preview_region = st.text_input(
                "지역 (시/군/구)", value="서울시", key="notif_preview_region"
            )
            preview_gender = st.radio(
                "성별", ["여성", "남성"], horizontal=True, key="notif_preview_gender"
            )
            if st.button("알림 본문 생성해보기", key="notif_preview_btn"):
                with st.spinner("날씨 조회 + 옷장 추천 계산 중..."):
                    notif = build_daily_outfit_notification(
                        region=preview_region,
                        wardrobe_data=wardrobe_data,
                        gender=preview_gender,
                    )
                # 모바일 푸시처럼 카드형으로 표시
                with st.container(border=True):
                    st.markdown(f"### {notif['title']}")
                    st.markdown(f"**{notif['body']}**")
                    weather_meta = notif.get("weather", {})
                    if weather_meta:
                        st.caption(
                            f"🌡️ {weather_meta.get('temp', '?')}°C · {weather_meta.get('condition', '?')}"
                        )
                if notif.get("items"):
                    st.caption("매칭된 옷장 아이템: " + ", ".join(
                        f"[{it['category']}] {it['name']}" for it in notif["items"]
                    ))

        # --- 개별 옷에 이미지 첨부 ---
        # 기존 옷장 아이템엔 image_url이 비어있어서 추천 결과에 썸네일이 안 떠.
        # 이 도구로 옷 하나씩 사진 첨부하면, Gemini Vision이 의류 영역 자동 추출 →
        # 썸네일을 image_url에 저장 + CLIP 임베딩 재계산까지 한 번에.
        with st.expander("🖼️ 기존 옷에 이미지 추가 (추천 결과에 썸네일 띄우기)"):
            st.caption(
                "기존에 등록된 옷들은 사진이 없어서 추천 카드에 텍스트만 나옵니다. "
                "옷 하나당 이미지 한 장씩 업로드하면 자동으로 의류 영역만 잘라서 옷장에 저장돼요."
            )

            attach_ids = [item.get("id") for item in wardrobe_data if item.get("id") is not None]
            if not attach_ids:
                st.caption("아직 등록된 옷이 없어요.")
            else:
                # id + 이름을 같이 보여주면 선택이 쉬워짐
                attach_options = {
                    f"#{it.get('id')} · {it.get('name', '이름 없음')}": it.get("id")
                    for it in wardrobe_data if it.get("id") is not None
                }
                attach_label = st.selectbox(
                    "이미지를 추가할 옷 선택",
                    list(attach_options.keys()),
                    key="attach_image_target",
                )
                target_attach_id = attach_options[attach_label]

                # 현재 이 옷에 이미지가 있나 미리보기
                target_item = next((it for it in wardrobe_data if it.get("id") == target_attach_id), None)
                if target_item and target_item.get("image_url", "").startswith("data:image/"):
                    st.caption("이미 이미지가 등록돼 있어요. 새로 올리면 덮어씌워집니다.")
                    st.image(target_item["image_url"], width=160)

                uploaded_attach = st.file_uploader(
                    "사진 업로드 (png, jpg, jpeg)",
                    type=["png", "jpg", "jpeg"],
                    key=f"attach_uploader_{target_attach_id}",
                )

                auto_crop = st.checkbox(
                    "Gemini Vision으로 의류 영역 자동 추출 (배경 노이즈 제거)",
                    value=True,
                    key="attach_auto_crop",
                )
                reembed_clip = st.checkbox(
                    "CLIP 임베딩도 같이 재계산 (시각 정확도 ↑)",
                    value=True,
                    key="attach_reembed_clip",
                )

                if st.button("이 옷에 이미지 첨부", type="primary", key="attach_submit"):
                    if not uploaded_attach:
                        st.warning("이미지를 먼저 업로드해 주세요.")
                    else:
                        with st.spinner("이미지 분석 중..."):
                            raw_bytes = uploaded_attach.read()

                            # 1. 의류 영역 추출
                            if auto_crop:
                                prepared = prepare_images_for_embedding(raw_bytes, multi_item_mode=False)
                                used = prepared[0] if prepared else {"image_bytes": raw_bytes, "meta": {}}
                            else:
                                from skills.preprocess import image_bytes_to_thumbnail_data_url
                                used = {
                                    "image_bytes": raw_bytes,
                                    "meta": {
                                        "preprocessed": False,
                                        "source": "manual_attach",
                                        "thumbnail_data_url": image_bytes_to_thumbnail_data_url(raw_bytes),
                                    },
                                }

                            new_thumb = used["meta"].get("thumbnail_data_url")
                            new_embedding = None
                            new_clip = None

                            # 2. 임베딩 재계산 (옵션)
                            if reembed_clip:
                                dual = get_dual_embedding(used["image_bytes"])
                                new_embedding = dual["embedding"]
                                new_clip = dual["clip_embedding"]

                            # 3. 옷장 DB 업데이트
                            updated = list(wardrobe_data)
                            for it in updated:
                                if it.get("id") == target_attach_id:
                                    if new_thumb:
                                        it["image_url"] = new_thumb
                                    if new_embedding is not None:
                                        it["embedding"] = new_embedding
                                    if new_clip is not None:
                                        it["clip_embedding"] = new_clip
                                    it["embedding_meta"] = used["meta"]
                                    break
                            save_wardrobe(updated)

                        st.success(
                            f"✅ #{target_attach_id}번 옷에 이미지 추가 완료! "
                            f"({'영역 자동 추출됨' if used['meta'].get('source') == 'cropped' else '원본 저장'}"
                            f"{', CLIP 재임베딩됨' if reembed_clip else ''})"
                        )
                        st.rerun()

        # --- 무신사 검색으로 이미지 일괄 자동 채우기 ---
        # 옛날 wardrobe_db.json의 64벌은 image_url이 비어있어서 추천 카드에 텍스트만 나옴.
        # 옷 이름으로 무신사를 검색해서 첫 상품 이미지를 자동으로 가져옴.
        # 2단계 워크플로: ① 검색 → ② 미리보기에서 잘못 매칭된 것만 체크 해제 후 저장.
        with st.expander("🛍️ 무신사에서 이미지 일괄 자동 채우기 (옛날 옷용)"):
            # image_url이 비어있거나 외부 URL이 아닌 아이템만 대상
            missing_image_items = [
                it for it in wardrobe_data
                if not (it.get("image_url") or "").startswith(("data:image/", "http"))
            ]
            st.caption(
                f"이미지 없는 옷: **{len(missing_image_items)}벌** / 전체 {len(wardrobe_data)}벌"
            )

            if not missing_image_items:
                st.success("모든 옷에 이미지가 등록돼 있어요!")
            else:
                col_n, col_btn = st.columns([1, 2])
                with col_n:
                    batch_size = st.number_input(
                        "한 번에 처리할 옷 개수",
                        min_value=1,
                        max_value=len(missing_image_items),
                        value=min(10, len(missing_image_items)),
                        step=5,
                        help="너무 많이 한 번에 돌리면 무신사가 막을 수 있어요. 10벌씩 권장.",
                    )

                with col_btn:
                    st.write("")  # 정렬용 빈 줄
                    if st.button(
                        f"🔍 무신사에서 {batch_size}벌 자동 검색",
                        key="musinsa_search_btn",
                        type="primary",
                    ):
                        from skills.musinsa_search import (
                            find_musinsa_product_image,
                            download_image_to_bytes,
                        )

                        candidates: list[dict] = []
                        progress = st.progress(0, text="무신사 검색 시작...")
                        targets = missing_image_items[:batch_size]

                        for idx, item in enumerate(targets):
                            item_name = item.get("name", "")
                            progress.progress(
                                (idx + 1) / len(targets),
                                text=f"[{idx + 1}/{len(targets)}] '{item_name}' 검색 중...",
                            )
                            result = find_musinsa_product_image(item_name)
                            if result and result.get("image_url"):
                                # 이미지를 미리 다운로드해서 base64 thumbnail로 변환 (미리보기용)
                                img_bytes = download_image_to_bytes(result["image_url"])
                                if img_bytes:
                                    candidates.append({
                                        "item_id": item.get("id"),
                                        "item_name": item_name,
                                        "item_category": item.get("category", ""),
                                        "external_url": result["image_url"],
                                        "image_bytes": img_bytes,
                                        "source": result["source"],
                                        "query": result["query"],
                                    })

                        progress.empty()
                        # 세션 스테이트에 후보 저장 — 다음 단계(미리보기/저장)에서 사용
                        st.session_state["musinsa_candidates"] = candidates
                        if not candidates:
                            st.warning(
                                "무신사에서 매칭된 상품이 없어요. "
                                "API가 막혔거나 검색어가 너무 구체적일 수 있어요. "
                                "위의 '🖼️ 기존 옷에 이미지 추가' 도구로 수동 첨부해 보세요."
                            )
                        else:
                            st.success(f"✅ {len(candidates)}/{len(targets)}벌 매칭 성공! 아래서 확인하세요.")

                # --- 검색 결과 미리보기 + 선택적 저장 ---
                candidates = st.session_state.get("musinsa_candidates", [])
                if candidates:
                    st.markdown("---")
                    st.markdown("**🖼️ 미리보기** — 잘못 매칭된 건 체크 해제하고 저장하세요")

                    # 후보를 한 행에 3개씩 그리드로 표시
                    checked_ids: list[int] = []
                    cols_per_row = 3
                    for row_start in range(0, len(candidates), cols_per_row):
                        row_candidates = candidates[row_start : row_start + cols_per_row]
                        cols = st.columns(cols_per_row)
                        for col, cand in zip(cols, row_candidates):
                            with col:
                                with st.container(border=True):
                                    st.image(cand["image_bytes"], use_container_width=True)
                                    st.caption(f"**#{cand['item_id']}** · {cand['item_category']}")
                                    st.caption(f"{cand['item_name'][:30]}{'…' if len(cand['item_name']) > 30 else ''}")
                                    checked = st.checkbox(
                                        "이대로 저장",
                                        value=True,
                                        key=f"musinsa_keep_{cand['item_id']}",
                                    )
                                    if checked:
                                        checked_ids.append(cand["item_id"])

                    st.markdown("---")
                    save_col1, save_col2 = st.columns([2, 1])
                    with save_col1:
                        st.info(
                            f"선택된 {len(checked_ids)}/{len(candidates)}벌을 저장합니다. "
                            f"저장 시 의류 영역 자동 추출 + CLIP 재임베딩이 같이 실행돼요."
                        )
                    with save_col2:
                        if st.button("✅ 선택된 것만 저장", type="primary", key="musinsa_commit"):
                            if not checked_ids:
                                st.warning("선택된 항목이 없어요.")
                            else:
                                from skills.preprocess import prepare_images_for_embedding
                                updated = list(wardrobe_data)
                                save_progress = st.progress(0, text="저장 중...")
                                success_count = 0
                                to_save = [c for c in candidates if c["item_id"] in checked_ids]
                                for idx, cand in enumerate(to_save):
                                    save_progress.progress(
                                        (idx + 1) / len(to_save),
                                        text=f"[{idx + 1}/{len(to_save)}] #{cand['item_id']} 처리 중...",
                                    )
                                    # 의류 영역 자동 추출 + 썸네일 + 임베딩 한방에
                                    prepared = prepare_images_for_embedding(
                                        cand["image_bytes"], multi_item_mode=False
                                    )
                                    used = prepared[0] if prepared else None
                                    if not used:
                                        continue
                                    thumb_url = used["meta"].get("thumbnail_data_url")
                                    dual = get_dual_embedding(used["image_bytes"])
                                    for it in updated:
                                        if it.get("id") == cand["item_id"]:
                                            if thumb_url:
                                                it["image_url"] = thumb_url
                                            if dual.get("embedding"):
                                                it["embedding"] = dual["embedding"]
                                            if dual.get("clip_embedding"):
                                                it["clip_embedding"] = dual["clip_embedding"]
                                            it["embedding_meta"] = used["meta"]
                                            it["embedding_meta"]["musinsa_source_url"] = cand["external_url"]
                                            success_count += 1
                                            break
                                save_progress.empty()
                                save_wardrobe(updated)
                                st.success(
                                    f"✅ {success_count}/{len(to_save)}벌 저장 완료! "
                                    "추천 다시 돌려보면 썸네일이 보일 거예요."
                                )
                                # 후보 클리어 + 새로고침
                                st.session_state.pop("musinsa_candidates", None)
                                st.rerun()

        # --- CLIP 일괄 재임베딩 ---
        # Gemini multimodal embedding이 무료 티어에서 막혀서 CLIP을 메인으로 채택.
        # 옷장의 기존 Gemini 임베딩(3072차원)은 CLIP(512차원)과 차원이 안 맞아서
        # 직접 비교가 불가능. 이 버튼이 모든 아이템을 CLIP 공간으로 재임베딩합니다.
        # 이미지가 있으면 CLIP 이미지 인코더, 없으면 텍스트 메타로 CLIP 텍스트 인코더.
        with st.expander("🔁 CLIP으로 일괄 재임베딩 (Gemini → CLIP 전환)"):
            st.caption(
                "Gemini Embedding이 무료 티어에서 막혀있어서, 옷장 임베딩을 CLIP으로 통일해야 "
                "무신사 매칭/손민수 검색이 정상 동작합니다. 이미지가 없는 아이템은 "
                "이름·카테고리·색상·핏 메타를 CLIP 텍스트 인코더로 변환해 같은 벡터 공간에 넣어요."
            )

            if st.button("전체 옷장 CLIP 재임베딩 시작", key="reembed_all_clip"):
                with st.spinner(f"{len(wardrobe_data)}벌 재임베딩 중... (CLIP)"):
                    success = 0
                    text_fallback = 0
                    failed = 0
                    updated = list(wardrobe_data)  # shallow copy

                    for idx, item in enumerate(updated):
                        # 이미지 URL이 있고 http로 시작하면 다운로드해서 이미지 임베딩
                        # (현재 옷장 데이터엔 image_url이 비어있어서 대부분 텍스트 경로)
                        embedded = False
                        img_url = item.get("image_url", "")
                        if img_url and img_url.startswith("http"):
                            try:
                                import requests as _req
                                resp = _req.get(img_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                                if resp.status_code == 200 and resp.content:
                                    new_vec = get_image_embedding_vector(resp.content)
                                    if new_vec is not None:
                                        item["embedding"] = new_vec
                                        item["clip_embedding"] = new_vec  # 어차피 CLIP이 메인
                                        item["embedding_meta"] = {
                                            "preprocessed": False,
                                            "source": "clip_image_reembed",
                                            "backbone": "CLIP-ViT-B/32",
                                        }
                                        success += 1
                                        embedded = True
                            except Exception as e:
                                logger.warning("재임베딩 이미지 다운로드 실패 (id=%s): %s", item.get("id"), e)

                        if embedded:
                            continue

                        # 텍스트 fallback — name + category + color + fit_info + tags
                        text_repr = build_item_text_for_embedding(item)
                        if not text_repr:
                            failed += 1
                            continue

                        text_vec = get_clip_text_embedding(text_repr)
                        if text_vec is not None:
                            item["embedding"] = text_vec
                            item["clip_embedding"] = text_vec
                            item["embedding_meta"] = {
                                "preprocessed": False,
                                "source": "clip_text_reembed",
                                "backbone": "CLIP-ViT-B/32",
                                "text_repr": text_repr,
                            }
                            text_fallback += 1
                        else:
                            failed += 1

                    save_wardrobe(updated)

                st.success(
                    f"재임베딩 완료 — 이미지 {success}개 / 텍스트 fallback {text_fallback}개 / 실패 {failed}개"
                )
                if failed > 0:
                    st.warning(
                        "실패한 아이템은 CLIP 모델이 설치 안 됐거나 텍스트 메타가 비어있는 경우예요. "
                        "PowerShell에서 'pip install torch git+https://github.com/openai/CLIP.git' 확인."
                    )
                st.rerun()


# ==========================================
# 탭 3: 날씨 맞춤 추천
# ==========================================
with tab_weather:
    st.subheader("지역 날씨 & 무신사 트렌드 맞춤 추천")
    st.write("지역과 날짜를 입력하면, 날씨 + 트렌드 + 내 옷장을 종합해서 코디를 제안해 드려요.")

    col_region, col_date, col_gender = st.columns(3)
    with col_region:
        selected_region = st.text_input(
            "지역 입력 (시/군/구 단위)", value="용인시", placeholder="예: 용인시, 강릉시"
        )
    with col_date:
        selected_date = st.date_input("외출 날짜")
    with col_gender:
        selected_gender = st.radio("성별", ["여성", "남성"], horizontal=True)

    st.markdown("---")

    if st.button("추천 코디 보기", type="primary"):
        if not selected_region.strip():
            st.warning("지역을 입력해 주세요.")
        else:
            with st.spinner(f"'{selected_region}'의 날씨와 내 옷장을 분석 중입니다..."):
                current_wardrobe = load_wardrobe()
                recommendation = recommend_fashion_for_weather(
                    selected_date, selected_region, selected_gender, current_wardrobe
                )

            if recommendation.get("success"):
                st.success("분석 완료!")
                with st.container(border=True):
                    # 풍부한 날씨 패널 — 팀원 weather_service.py 머지로 확장
                    weather_data = recommendation.get("weather_data") or {}
                    if weather_data and weather_data.get("temp") is not None:
                        source_label = (
                            "OpenWeatherMap" if weather_data.get("source") == "openweathermap"
                            else "wttr.in (무료)"
                        )
                        st.caption(
                            f"📍 {weather_data.get('city', selected_region)} "
                            f"· 날씨 데이터: {source_label} · 검색 엔진: **{recommendation['engine_used']}**"
                        )
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric(
                                "현재 기온",
                                f"{weather_data['temp']}°C",
                                delta=(
                                    f"체감 {weather_data['feels_like']}°C"
                                    if weather_data.get("feels_like") is not None
                                    and weather_data["feels_like"] != weather_data["temp"]
                                    else None
                                ),
                                delta_color="off",
                            )
                        with m2:
                            st.metric("날씨", weather_data.get("condition", "-"))
                        with m3:
                            st.metric("옷차림 단계", weather_data.get("outfit_season", "-"))
                        with m4:
                            humidity = weather_data.get("humidity")
                            st.metric(
                                "습도",
                                f"{humidity}%" if humidity is not None else "-",
                            )
                        # 옷차림 단계별 추천 태그 힌트
                        from skills.weather_fashion import _OUTFIT_SEASON_KEYWORDS
                        season = weather_data.get("outfit_season")
                        if season and season in _OUTFIT_SEASON_KEYWORDS:
                            hint_tags = ", ".join(_OUTFIT_SEASON_KEYWORDS[season])
                            st.caption(f"💡 이 단계 추천 태그: {hint_tags}")
                    else:
                        st.info(f"AI 날씨 분석: {recommendation['weather_context']}")
                        st.caption(f"검색 엔진: **{recommendation['engine_used']}** 가동 중")

                    # 폴백 모드: 임베딩 미보유 또는 무신사 API 실패 시 텍스트 LLM 결과 표시
                    if recommendation.get("fallback") and recommendation.get("text_recommendation"):
                        st.warning(
                            "이미지 기반 추천이 불가능해서 텍스트 LLM 폴백으로 전환했어요. "
                            "(backend wardrobe.py의 /recommend 패턴과 동일한 방식)"
                        )
                        st.markdown(recommendation["text_recommendation"])
                    else:
                        # 정상 경로: 무신사 스냅 + 카테고리별 매칭 결과
                        col_snap, col_outfit = st.columns(2)
                        with col_snap:
                            st.success("이 시기 무신사 인기 스냅")
                            st.image(recommendation["best_snap"]["url"], use_container_width=True)

                        with col_outfit:
                            st.success("내 옷장 맞춤 코디 제안")
                            st.write("스냅 무드를 내 옷장 아이템으로 재현해 보세요!")
                            for category in ["아우터", "상의", "하의", "신발"]:
                                matched_item = recommendation["recommendations"].get(category)
                                if matched_item:
                                    # backend ClothingItem 호환: name 우선, item_name fallback
                                    display_name = (
                                        matched_item.get("name")
                                        or matched_item.get("item_name", "이름 없음")
                                    )
                                    # 매칭된 옷장 아이템 썸네일 — 자동 분리/등록 시 저장됐다면 보임
                                    thumb = matched_item.get("image_url", "")
                                    if thumb:
                                        thumb_c, info_c = st.columns([1, 3])
                                        with thumb_c:
                                            st.image(thumb, use_container_width=True)
                                        with info_c:
                                            st.markdown(f"**[{category}]** {display_name}")
                                            tag_text = matched_item.get("tags") or "-"
                                            st.caption(
                                                f"색상: {matched_item.get('color', '?')} / "
                                                f"핏: {matched_item.get('fit_info', '?')} / "
                                                f"태그: {tag_text}"
                                            )
                                    else:
                                        st.markdown(f"**[{category}]** {display_name}")
                                        tag_text = matched_item.get("tags") or "-"
                                        st.caption(
                                            f"  색상: {matched_item.get('color', '?')} / "
                                            f"핏: {matched_item.get('fit_info', '?')} / "
                                            f"태그: {tag_text}"
                                        )
                                else:
                                    st.markdown(f"**[{category}]** 옷장에 어울리는 아이템이 없어요.")
            else:
                st.error(f"오류: {recommendation.get('comment', '알 수 없는 오류')}")


# ==========================================
# 탭 4: 코디 손민수 (하이브리드 유사도 검색)
# 고도화: 의류 영역 자동 추출 + CLIP 앙상블 + 의미 라벨링
# ==========================================
with tab_snap:
    st.subheader("핀터레스트 코디 손민수")
    reference_file = st.file_uploader(
        "따라하고 싶은 코디 업로드 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"]
    )

    col_pre, col_clip = st.columns(2)
    with col_pre:
        use_preprocess = st.checkbox(
            "의류 영역 자동 추출 (배경 노이즈 제거)",
            value=True,
            help="PDF2 발견: 배경이 다르면 코트 vs 자켓 유사도가 코트 vs 신발보다 낮아지는 현상. "
                 "의류 영역만 잘라서 임베딩하면 정상 동작합니다.",
        )
    with col_clip:
        ensemble_clip_search = st.checkbox(
            "CLIP 앙상블 점수 사용",
            value=True,
            help="CLIP 보조 임베딩이 저장된 아이템에 한해 Gemini 점수와 가중 평균합니다 "
                 "(Gemini 70%, CLIP 30%).",
        )

    if st.button("내 옷장에서 닮은 옷 찾기", type="primary"):
        if not reference_file:
            st.warning("사진을 먼저 첨부해 주세요.")
        else:
            with st.spinner("AI가 의류 영역 추출 → 시각 패턴 → 카테고리 → 앙상블 점수를 계산 중입니다..."):
                reference_bytes = reference_file.read()
                if not reference_bytes:
                    st.error("이미지 파일을 읽지 못했습니다. 다시 업로드해 주세요.")
                else:
                    current_wardrobe = load_wardrobe()
                    similar_items = hybrid_search_clothes(
                        reference_bytes, current_wardrobe,
                        use_preprocess=use_preprocess,
                        ensemble_clip=ensemble_clip_search,
                    )

            if similar_items:
                # 첫 결과의 전처리 메타로 진단 정보 표시
                first = similar_items[0]
                query_meta = first.get("_query_preprocess", {})
                if query_meta.get("preprocessed"):
                    detected_cat = query_meta.get("detected_category") or "?"
                    st.info(f"📸 업로드 이미지에서 '{detected_cat}' 영역만 추출해 검색했어요 (배경 노이즈 제거).")
                elif query_meta.get("source") == "original_fallback":
                    st.caption("ℹ️ 의류 객체 검출 실패 — 원본 이미지로 검색했어요.")

                st.success("가장 유사한 아이템입니다.")
                for rank, item in enumerate(similar_items, start=1):
                    # backend ClothingItem 호환: name 우선, 구 데이터 위해 item_name fallback
                    display_name = item.get("name") or item.get("item_name", "이름 없음")
                    label = item.get("similarity_label", "")
                    final_score = item.get("final_score", 0)

                    # PDF2 임계값 라벨에 색상 부여
                    if label == "거의 동일":
                        badge = f"🟢 **{label}**"
                    elif label == "추천 가능":
                        badge = f"🟡 **{label}**"
                    elif label == "약한 관련":
                        badge = f"🟠 {label}"
                    else:
                        badge = f"⚪ {label}"

                    # 점수 분해 — 디버깅·신뢰도 확인용
                    score_detail = f"Gemini {item.get('gemini_score', 0)}%"
                    if item.get("clip_score") is not None:
                        score_detail += f" + CLIP {item.get('clip_score')}%"
                    if item.get("category_penalty"):
                        score_detail += f" (카테고리 페널티 {item.get('category_penalty')})"

                    tag_text = item.get("tags") or "-"
                    base_caption = (
                        f"색상: {item.get('color', '?')} / "
                        f"핏: {item.get('fit_info', '?')} / "
                        f"태그: {tag_text}"
                    )

                    # 옷장 아이템 썸네일이 있으면 좌측에 표시 (시연용)
                    thumb = item.get("image_url", "")
                    if thumb:
                        thumb_c, info_c = st.columns([1, 4])
                        with thumb_c:
                            st.image(thumb, use_container_width=True)
                        with info_c:
                            st.markdown(f"**{rank}. {display_name}** — {badge} ({final_score}%)")
                            st.caption(f"{base_caption}\n\n  └ {score_detail}")
                    else:
                        st.markdown(f"**{rank}. {display_name}** — {badge} ({final_score}%)")
                        st.caption(f"{base_caption}\n\n  └ {score_detail}")
            else:
                st.warning("유사한 아이템을 찾지 못했거나, 옷장에 임베딩 데이터가 없습니다.")
