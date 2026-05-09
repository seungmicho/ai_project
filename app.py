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
)

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
# 나중에 SQLite나 Supabase로 마이그레이션할 때도 이 두 함수만 교체하면 됩니다.
# ==========================================
WARDROBE_DB_PATH = "wardrobe_db.json"


def load_wardrobe() -> list[dict]:
    """저장된 옷장 데이터를 불러옵니다. 파일이 없거나 손상됐을 때도 빈 리스트로 안전하게 시작합니다."""
    if not os.path.exists(WARDROBE_DB_PATH):
        return []
    try:
        with open(WARDROBE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # DB 파일이 깨진 경우 — 앱을 멈추는 것보다 빈 상태로 시작하는 게 낫다고 판단
        logger.warning("옷장 DB 파일을 읽는 데 실패했습니다. 새 DB로 시작합니다. 원인: %s", e)
        return []


def save_wardrobe(outfit_data: list[dict]) -> None:
    """옷장 데이터를 저장합니다. 쓰기 도중 오류가 생겨도 기존 데이터를 보호합니다."""
    try:
        with open(WARDROBE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(outfit_data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error("옷장 DB 저장에 실패했습니다: %s", e)
        st.error("DB 저장 중 오류가 발생했습니다. 디스크 공간을 확인해 주세요.")


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

                        # Step 2: 파일 커서를 처음으로 되돌린 뒤 임베딩 벡터 추출
                        # (parse_screenshot 내부에서 이미 read()를 한 번 했기 때문에 필수!)
                        file.seek(0)
                        image_bytes = file.read()
                        style_vector = get_image_embedding_vector(image_bytes)

                        # Step 3: 임베딩 실패 시에도 아이템 자체는 저장 (나중에 재임베딩 가능)
                        outfit_items = parse_result["data"].get("purchased_items", [])
                        if not outfit_items:
                            fail_log.append(f"[{file.name}] 분석은 됐지만 아이템을 찾지 못했어요.")
                            continue

                        for item in outfit_items:
                            item["embedding"] = style_vector  # None일 수도 있음 — 이후 검색 시 필터링
                        
                        collected_items.extend(outfit_items)
                        success_count += 1

                    # 결과 리포트
                    if success_count > 0:
                        st.session_state["parsed_outfit_items"] = collected_items
                        st.success(
                            f"총 {len(uploaded_files)}장 중 {success_count}장 분석 완료! 아래에서 수정 후 저장하세요."
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
        outfit_df = pd.DataFrame(st.session_state["parsed_outfit_items"])
        edited_outfit_df = st.data_editor(
            outfit_df, use_container_width=True, num_rows="dynamic"
        )

        if st.button("내 옷장에 등록하기", type="primary"):
            current_wardrobe = load_wardrobe()
            new_items = edited_outfit_df.to_dict(orient="records")
            current_wardrobe.extend(new_items)
            save_wardrobe(current_wardrobe)

            st.balloons()
            st.success("성공적으로 옷장에 저장되었습니다! [내 옷장 관리] 탭에서 확인해 보세요.")
            st.session_state["parsed_outfit_items"] = []
    else:
        st.info("위에서 구매 내역을 먼저 분석해 주세요.")


# ==========================================
# 탭 2: 내 옷장 관리
# ==========================================
with tab_wardrobe:
    st.subheader("나의 옷장 데이터베이스")
    wardrobe_data = load_wardrobe()

    if not wardrobe_data:
        st.warning("아직 옷장에 등록된 옷이 없습니다.")
    else:
        st.write(f"현재 총 **{len(wardrobe_data)}벌**의 옷이 등록되어 있습니다.")
        wardrobe_df = pd.DataFrame(wardrobe_data)
        edited_wardrobe_df = st.data_editor(
            wardrobe_df, key="wardrobe_editor", use_container_width=True, num_rows="dynamic"
        )

        if st.button("변경사항 DB에 업데이트"):
            updated_data = edited_wardrobe_df.to_dict(orient="records")
            save_wardrobe(updated_data)
            st.success("옷장 데이터가 업데이트되었습니다!")


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
                    st.info(f"AI 날씨 분석: {recommendation['weather_context']}")
                    st.caption(f"검색 엔진: **{recommendation['engine_used']}** 가동 중")

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
                                st.markdown(f"**[{category}]** {matched_item['item_name']}")
                                st.caption(
                                    f"  색상: {matched_item.get('color', '?')} / "
                                    f"핏: {matched_item.get('fit_info', '?')}"
                                )
                            else:
                                st.markdown(f"**[{category}]** 옷장에 어울리는 아이템이 없어요.")
            else:
                st.error(f"오류: {recommendation.get('comment', '알 수 없는 오류')}")


# ==========================================
# 탭 4: 코디 손민수 (하이브리드 유사도 검색)
# ==========================================
with tab_snap:
    st.subheader("핀터레스트 코디 손민수")
    reference_file = st.file_uploader(
        "따라하고 싶은 코디 업로드 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"]
    )

    if st.button("내 옷장에서 닮은 옷 찾기", type="primary"):
        if not reference_file:
            st.warning("사진을 먼저 첨부해 주세요.")
        else:
            with st.spinner("AI가 시각 패턴과 카테고리를 분석 중입니다..."):
                reference_bytes = reference_file.read()
                if not reference_bytes:
                    st.error("이미지 파일을 읽지 못했습니다. 다시 업로드해 주세요.")
                else:
                    current_wardrobe = load_wardrobe()
                    similar_items = hybrid_search_clothes(reference_bytes, current_wardrobe)

            if similar_items:
                st.success("가장 유사한 아이템입니다.")
                for rank, item in enumerate(similar_items, start=1):
                    st.markdown(f"**{rank}. {item.get('item_name', '이름 없음')}**")
                    st.caption(
                        f"  색상: {item.get('color', '?')} / "
                        f"핏: {item.get('fit_info', '?')} "
                        f"(종합 일치율: {item.get('final_score', 0)}%)"
                    )
            else:
                st.warning("유사한 아이템을 찾지 못했거나, 옷장에 임베딩 데이터가 없습니다.")
