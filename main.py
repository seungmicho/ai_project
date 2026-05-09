# Streamlit 없이 터미널에서 파서 스킬을 빠르게 검증
# 실제 서비스 진입점은 app.py (streamlit run app.py)
import json
from skills.shopping_parser import parse_shopping_history


def run_local_agent() -> None:
    print("[AI 비서] 시스템 시동 완료!")
    print("=" * 40)

    # 실제 사용자가 입력할 법한 자연어 문장으로 테스트
    user_input = "나 어제 무신사에서 검정색 와이드 슬랙스 하의랑 아이보리 크롭 가디건 상의 결제했어. 내 옷장에 추가해 줘."
    print(f"[사용자] {user_input}\n")
    print("[AI 비서] 구매 내역 감지 — 쇼핑 파서 스킬 실행 중...\n")

    result = parse_shopping_history(user_input)

    if result["status"] == "success":
        print("[AI 비서] 분석 완료. 아래 데이터를 DB에 저장합니다.")
        print("-" * 40)
        for idx, item in enumerate(result["data"].get("purchased_items", []), start=1):
            print(f"[{idx}번 아이템]")
            print(f"  상품명   : {item.get('item_name', '?')}")
            print(f"  카테고리 : {item.get('category', '?')}")
            print(f"  색상     : {item.get('color', '?')}")
            print(f"  핏       : {item.get('fit_info', '?')}\n")
    else:
        print(f"[에러] 분석에 실패했습니다: {result.get('message', '원인 불명')}")


if __name__ == "__main__":
    run_local_agent()
