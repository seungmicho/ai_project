# FAISS ANN(근사치 검색)이 기존 Numpy 전수조사보다 얼마나 빠른지 벤치마크
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --- 가상 데이터 준비 ---
EMBEDDING_DIMENSION = 3072  # Gemini embedding-2-preview 기준 차원 수
NUM_WARDROBE_ITEMS = 100_000

print(f"가상의 옷 데이터 {NUM_WARDROBE_ITEMS:,}개 생성 중...")

wardrobe_vectors = np.random.random((NUM_WARDROBE_ITEMS, EMBEDDING_DIMENSION)).astype("float32")
faiss.normalize_L2(wardrobe_vectors)

query_vector = np.random.random((1, EMBEDDING_DIMENSION)).astype("float32")
faiss.normalize_L2(query_vector)

print("-" * 40)

# ==========================================
# 방식 1: Numpy 전수조사 (모든 아이템과 1:1 비교)
# ==========================================
start = time.perf_counter()
similarities = np.dot(wardrobe_vectors, query_vector.T).flatten()
top3_numpy = np.argsort(similarities)[-3:][::-1]
numpy_elapsed = time.perf_counter() - start
print(f"Numpy 전수조사 소요 시간: {numpy_elapsed:.4f}초")

# ==========================================
# 방식 2: FAISS ANN (클러스터로 나눠서 근사치 검색)
# ==========================================
# nlist=100: 10만 개를 100개 군집으로 나눔
# nprobe=10: 검색 시 100개 군집 중 10개만 열어봄 
NUM_CLUSTERS = 100
NUM_PROBE_CLUSTERS = 10

print(f"FAISS ANN 인덱스 학습 중... ({NUM_CLUSTERS}개 클러스터)")
quantizer = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
ann_index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIMENSION, NUM_CLUSTERS, faiss.METRIC_INNER_PRODUCT)
ann_index.train(wardrobe_vectors)
ann_index.add(wardrobe_vectors)
ann_index.nprobe = NUM_PROBE_CLUSTERS

start = time.perf_counter()
distances, top3_faiss = ann_index.search(query_vector, 3)
faiss_elapsed = time.perf_counter() - start
print(f"FAISS ANN 소요 시간: {faiss_elapsed:.4f}초")

print("-" * 40)
speedup_ratio = numpy_elapsed / faiss_elapsed
print(f"결론: FAISS ANN이 약 {speedup_ratio:.1f}배 빠릅니다.")

# ==========================================
# 시각화: 막대그래프
# ==========================================
labels = ["Numpy 전수조사", f"FAISS ANN\n(nprobe={NUM_PROBE_CLUSTERS})"]
times = [numpy_elapsed, faiss_elapsed]
bar_colors = ["#ff9999", "#66b3ff"]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, times, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.2)

ax.set_title(
    f"10만 개 옷장 데이터 검색 속도 비교\n(FAISS ANN 도입 시 약 {speedup_ratio:.1f}배 향상)",
    fontsize=14,
    fontweight="bold",
    pad=16,
)
ax.set_ylabel("소요 시간 (초)", fontsize=12)
ax.set_ylim(0, max(times) * 1.25)  

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(times) * 0.02,
        f"{height:.4f}초",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
output_filename = "faiss_ann_benchmark.png"
plt.savefig(output_filename, dpi=300)
print(f"\n벤치마크 그래프가 '{output_filename}'으로 저장되었습니다.")
