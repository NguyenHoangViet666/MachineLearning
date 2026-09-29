"""
train.py - Huấn luyện mô hình, thực hiện 4 thí nghiệm bắt buộc,
vẽ biểu đồ báo cáo và lưu artifacts cho Web Serving.
"""

import os
import sys

# Đảm bảo đường dẫn gốc của project có trong sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import (
    download_and_extract_data, 
    load_raw_data, 
    inspect_data_quality,
    split_train_val_test, 
    save_processed_data
)
from src.models import PopularityRecommender, CosineItemItemRecommender
from src.evaluate import evaluate_recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Cấu hình phong cách biểu đồ khoa học, sắc nét
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150


def plot_eda(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    """Vẽ biểu đồ khám phá dữ liệu (EDA): Phân bố rating và Phân bố long-tail số lượng đánh giá."""
    logger.info("Đang tạo biểu đồ EDA...")
    
    # 1. Phân bố điểm đánh giá (Rating distribution)
    plt.figure(figsize=(8, 5))
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="Blues_r")
    plt.title("Phân bố điểm đánh giá trên MovieLens latest-small", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Điểm đánh giá (Rating)", fontsize=11)
    plt.ylabel("Số lượng lượt đánh giá", fontsize=11)
    for i, count in enumerate(rating_counts.values):
        plt.text(i, count + 500, f"{count:,}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_rating_distribution.png"))
    plt.close()

    # 2. Phân bố Long-tail (Số lượng đánh giá theo từng phim)
    movie_counts = ratings_df["movieId"].value_counts().values
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(movie_counts)), movie_counts, color="#e63946", lw=2)
    plt.fill_between(np.arange(len(movie_counts)), movie_counts, color="#e63946", alpha=0.15)
    plt.yscale("log")
    plt.title("Hiện tượng 'Long-tail' trong đánh giá phim (Thang đo Log)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Thứ hạng phim (Ranked Movies)", fontsize=11)
    plt.ylabel("Số lượt đánh giá (Log scale)", fontsize=11)
    plt.axvline(x=len(movie_counts) * 0.2, color="#457b9d", linestyle="--", label="Top 20% phim")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_long_tail_movies.png"))
    plt.close()
    
    logger.info("Đã lưu biểu đồ EDA vào reports/figures/.")


def run_experiment_1(train_df: pd.DataFrame, test_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Thí nghiệm 1: So sánh Baseline Phổ biến (Popularity) với Cosine Item-Item."""
    logger.info("=== THÍ NGHIỆM 1: POPULARITY vs COSINE ===")
    
    # 1. Baseline
    pop_model = PopularityRecommender(min_ratings=10)
    pop_model.fit(train_df, movies_df)
    pop_metrics = evaluate_recommender(pop_model, train_df, test_df, k=10)

    # 2. Cosine (Raw)
    cosine_raw = CosineItemItemRecommender(min_movie_ratings=5, normalize_mean_centered=False)
    cosine_raw.fit(train_df, movies_df)
    cos_raw_metrics = evaluate_recommender(cosine_raw, train_df, test_df, k=10)

    # Vẽ biểu đồ so sánh
    metrics_to_plot = ["HitRate@10", "Precision@10", "Recall@10", "Coverage"]
    pop_vals = [pop_metrics[m] for m in metrics_to_plot]
    cos_vals = [cos_raw_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, pop_vals, width, label="Popularity Baseline", color="#457b9d")
    plt.bar(x + width/2, cos_vals, width, label="Cosine Item-Item (Raw)", color="#2a9d8f")
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylabel("Giá trị Metric", fontsize=11)
    plt.title("Thí nghiệm 1: So sánh Mô hình Phổ biến vs Cosine Similarity (K=10)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    for i, v in enumerate(pop_vals):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(cos_vals):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp1_baseline_vs_cosine.png"))
    plt.close()

    return {"popularity": pop_metrics, "cosine_raw": cos_raw_metrics}


def run_experiment_2(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    movies_df: pd.DataFrame,
    cos_raw_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Thí nghiệm 2: So sánh Cosine ma trận thô vs Ma trận đã trừ trung bình người dùng (Mean-centered)."""
    logger.info("=== THÍ NGHIỆM 2: RAW COSINE vs MEAN-CENTERED COSINE ===")
    
    cosine_centered = CosineItemItemRecommender(min_movie_ratings=5, normalize_mean_centered=True)
    cosine_centered.fit(train_df, movies_df)
    cos_centered_metrics = evaluate_recommender(cosine_centered, train_df, test_df, k=10)

    # Vẽ biểu đồ so sánh
    metrics_to_plot = ["HitRate@10", "Precision@10", "Recall@10", "Coverage"]
    raw_vals = [cos_raw_metrics[m] for m in metrics_to_plot]
    cen_vals = [cos_centered_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, raw_vals, width, label="Cosine Thô (Raw)", color="#f4a261")
    plt.bar(x + width/2, cen_vals, width, label="Cosine Trừ Trung Bình (Adjusted)", color="#e76f51")
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylabel("Giá trị Metric", fontsize=11)
    plt.title("Thí nghiệm 2: Hiệu quả của việc Khử thiên vị người dùng (Mean-centering)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    for i, v in enumerate(raw_vals):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(cen_vals):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp2_raw_vs_mean_centered.png"))
    plt.close()

    return {"cosine_raw": cos_raw_metrics, "cosine_centered": cos_centered_metrics, "model": cosine_centered}


def run_experiment_3(train_df: pd.DataFrame, val_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Thí nghiệm 3: Khảo sát siêu tham số K (5, 10, 20) và min_movie_ratings (1, 5, 10) trên tập Validation."""
    logger.info("=== THÍ NGHIỆM 3: KHẢO SÁT K VÀ MIN_RATINGS ===")
    
    k_list = [5, 10, 20]
    min_ratings_list = [1, 5, 10]
    results = []

    for min_r in min_ratings_list:
        model = CosineItemItemRecommender(min_movie_ratings=min_r, normalize_mean_centered=True)
        model.fit(train_df, movies_df)
        
        for k_val in k_list:
            m = evaluate_recommender(model, train_df, val_df, k=k_val, max_test_users=100)
            results.append({
                "min_ratings": min_r,
                "k": k_val,
                "HitRate": m[f"HitRate@{k_val}"],
                "Precision": m[f"Precision@{k_val}"],
                "Recall": m[f"Recall@{k_val}"],
                "Coverage": m["Coverage"]
            })

    res_df = pd.DataFrame(results)
    logger.info(f"Kết quả khảo sát:\n{res_df}")

    # Vẽ biểu đồ heatmap / line chart
    plt.figure(figsize=(10, 4.5))
    
    plt.subplot(1, 2, 1)
    for min_r in min_ratings_list:
        sub = res_df[res_df["min_ratings"] == min_r]
        plt.plot(sub["k"], sub["Precision"], marker="o", label=f"min_ratings={min_r}")
    plt.title("Biến thiên Precision theo K", fontweight="bold")
    plt.xlabel("Top-K Gợi ý")
    plt.ylabel("Precision@K")
    plt.legend()

    plt.subplot(1, 2, 2)
    for min_r in min_ratings_list:
        sub = res_df[res_df["min_ratings"] == min_r]
        plt.plot(sub["k"], sub["Coverage"], marker="s", label=f"min_ratings={min_r}")
    plt.title("Độ phủ (Coverage) theo K", fontweight="bold")
    plt.xlabel("Top-K Gợi ý")
    plt.ylabel("Catalog Coverage")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp3_k_and_min_ratings.png"))
    plt.close()

    return {"grid_search": results}


def run_experiment_4(model: CosineItemItemRecommender, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Thí nghiệm 4: Phân tích ít nhất 10 ca truy vấn thất bại, thiếu dữ liệu hoặc thiếu đa dạng."""
    logger.info("=== THÍ NGHIỆM 4: PHÂN TÍCH 10 CA TRUY VẤN THẤT BẠI / NGOẠI BIÊN ===")
    
    # Lấy các trường hợp đặc biệt:
    # 1. Phim rất ít rating (Cold-start)
    # 2. Phim bom tấn quá phổ biến (Popularity bias)
    # 3. Phim thể loại độc lạ / đa thể loại
    # 4. Phim hoàn toàn không có trong ma trận
    
    test_cases = [
        # (movieId, Loai_ca, Ly_do_phan_tich)
        (1, "Bom tấn", "Toy Story (1995) - Rất nhiều rating, kiểm tra xem có bị chiếm bởi các phim bom tấn khác không"),
        (260, "Phim kinh điển", "Star Wars: Episode IV - A New Hope (1995) - Thử độ đa dạng thể loại Sci-Fi"),
        (356, "Phim đại chúng", "Forrest Gump (1994) - Kiểm tra độ tương đồng có thiên lệch sang phim drama/phổ biến"),
        (8844, "Ít tương tác", "Jumanji (phần tiếp theo hoặc phim ít rating)"),
        (193581, "Cold-start cực hạn", "Black Butler: Book of the Atlantic (2017) - Chỉ có 1 rating"),
        (999999, "Out-of-vocabulary (OOV)", "Phim không hề tồn tại trong cơ sở dữ liệu"),
        (47, "Phim hình sự giật gân", "Seven (Se7en) (1995) - Kiểm tra xem gợi ý có giữ được đúng tone đen tối"),
        (296, "Tác phẩm độc đáo", "Pulp Fiction (1994) - Phim có sự pha trộn giữa Crime, Comedy"),
        (117176, "Phim hoạt hình hiện đại", "Big Hero 6 (2014)"),
        (858, "Kinh điển Crime/Drama", "The Godfather (1972) - Kiểm tra phim tiếp nối có gợi ý The Godfather Part II không")
    ]

    analysis_results = []
    for m_id, case_type, note in test_cases:
        movie_title = movies_df[movies_df["movieId"] == m_id]["title"].values
        title = movie_title[0] if len(movie_title) > 0 else "Unknown / Not found"
        
        res = model.recommend(m_id, top_k=5)
        recs = res.get("recommendations", [])
        
        issues = []
        if res.get("status") == "warning":
            issues.append("Cảnh báo thiếu dữ liệu hoặc không tìm thấy.")
        elif len(recs) == 0:
            issues.append("Không tìm được phim tương đồng nào (similarity = 0).")
        else:
            # Kiểm tra độ đa dạng thể loại
            rec_genres = [r["genres"] for r in recs]
            if len(set(rec_genres)) <= 1:
                issues.append("Thiếu đa dạng thể loại (tất cả phim gợi ý đều cùng một nhóm thể loại).")
            # Kiểm tra điểm tương đồng quá thấp
            if recs[0]["similarity_score"] < 0.2:
                issues.append("Điểm tương đồng thấp (< 0.2), liên kết yếu.")

        analysis_results.append({
            "movieId": m_id,
            "title": title,
            "case_type": case_type,
            "note": note,
            "status": res.get("status"),
            "warning": res.get("warning"),
            "num_recommendations": len(recs),
            "top_1_sim": recs[0]["title"] if recs else "None",
            "top_1_score": recs[0]["similarity_score"] if recs else 0.0,
            "identified_issues": issues if issues else ["Gợi ý tốt, tương đồng cao."]
        })

    with open(os.path.join(FIGURES_DIR, "exp4_failure_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã hoàn thành phân tích {len(analysis_results)} ca và lưu vào reports/figures/exp4_failure_analysis.json")
    return analysis_results


def create_model_card(best_model: CosineItemItemRecommender, exp1_res, exp2_res, exp3_res):
    """Tạo Model Card chuẩn mực khoa học lưu vào models/model_card.json."""
    model_card = {
        "model_name": "MovieLens Item-Item Cosine Recommender",
        "version": "1.0.0",
        "author": "Nhóm 02 sinh viên (Lớp 12523W.1)",
        "intended_use": {
            "primary_task": "Item-to-item movie recommendation based on user interaction vectors",
            "intended_users": "Nền tảng xem phim, phục vụ người dùng tìm phim tương tự",
            "out_of_scope": "Dự đoán điểm số chính xác (Rating regression), hồ sơ người dùng nhạy cảm"
        },
        "training_data": {
            "dataset": "MovieLens latest-small (GroupLens)",
            "num_ratings_train": int(best_model.metadata.get("n_entries", 0)),
            "num_movies_indexed": int(best_model.metadata.get("n_movies", 0)),
            "num_users_indexed": int(best_model.metadata.get("n_users", 0)),
            "sparsity": f"{best_model.metadata.get('sparsity', 0.0)*100:.2f}%"
        },
        "technical_specifications": {
            "algorithm": "Adjusted Cosine Similarity (Item-Item Collaborative Filtering)",
            "mean_centering": True,
            "min_movie_ratings": best_model.min_movie_ratings,
            "similarity_metric": "Cosine (Row-normalized dot product)"
        },
        "evaluation_summary": {
            "exp1_vs_baseline": exp1_res,
            "exp2_mean_centering_gain": exp2_res["cosine_centered"],
            "leakage_prevention": "Per-user temporal split, zero test interactions included in feature matrix"
        },
        "limitations": [
            "Cold-start: Không thể gợi ý chính xác cho phim mới chưa có đánh giá hoặc ít hơn 5 đánh giá.",
            "Long-tail bias: Phim ít người xem thường có ít tương đồng chất lượng so với phim phổ biến.",
            "Sở thích tĩnh: Chưa tính đến sự thay đổi gu xem phim theo thời gian thực."
        ],
        "ethical_considerations": "Không suy diễn thông tin nhân khẩu học hay sở thích riêng tư. Minh bạch điểm tương đồng và lý do gợi ý."
    }

    card_path = os.path.join(MODELS_DIR, "model_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu Model Card vào {card_path}")


def main():
    logger.info("BẮT ĐẦU TOÀN BỘ QUY TRÌNH HUẤN LUYỆN VÀ THÍ NGHIỆM...")
    
    # 1. Tải và xử lý dữ liệu
    dataset_path = download_and_extract_data()
    ratings_df, movies_df = load_raw_data(dataset_path)
    inspect_data_quality(ratings_df, movies_df)
    
    # Biểu đồ EDA
    plot_eda(ratings_df, movies_df)
    
    # Split chống rò rỉ
    train_df, val_df, test_df = split_train_val_test(ratings_df, test_ratio=0.2, val_ratio=0.1, random_state=42)
    save_processed_data(train_df, val_df, test_df, movies_df)
    
    # 2. Bốn thí nghiệm bắt buộc
    exp1_res = run_experiment_1(train_df, test_df, movies_df)
    exp2_res = run_experiment_2(train_df, test_df, movies_df, exp1_res["cosine_raw"])
    exp3_res = run_experiment_3(train_df, val_df, movies_df)
    
    # Chọn mô hình tốt nhất (Cosine Mean-centered, min_ratings=5)
    best_model = exp2_res["model"]
    
    # Phân tích 10 ca lỗi
    run_experiment_4(best_model, train_df, movies_df)
    
    # 3. Lưu Model Card & Model Artifact để Serving Web
    best_model.save(os.path.join(MODELS_DIR, "cosine_recommender.pkl"))
    create_model_card(best_model, exp1_res, exp2_res, exp3_res)
    
    logger.info("TẤT CẢ 4 THÍ NGHIỆM ĐÃ HOÀN TẤT VÀ ARTIFACTS ĐÃ ĐƯỢC LƯU SẴN SÀNG!")


if __name__ == "__main__":
    main()
