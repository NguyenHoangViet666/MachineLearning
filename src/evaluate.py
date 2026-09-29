"""
evaluate.py - Đánh giá mô hình gợi ý phim: Hit-Rate@K, Precision@K, Recall@K, Catalog Coverage, và Query Latency.
Tuân thủ nghiêm ngặt nguyên tắc KHÔNG RÒ RỈ DỮ LIỆU (No Data Leakage).
"""

import time
import logging
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_recommender(
    model, 
    train_ratings: pd.DataFrame, 
    test_ratings: pd.DataFrame, 
    k: int = 10,
    positive_threshold: float = 3.5,
    max_test_users: int = 200
) -> Dict[str, float]:
    """
    Đánh giá hệ thống gợi ý trên tập Test đã được tách riêng.
    
    Quy trình:
    1. Với mỗi user có trong tập test:
       - Lấy tập ground-truth các phim user thích ở tập test (rating >= positive_threshold).
       - Lấy các phim user đã xem trong tập train.
       - Tạo danh sách gợi ý Top-K (từ model đã fit chỉ trên train).
       - Tính Hit (có ít nhất 1 phim đúng), Precision@K, Recall@K.
    2. Tính Catalog Coverage (tỷ lệ phim trong kho được gợi ý ít nhất 1 lần).
    3. Đo thời gian trung bình mỗi truy vấn (Latency).
    """
    # Lấy các user có tương tác tích cực trong test
    test_positives = test_ratings[test_ratings["rating"] >= positive_threshold]
    test_users = test_positives["userId"].unique()
    
    if len(test_users) > max_test_users:
        # Chọn mẫu ngẫu nhiên có cố định seed để tốc độ đánh giá nhanh và nhất quán
        np.random.seed(42)
        eval_users = np.random.choice(test_users, size=max_test_users, replace=False)
    else:
        eval_users = test_users

    hits = []
    precisions = []
    recalls = []
    all_recommended_items: Set[int] = set()
    query_latencies = []

    # Map phim user đã xem trong train để tránh recommend lại
    train_user_items = train_ratings.groupby("userId")["movieId"].apply(set).to_dict()
    test_user_positive_items = test_positives.groupby("userId")["movieId"].apply(set).to_dict()

    # Toàn bộ danh sách phim có trong train
    total_train_movies = set(train_ratings["movieId"].unique())

    is_popularity = hasattr(model, "popular_movies")

    for user_id in eval_users:
        ground_truth = test_user_positive_items.get(user_id, set())
        if not ground_truth:
            continue
            
        train_items = train_user_items.get(user_id, set())
        
        start_time = time.perf_counter()
        
        # Tạo Top-K recommendations cho user
        if is_popularity:
            # Baseline Popularity: Lấy top phim phổ biến mà user chưa xem trong train
            recs = model.recommend(top_k=k + len(train_items))
            rec_movie_ids = [r["movieId"] for r in recs if r["movieId"] not in train_items][:k]
        else:
            # Item-Item Cosine: Tổng hợp điểm tương đồng từ các phim user đã thích nhất trong train
            # Chọn tối đa 5 phim user cho điểm cao nhất trong train làm seed
            user_train_df = train_ratings[train_ratings["userId"] == user_id]
            top_user_movies = user_train_df.sort_values(by="rating", ascending=False).head(5)["movieId"].values
            
            candidate_scores: Dict[int, float] = {}
            for m_id in top_user_movies:
                rec_res = model.recommend(movie_id=m_id, top_k=k * 2)
                for item in rec_res.get("recommendations", []):
                    target_id = item["movieId"]
                    if target_id not in train_items:
                        candidate_scores[target_id] = candidate_scores.get(target_id, 0.0) + item["similarity_score"]
            
            # Sắp xếp lấy Top-K
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            rec_movie_ids = [m_id for m_id, score in sorted_candidates[:k]]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        query_latencies.append(latency_ms)

        # Cập nhật danh sách catalog coverage
        all_recommended_items.update(rec_movie_ids)

        # Tính metric cho user này
        rec_set = set(rec_movie_ids)
        intersect = rec_set.intersection(ground_truth)
        
        hit = 1.0 if len(intersect) > 0 else 0.0
        prec = len(intersect) / float(k) if k > 0 else 0.0
        rec = len(intersect) / float(len(ground_truth)) if len(ground_truth) > 0 else 0.0

        hits.append(hit)
        precisions.append(prec)
        recalls.append(rec)

    hit_rate = float(np.mean(hits)) if hits else 0.0
    precision_at_k = float(np.mean(precisions)) if precisions else 0.0
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    coverage = float(len(all_recommended_items) / len(total_train_movies)) if total_train_movies else 0.0
    avg_latency = float(np.mean(query_latencies)) if query_latencies else 0.0

    results = {
        f"HitRate@{k}": round(hit_rate, 4),
        f"Precision@{k}": round(precision_at_k, 4),
        f"Recall@{k}": round(recall_at_k, 4),
        "Coverage": round(coverage, 4),
        "AvgLatencyMs": round(avg_latency, 2),
        "EvaluatedUsers": len(hits)
    }

    logger.info(f"Kết quả đánh giá (K={k}): {results}")
    return results
