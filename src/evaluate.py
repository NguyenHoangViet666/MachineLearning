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
    max_test_users: int = None,
    random_state: int = 42
) -> Dict[str, Any]:
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
    
    if max_test_users is not None and len(test_users) > max_test_users:
        # Chọn mẫu ngẫu nhiên có cố định seed để tốc độ đánh giá nhanh và nhất quán
        eval_users = np.random.default_rng(random_state).choice(test_users, size=max_test_users, replace=False)
    else:
        eval_users = test_users

    hits = []
    precisions = []
    recalls = []
    all_recommended_items: Set[int] = set()
    query_latencies = []
    user_results = []

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
        user_results.append({"userId": int(user_id), "train_interactions": len(train_items), "hit": hit, "precision": prec, "recall": rec})

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
    rng = np.random.default_rng(random_state)
    results["ConfidenceIntervals95"] = {}
    for label, values in [(f"HitRate@{k}", hits), (f"Precision@{k}", precisions), (f"Recall@{k}", recalls)]:
        if values:
            samples = rng.choice(np.asarray(values), size=(1000, len(values)), replace=True).mean(axis=1)
            results["ConfidenceIntervals95"][label] = [round(float(v), 4) for v in np.quantile(samples, [0.025, 0.975])]
    results["ByHistoryLength"] = {}
    for name, lower, upper in [("short", 0, 50), ("medium", 50, 200), ("long", 200, float("inf"))]:
        rows = [r for r in user_results if lower <= r["train_interactions"] < upper]
        results["ByHistoryLength"][name] = {"users": len(rows), "hit_rate": float(np.mean([r["hit"] for r in rows])) if rows else None, "precision": float(np.mean([r["precision"] for r in rows])) if rows else None}
    results["Protocol"] = {"unit": "user; aggregate cosine of up to five highest-rated training movies", "positive_threshold": positive_threshold, "seed": random_state, "bootstrap": "1000 user resamples; descriptive uncertainty, not repeated test tuning", "eligible_users": len(test_users), "sample_limit": max_test_users}

    logger.info(f"Kết quả đánh giá (K={k}): {results}")
    return results


def main():
    import argparse
    import json
    from pathlib import Path
    from src.models import CosineItemItemRecommender
    parser = argparse.ArgumentParser(description="Evaluate a saved model without retraining")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", default="models/cosine_recommender.pkl")
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    config = json.loads(Path("models/config.json").read_text(encoding="utf-8"))
    if args.split == "test":
        # The final test is already frozen by train. Inspect it, do not recompute.
        frozen = json.loads(Path("reports/experiment_results.json").read_text(encoding="utf-8"))
        if frozen["status"] != "complete":
            raise SystemExit("Final test is incomplete; inspect the recorded checkpoint. Do not silently rerun test.")
        print(json.dumps(frozen["test"], ensure_ascii=False, indent=2))
        return
    model = CosineItemItemRecommender.load(args.model)
    result = evaluate_recommender(model, pd.read_csv("data/processed/train_ratings.csv"), pd.read_csv("data/processed/val_ratings.csv"), k=config["selected"]["k"], random_state=config["seed"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
