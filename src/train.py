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
from src.data import clean_and_document
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


def analyze_failure_cases(model, movies_df):
    """Audit real failures; never label a successful query as a failure."""
    from collections import Counter
    metadata = movies_df.set_index("movieId").to_dict(orient="index")
    cases = []
    indexed = sorted(model.builder.movie2idx)
    cold = sorted(set(metadata) - set(indexed))
    # Indexed cases are examined first to include diversity/weak-signal failures.
    for movie_id in indexed + cold:
        result = model.recommend(movie_id, top_k=5)
        recs = result.get("recommendations", [])
        issues = []
        genre_fraction = 0.0
        if result["status"] == "warning":
            issues.append("Không đủ rating train để tạo vector (cold-start).")
        elif not recs:
            issues.append("Không có phim với cosine dương.")
        else:
            counts = Counter(g for r in recs for g in r["genres"].split("|") if g != "(no genres listed)")
            genre_fraction = max(counts.values(), default=0) / len(recs)
            if genre_fraction >= 0.8:
                issues.append("Ít nhất 80% gợi ý cùng một thể loại; rủi ro thiếu đa dạng, không kết luận chất lượng thấp.")
            if recs[0]["similarity_score"] < 0.2:
                issues.append("Cosine cao nhất dưới 0.2; tín hiệu tương đồng yếu.")
            if len(recs) < 5:
                issues.append("Không đủ 5 gợi ý với cosine dương.")
        if not issues:
            continue
        cases.append({"movieId": int(movie_id), "title": metadata[movie_id]["title"], "case_type": "Cold-start" if result["status"] == "warning" else "Tín hiệu yếu / thiếu đa dạng", "note": "Quét movieId tăng dần; chỉ giữ ca vi phạm tiêu chí công bố trước.", "status": result["status"], "warning": result.get("warning"), "num_recommendations": len(recs), "top_1_sim": recs[0]["title"] if recs else "None", "top_1_score": recs[0]["similarity_score"] if recs else 0.0, "dominant_genre_fraction": genre_fraction, "identified_issues": issues, "recommendations": recs, "suggested_action": "Thu thập thêm tương tác / hiển thị cảnh báo" if not recs else "Cân nhắc reranking đa dạng và kiểm tra thủ công"})
        # Reserve three slots for real cold-start examples, when available.
        if len(cases) >= (7 if movie_id in model.builder.movie2idx and cold else 10):
            if movie_id in model.builder.movie2idx and cold:
                break
            if len(cases) >= 10:
                break
    if cold and len(cases) < 10:
        for movie_id in cold[:10 - len(cases)]:
            result = model.recommend(movie_id, top_k=5)
            cases.append({"movieId": int(movie_id), "title": metadata[movie_id]["title"], "case_type": "Cold-start", "note": "Phim thật trong catalog, thiếu rating train.", "status": "warning", "warning": result["warning"], "num_recommendations": 0, "top_1_sim": "None", "top_1_score": 0.0, "identified_issues": [result["warning"]], "recommendations": [], "suggested_action": "Thu thập thêm rating; chưa phục vụ gợi ý."})
    if len(cases) < 10:
        raise ValueError("Fewer than ten genuine failure/diversity cases; expand the documented audit")
    with open(os.path.join(FIGURES_DIR, "exp4_failure_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(cases[:10], f, ensure_ascii=False, indent=2)
    return cases[:10]


def select_candidate(rows):
    # Freeze the rule before touching test: Precision@K, then coverage;
    # prefer smaller K and simpler raw representation on exact ties.
    return max(rows, key=lambda r: (r["metrics"][f"Precision@{r['k']}"], r["metrics"]["Coverage"], -r["k"], not r["mean_centered"], -r["min_ratings"]))


def main():
    import argparse
    import platform
    import hashlib
    from pathlib import Path
    from datetime import datetime, timezone
    from importlib.metadata import version
    parser = argparse.ArgumentParser(description="Validation selection followed by one frozen final-test run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development", action="store_true", help="Validation only; no final-test access or serving artifact changes")
    args = parser.parse_args()
    results_path = Path("reports/experiment_results.json")
    if results_path.exists() and not args.development:
        raise SystemExit("Final test already frozen. Use --development for further validation work; inspect test with python -m src.evaluate --split test.")
    dataset_path = download_and_extract_data()
    ratings_df, movies_df = load_raw_data(dataset_path)
    ratings_df, movies_df = clean_and_document(ratings_df, movies_df, dataset_path)
    train_df, val_df, test_df = split_train_val_test(ratings_df, random_state=args.seed)
    save_processed_data(train_df, val_df, test_df, movies_df)
    plot_eda(train_df, movies_df)
    baseline = PopularityRecommender()
    baseline.fit(train_df, movies_df)
    baseline_val = evaluate_recommender(baseline, train_df, val_df, k=10, random_state=args.seed)
    rows, candidates = [], {}
    for centered in (False, True):
        for minimum in (1, 5, 10):
            candidate = CosineItemItemRecommender(minimum, centered)
            candidate.fit(train_df, movies_df)
            for k in (5, 10, 20):
                metrics = evaluate_recommender(candidate, train_df, val_df, k=k, random_state=args.seed)
                rows.append({"mean_centered": centered, "min_ratings": minimum, "k": k, "metrics": metrics})
            # Only retain the two validation-selected representations, not six dense matrices.
            best_variant = select_candidate([r for r in rows if r["mean_centered"] == centered])
            if best_variant["min_ratings"] == minimum:
                candidates[centered] = candidate
    selected = select_candidate(rows)
    model = candidates[selected["mean_centered"]]
    raw_row = select_candidate([r for r in rows if not r["mean_centered"]])
    centered_row = select_candidate([r for r in rows if r["mean_centered"]])
    # Controlled ablation plots share the same threshold/K and validation cohort.
    controlled = {r["mean_centered"]: r["metrics"] for r in rows if r["min_ratings"] == 5 and r["k"] == 10}
    for filename, entries in [("exp1_baseline_vs_cosine.png", {"Count popularity": baseline_val, "Raw cosine": controlled[False]}), ("exp2_raw_vs_mean_centered.png", {"Raw": controlled[False], "Mean-centered": controlled[True]})]:
        labels = ["HitRate@10", "Precision@10", "Recall@10", "Coverage"]
        pd.DataFrame({name: [metric[label] for label in labels] for name, metric in entries.items()}, index=labels).plot.bar(title="Validation (train-only features)", figsize=(9, 5))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename))
        plt.close()
    grid_df = pd.DataFrame([{**{k: r[k] for k in ("mean_centered", "min_ratings", "k")}, "Precision": r["metrics"][f"Precision@{r['k']}"], "Coverage": r["metrics"]["Coverage"]} for r in rows])
    grid_df.to_csv("reports/validation_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for centered, axis in zip((False, True), axes):
        sub = grid_df[grid_df.mean_centered == centered]
        sns.heatmap(sub.pivot(index="min_ratings", columns="k", values="Precision"), annot=True, fmt=".4f", ax=axis)
        axis.set_title("Validation Precision: " + ("Mean-centered" if centered else "Raw"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp3_k_and_min_ratings.png"))
    plt.close()
    if args.development:
        Path("reports/development_validation.json").write_text(json.dumps({"grid": rows, "selected": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Validation-only development complete; final test was not evaluated.")
        return
    config = {"seed": args.seed, "split": {"strategy": "per-user temporal; timestamp then movieId; ties may cross split boundaries", "train": 0.7, "validation": 0.1, "test": 0.2, "randomized": False}, "selection_rule": "Validation Precision@K, coverage, smaller K, raw, smaller min_ratings", "selected": {k: selected[k] for k in ("mean_centered", "min_ratings", "k")}, "evaluation": {"positive_threshold": 3.5, "max_users": None, "unit": "user"}, "python": platform.python_version(), "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "fastapi", "pydantic", "matplotlib", "seaborn")}}
    # Persist selection and artifact BEFORE any final-test metrics exist.
    model.save(os.path.join(MODELS_DIR, "cosine_recommender.pkl"))
    config["artifact_sha256"] = hashlib.sha256(Path(MODELS_DIR, "cosine_recommender.pkl").read_bytes()).hexdigest()
    Path(MODELS_DIR, "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = {"status": "test_started", "config": config, "validation": {"baseline_at_10": baseline_val, "grid": rows}, "test": {}}
    results_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    test_results = {"popularity": evaluate_recommender(baseline, train_df, test_df, k=10, random_state=args.seed), "cosine_raw": evaluate_recommender(candidates[False], train_df, test_df, k=10, random_state=args.seed), "cosine_centered": evaluate_recommender(candidates[True], train_df, test_df, k=10, random_state=args.seed)}
    selected_name = "cosine_centered" if selected["mean_centered"] else "cosine_raw"
    final_metrics = test_results[selected_name] if selected["k"] == 10 else evaluate_recommender(model, train_df, test_df, k=selected["k"], random_state=args.seed)
    test_results["selected"] = final_metrics
    checkpoint.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "test": test_results, "test_comparison": {"k": 10, "raw_min_ratings": raw_row["min_ratings"], "centered_min_ratings": centered_row["min_ratings"], "note": "Thresholds selected independently on validation; controlled ablation is in validation plots."}})
    results_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    analyze_failure_cases(model, movies_df)
    card = {"model_name": "MovieLens Item-Item Cosine Recommender", "version": "2.0.0", "intended_use": {"primary_task": "Item-to-item retrieval; evaluation aggregates five training seeds per user", "out_of_scope": "Rating prediction; sensitive preference inference; causal claims"}, "training_data": {"dataset": "MovieLens latest-small (GroupLens)", "num_ratings_train": len(train_df), "num_movies_indexed": model.metadata["n_movies"], "num_users_indexed": model.metadata["n_users"], "sparsity": model.metadata["sparsity"]}, "technical_specifications": config["selected"], "evaluation_summary": {"validation": checkpoint["validation"], "test": test_results, "selected_model": selected_name, "selection_rule": config["selection_rule"]}, "limitations": [f"Cold-start: fewer than {model.min_movie_ratings} train ratings cannot be served.", "Long-tail and popularity bias; historical preferences may drift.", "Only positive cosine scores are served; unknown release years are excluded when filtering by year.", "Per-user temporal split does not enforce one global prediction date.", "Bootstrap intervals describe user variability; no multiple-seed claims for deterministic splitting."], "ethical_considerations": "Historical recommendations are not absolute movie quality; do not infer sensitive traits.", "reproducibility": {"config": "models/config.json", "data_manifest": "data/processed/data_manifest.json", "results": "reports/experiment_results.json"}}
    Path(MODELS_DIR, "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Final artifacts and independent test results frozen.")


if __name__ == "__main__":
    main()
