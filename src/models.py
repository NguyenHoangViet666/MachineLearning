"""
models.py - Các mô hình gợi ý phim: Baseline (Popularity) và Cosine Item-Item.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from src.features import InteractionMatrixBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PopularityRecommender:
    """
    Baseline Model: Gợi ý các bộ phim phổ biến nhất (nhiều lượt đánh giá & điểm cao nhất)
    từ tập Train. Không phụ thuộc vào nội dung hay tương tác cá nhân hóa.
    """
    def __init__(self, min_ratings: int = 10):
        self.min_ratings = min_ratings
        self.popular_movies: List[Dict[str, Any]] = []
        self.movies_df: Optional[pd.DataFrame] = None

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame):
        """Tính toán bảng xếp hạng độ phổ biến từ tập Train."""
        self.movies_df = movies_df.copy()
        
        # Thống kê số lượt đánh giá và điểm trung bình cho mỗi phim
        stats = train_ratings.groupby("movieId").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        ).reset_index()

        # Áp dụng công thức weighted rating (IMDb / Bayesian damping)
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        # Trong đó: v = rating_count, m = min_ratings, R = rating_mean, C = mean rating của toàn tập
        C = train_ratings["rating"].mean()
        m = self.min_ratings
        
        stats["score"] = (stats["rating_count"] / (stats["rating_count"] + m)) * stats["rating_mean"] + \
                         (m / (stats["rating_count"] + m)) * C
        
        stats = stats.sort_values(by="score", ascending=False)
        stats = stats.merge(movies_df, on="movieId", how="left")
        
        self.popular_movies = stats.to_dict(orient="records")
        logger.info(f"Baseline Popularity fitted thành công với {len(self.popular_movies)} phim.")

    def recommend(self, movie_id: Optional[int] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Gợi ý Top-K phim phổ biến nhất (loại bỏ phim input nếu có)."""
        recs = []
        for item in self.popular_movies:
            if movie_id is not None and item["movieId"] == movie_id:
                continue
            recs.append({
                "movieId": int(item["movieId"]),
                "title": item.get("title", ""),
                "genres": item.get("genres", ""),
                "score": float(item["score"]),
                "rating_count": int(item["rating_count"]),
                "rating_mean": float(item["rating_mean"]),
                "reason": "Top phim phổ biến và được đánh giá cao nhất trên hệ thống."
            })
            if len(recs) >= top_k:
                break
        return recs


class CosineItemItemRecommender:
    """
    Mô hình chính: Item-Item Collaborative Filtering sử dụng Cosine Similarity.
    Hỗ trợ cả ma trận Raw Ratings và Mean-centered Ratings (Adjusted Cosine).
    """
    def __init__(
        self, 
        min_movie_ratings: int = 5, 
        normalize_mean_centered: bool = True
    ):
        self.min_movie_ratings = min_movie_ratings
        self.normalize_mean_centered = normalize_mean_centered
        self.builder = InteractionMatrixBuilder(min_movie_ratings=min_movie_ratings)
        self.similarity_matrix: Optional[np.ndarray] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.movie_stats: Dict[int, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame):
        """Huấn luyện mô hình: tạo ma trận và tính toán ma trận Cosine Similarity."""
        self.movies_df = movies_df.copy()
        
        # Lưu thống kê số lượng đánh giá của từng phim trên tập train
        counts = train_ratings.groupby("movieId").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        ).to_dict(orient="index")
        self.movie_stats = counts

        # Xây dựng ma trận thưa Item-User
        item_user_csr, meta = self.builder.fit_transform(
            train_ratings, 
            normalize_mean_centered=self.normalize_mean_centered
        )
        self.metadata = meta

        logger.info(f"Đang tính toán Cosine Similarity matrix ({item_user_csr.shape[0]}x{item_user_csr.shape[0]})...")
        
        # Chuẩn hóa L2 từng hàng và nhân ma trận thưa để tính cosine
        from sklearn.preprocessing import normalize
        normalized_csr = normalize(item_user_csr, norm="l2", axis=1)
        # Cosine similarity giữa các item: S = X_norm * X_norm^T
        self.similarity_matrix = (normalized_csr @ normalized_csr.T).toarray()
        
        # Đặt đường chéo chính (tự tương đồng với chính nó) bằng 0 để tránh recommend lại chính nó
        np.fill_diagonal(self.similarity_matrix, 0.0)
        
        logger.info("Hoàn tất tính toán Cosine Similarity matrix.")

    def recommend(
        self, 
        movie_id: int, 
        top_k: int = 10, 
        filter_genre: Optional[str] = None,
        filter_genres: Optional[List[str]] = None,
        genre_match_mode: str = "any"
    ) -> Dict[str, Any]:
        """
        Gợi ý Top-K phim tương đồng với movie_id chỉ định.
        Kèm kiểm tra dữ liệu thưa / cảnh báo cold-start và giải thích lý do.
        Hỗ trợ lọc nhiều thể loại theo 3 chế độ:
        - 'any': Khớp ít nhất 1 thể loại (OR)
        - 'all': Khớp đầy đủ tất cả thể loại (AND)
        - 'exclude': Loại trừ phim chứa các thể loại này (NOT)
        """
        if movie_id not in self.builder.movie2idx:
            # Trường hợp phim không có trong train hoặc bị loại do quá ít rating (< min_ratings)
            count = self.movie_stats.get(movie_id, {}).get("rating_count", 0)
            return {
                "movie_id": movie_id,
                "status": "warning",
                "message": f"Phim ID {movie_id} chỉ có {count} lượt đánh giá trong tập train (ngưỡng tối thiểu là {self.min_movie_ratings}). Không đủ dữ liệu để tính tương đồng.",
                "recommendations": []
            }

        item_idx = self.builder.movie2idx[movie_id]
        scores = self.similarity_matrix[item_idx]
        
        # Chuẩn hóa danh sách thể loại cần lọc
        active_genres: List[str] = []
        if filter_genres:
            for g in filter_genres:
                if isinstance(g, str):
                    for sub_g in g.split(","):
                        sub_clean = sub_g.strip()
                        if sub_clean and sub_clean.lower() != "all" and sub_clean not in active_genres:
                            active_genres.append(sub_clean)
        elif filter_genre and filter_genre.strip().lower() != "all":
            for sub_g in filter_genre.split(","):
                sub_clean = sub_g.strip()
                if sub_clean and sub_clean.lower() != "all" and sub_clean not in active_genres:
                    active_genres.append(sub_clean)

        genre_match_mode = (genre_match_mode or "any").lower().strip()
        if genre_match_mode not in ("any", "all", "exclude"):
            genre_match_mode = "any"

        # Lấy các chỉ số có điểm cao nhất
        ranked_indices = np.argsort(scores)[::-1]
        
        query_movie_row = self.movies_df[self.movies_df["movieId"] == movie_id]
        query_title = query_movie_row["title"].values[0] if len(query_movie_row) > 0 else f"Phim {movie_id}"
        query_genres = set(query_movie_row["genres"].values[0].split("|")) if len(query_movie_row) > 0 else set()
        
        query_rating_count = self.movie_stats.get(movie_id, {}).get("rating_count", 0)
        warning = None
        if query_rating_count < 10:
            warning = f"Cảnh báo: Phim '{query_title}' có ít đánh giá ({query_rating_count} lượt), độ tin cậy của gợi ý có thể bị ảnh hưởng."

        recommendations = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
                
            sim_movie_id = self.builder.idx2movie[idx]
            movie_meta = self.movies_df[self.movies_df["movieId"] == sim_movie_id]
            if len(movie_meta) == 0:
                continue
                
            m_title = movie_meta["title"].values[0]
            m_genres = movie_meta["genres"].values[0]
            m_genres_set = set(m_genres.split("|")) if isinstance(m_genres, str) else set()
            m_genres_lower = {g.strip().lower() for g in m_genres_set}
            
            # Lọc theo thể loại và chế độ (any / all / exclude)
            if active_genres:
                active_lower = [g.lower() for g in active_genres]
                if genre_match_mode == "all":
                    if not all(req in m_genres_lower for req in active_lower):
                        continue
                elif genre_match_mode == "exclude":
                    if any(req in m_genres_lower for req in active_lower):
                        continue
                else:  # any
                    if not any(req in m_genres_lower for req in active_lower):
                        continue

            # Lý do tương đồng & thông tin khớp thể loại
            common_genres = query_genres.intersection(m_genres_set)
            matched_filters = [g for g in active_genres if g.lower() in m_genres_lower] if active_genres else []
            
            if active_genres:
                if genre_match_mode == "exclude":
                    filter_info = f"đã loại trừ thể loại ({', '.join(active_genres)})"
                elif genre_match_mode == "all":
                    filter_info = f"đủ tất cả thể loại ({', '.join(matched_filters)})"
                else:
                    filter_info = f"khớp thể loại ({', '.join(matched_filters)})"

                if common_genres:
                    genre_reason = f"cùng thể loại ({', '.join(common_genres)}) và {filter_info}"
                else:
                    genre_reason = f"{filter_info}"
            elif common_genres:
                genre_reason = f"cùng thể loại ({', '.join(common_genres)})"
            else:
                genre_reason = "thể loại bổ trợ"

            reason = f"Được nhiều người dùng cùng yêu thích giống như '{query_title}', {genre_reason}."

            recommendations.append({
                "movieId": int(sim_movie_id),
                "title": m_title,
                "genres": m_genres,
                "similarity_score": round(score, 4),
                "rating_count": int(self.movie_stats.get(sim_movie_id, {}).get("rating_count", 0)),
                "rating_mean": round(float(self.movie_stats.get(sim_movie_id, {}).get("rating_mean", 0.0)), 2),
                "reason": reason
            })
            
            if len(recommendations) >= top_k:
                break

        return {
            "movie_id": movie_id,
            "query_movie_title": query_title,
            "rating_count": query_rating_count,
            "status": "success",
            "warning": warning,
            "k": top_k,
            "filter_genre": active_genres[0] if len(active_genres) == 1 else (", ".join(active_genres) if active_genres else None),
            "filter_genres": active_genres,
            "genre_match_mode": genre_match_mode,
            "recommendations": recommendations
        }

    def save(self, filepath: str):
        """Lưu toàn bộ model và ma trận similarity vào file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Đã lưu model vào {filepath}")

    @staticmethod
    def load(filepath: str) -> "CosineItemItemRecommender":
        """Nạp model từ file."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Đã nạp model từ {filepath}")
        return model
