"""
features.py - Module tạo ma trận đặc trưng User-Item, chuẩn hóa và xử lý ma trận thưa.
"""

import logging
from typing import Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class InteractionMatrixBuilder:
    """
    Xây dựng ma trận tương tác Item-User từ dữ liệu ratings.
    Mỗi hàng biểu diễn một bộ phim (Item), mỗi cột biểu diễn một người dùng (User).
    """
    def __init__(self, min_movie_ratings: int = 1):
        self.min_movie_ratings = min_movie_ratings
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.user_means: Dict[int, float] = {}

    def fit_transform(
        self, 
        ratings_df: pd.DataFrame, 
        normalize_mean_centered: bool = False
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """
        Tạo ma trận thưa Item-User (phim x người dùng).
        Nếu normalize_mean_centered=True, rating sẽ được trừ đi trung bình của từng user (Adjusted Cosine).
        """
        # 1. Lọc phim theo số lượng rating tối thiểu (chỉ học từ train)
        movie_counts = ratings_df["movieId"].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index.values
        filtered_df = ratings_df[ratings_df["movieId"].isin(valid_movies)].copy()

        logger.info(
            f"Lọc phim với min_ratings={self.min_movie_ratings}: "
            f"Từ {ratings_df['movieId'].nunique()} còn {len(valid_movies)} phim."
        )

        # 2. Xây dựng index mapping
        unique_movies = np.sort(valid_movies)
        unique_users = np.sort(filtered_df["userId"].unique())

        self.movie2idx = {m_id: i for i, m_id in enumerate(unique_movies)}
        self.idx2movie = {i: m_id for i, m_id in enumerate(unique_movies)}
        self.user2idx = {u_id: i for i, u_id in enumerate(unique_users)}
        self.idx2user = {i: u_id for i, u_id in enumerate(unique_users)}

        # 3. Tính trung bình rating của từng người dùng trên tập train
        self.user_means = filtered_df.groupby("userId")["rating"].mean().to_dict()

        row_indices = filtered_df["movieId"].map(self.movie2idx).values
        col_indices = filtered_df["userId"].map(self.user2idx).values
        
        if normalize_mean_centered:
            # Trừ điểm trung bình của user: r'_ui = r_ui - mu_u
            ratings_values = (filtered_df["rating"] - filtered_df["userId"].map(self.user_means)).values
        else:
            ratings_values = filtered_df["rating"].values

        n_rows = len(unique_movies)
        n_cols = len(unique_users)

        # Tạo ma trận thưa CSR: kích thước (n_movies, n_users)
        item_user_matrix = csr_matrix(
            (ratings_values, (row_indices, col_indices)), 
            shape=(n_rows, n_cols), 
            dtype=np.float32
        )

        metadata = {
            "n_movies": n_rows,
            "n_users": n_cols,
            "n_entries": item_user_matrix.nnz,
            "sparsity": 1.0 - (item_user_matrix.nnz / (n_rows * n_cols)),
            "normalize_mean_centered": normalize_mean_centered,
            "min_movie_ratings": self.min_movie_ratings
        }
        
        logger.info(f"Đã tạo ma trận Item-User {n_rows}x{n_cols} với độ thưa: {metadata['sparsity']*100:.2f}%")
        return item_user_matrix, metadata
