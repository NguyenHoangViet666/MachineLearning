"""
data.py - Module quản lý tải, kiểm tra chất lượng và chia tập dữ liệu MovieLens.
Đảm bảo tính tái lập (reproducibility) và chống rò rỉ dữ liệu (data leakage).
"""

import os
import io
import zipfile
import urllib.request
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def download_and_extract_data(url: str = DATA_URL, target_dir: str = RAW_DATA_DIR) -> str:
    """Tải file zip từ nguồn GroupLens và giải nén vào thư mục raw."""
    os.makedirs(target_dir, exist_ok=True)
    dataset_path = os.path.join(target_dir, "ml-latest-small")
    
    ratings_file = os.path.join(dataset_path, "ratings.csv")
    movies_file = os.path.join(dataset_path, "movies.csv")
    
    if os.path.exists(ratings_file) and os.path.exists(movies_file):
        logger.info(f"Dữ liệu đã tồn tại tại: {dataset_path}")
        return dataset_path

    logger.info(f"Đang tải dữ liệu từ {url}...")
    import requests
    import certifi
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, verify=certifi.where(), timeout=60)
        response.raise_for_status()
        zip_data = response.content
    except Exception as e:
        logger.warning(f"Tải có SSL gặp lỗi ({e}), đang thử chế độ không verify...")
        response = requests.get(url, headers=headers, verify=False, timeout=60)
        response.raise_for_status()
        zip_data = response.content

    logger.info(f"Tải thành công ({len(zip_data) / 1024 / 1024:.2f} MB). Đang giải nén...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        z.extractall(target_dir)

    logger.info(f"Giải nén hoàn tất vào {dataset_path}")
    return dataset_path


def load_raw_data(dataset_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc dữ liệu ratings và movies."""
    if dataset_path is None:
        dataset_path = os.path.join(RAW_DATA_DIR, "ml-latest-small")
    
    ratings_path = os.path.join(dataset_path, "ratings.csv")
    movies_path = os.path.join(dataset_path, "movies.csv")
    
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        dataset_path = download_and_extract_data(DATA_URL, RAW_DATA_DIR)
        ratings_path = os.path.join(dataset_path, "ratings.csv")
        movies_path = os.path.join(dataset_path, "movies.csv")

    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    return ratings_df, movies_df


def inspect_data_quality(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Kiểm tra tính toàn vẹn dữ liệu: missing, duplicates, schema, range."""
    report = {
        "n_ratings": int(len(ratings_df)),
        "n_users": int(ratings_df["userId"].nunique()),
        "n_movies_rated": int(ratings_df["movieId"].nunique()),
        "n_total_movies": int(len(movies_df)),
        "rating_min": float(ratings_df["rating"].min()),
        "rating_max": float(ratings_df["rating"].max()),
        "rating_mean": float(ratings_df["rating"].mean()),
        "ratings_missing": int(ratings_df.isnull().sum().sum()),
        "movies_missing": int(movies_df.isnull().sum().sum()),
        "ratings_duplicates": int(ratings_df.duplicated(subset=["userId", "movieId"]).sum()),
    }
    
    logger.info("=== BÁO CÁO CHẤT LƯỢNG DỮ LIỆU ===")
    for k, v in report.items():
        logger.info(f" - {k}: {v}")
    
    if report["ratings_duplicates"] > 0:
        logger.warning(f"Phát hiện {report['ratings_duplicates']} dòng trùng lặp cặp (userId, movieId)!")
    if report["ratings_missing"] > 0:
        logger.warning(f"Phát hiện {report['ratings_missing']} giá trị null trong ratings!")
        
    return report


def split_train_val_test(
    ratings_df: pd.DataFrame, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia tập dữ liệu theo từng User (Stratified Temporal/Per-user Split) để chống Data Leakage.
    Mỗi người dùng sẽ có:
      - (1 - val_ratio - test_ratio) tương tác trong tập Train
      - val_ratio tương tác trong tập Validation
      - test_ratio tương tác trong tập Test (thường là các đánh giá mới nhất theo thời gian).
    """
    logger.info(f"Đang chia Train/Val/Test với tỷ lệ test={test_ratio}, val={val_ratio} (random_state={random_state})...")
    
    # Sắp xếp theo userId và timestamp
    sorted_df = ratings_df.sort_values(by=["userId", "timestamp"]).copy()
    
    train_rows = []
    val_rows = []
    test_rows = []
    
    grouped = sorted_df.groupby("userId")
    for user_id, group in grouped:
        n_items = len(group)
        # Nếu user có quá ít rating (ví dụ < 5), giữ lại cho train để đảm bảo cold-start
        if n_items < 5:
            train_rows.append(group)
            continue
            
        n_test = max(1, int(n_items * test_ratio))
        n_val = max(1, int(n_items * val_ratio))
        n_train = n_items - n_test - n_val
        
        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train:n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val:])
        
    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    
    logger.info(f"Chia tập thành công: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Kiểm tra rò rỉ: đảm bảo không có cặp (user, movie) nào vừa ở test vừa ở train
    train_pairs = set(zip(train_df["userId"], train_df["movieId"]))
    test_pairs = set(zip(test_df["userId"], test_df["movieId"]))
    leakage = train_pairs.intersection(test_pairs)
    assert len(leakage) == 0, f"DATA LEAKAGE DETECTED! {len(leakage)} cặp bị trùng lặp!"
    logger.info("Xác nhận: KHÔNG CÓ RÒ RỈ DỮ LIỆU giữa Train và Test!")
    
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    movies_df: pd.DataFrame,
    target_dir: str = PROCESSED_DATA_DIR
):
    """Lưu các tệp dữ liệu đã phân chia và tiền xử lý vào thư mục data/processed."""
    os.makedirs(target_dir, exist_ok=True)
    train_df.to_csv(os.path.join(target_dir, "train_ratings.csv"), index=False)
    val_df.to_csv(os.path.join(target_dir, "val_ratings.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, "test_ratings.csv"), index=False)
    movies_df.to_csv(os.path.join(target_dir, "movies_clean.csv"), index=False)
    logger.info(f"Đã lưu các file dữ liệu vào: {target_dir}")


def main():
    logger.info("Bắt đầu quy trình chuẩn bị dữ liệu...")
    dataset_path = download_and_extract_data()
    ratings_df, movies_df = load_raw_data(dataset_path)
    inspect_data_quality(ratings_df, movies_df)
    train_df, val_df, test_df = split_train_val_test(ratings_df, test_ratio=0.2, val_ratio=0.1, random_state=42)
    save_processed_data(train_df, val_df, test_df, movies_df)
    logger.info("Quy trình chuẩn bị dữ liệu hoàn tất xuất sắc!")


if __name__ == "__main__":
    main()
