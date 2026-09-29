"""
main.py - FastAPI Web Application & Recommendation Serving API.
Tách biệt hoàn toàn Offline Training và Online Serving.
Chỉ nạp Model Artifact và Metadata đã lưu, không train lại khi có request.
"""

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Thêm project root vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CosineItemItemRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MovieLens Movie Recommendation System API",
    description="Hệ thống gợi ý phim tương đồng sử dụng Vector và Cosine Similarity",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount thư mục static và reports (để hiển thị biểu đồ trên giao diện)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Biến toàn cục nạp model và metadata khi khởi động server
MODEL: Optional[CosineItemItemRecommender] = None
MOVIES_DF: Optional[Any] = None
GENRES_LIST: List[str] = []


def load_assets():
    """Nạp Model và dữ liệu phim đã được lưu sẵn (Offline Precomputed)."""
    global MODEL, MOVIES_DF, GENRES_LIST
    
    # 1. Nạp danh sách phim
    movies_path = os.path.join(PROCESSED_DATA_DIR, "movies_clean.csv")
    if os.path.exists(movies_path):
        import pandas as pd
        MOVIES_DF = pd.read_csv(movies_path)
        all_genres = set()
        for g_str in MOVIES_DF["genres"].dropna():
            for g in g_str.split("|"):
                if g.strip() and g.strip() != "(no genres listed)":
                    all_genres.add(g.strip())
        GENRES_LIST = sorted(list(all_genres))
        logger.info(f"Đã nạp {len(MOVIES_DF)} phim và {len(GENRES_LIST)} thể loại.")
    else:
        logger.warning(f"Chưa tìm thấy {movies_path}. Vui lòng chạy train.py trước!")

    # 2. Nạp Model đã lưu
    model_path = os.path.join(MODELS_DIR, "cosine_recommender.pkl")
    if os.path.exists(model_path):
        MODEL = CosineItemItemRecommender.load(model_path)
        logger.info("Model CosineItemItemRecommender đã được nạp thành công.")
    else:
        logger.warning(f"Chưa tìm thấy {model_path}. Server đang chạy ở chế độ chờ model.")


@app.on_event("startup")
def on_startup():
    load_assets()


# Pydantic Schemas
class MovieRecommendationItem(BaseModel):
    movieId: int = Field(..., description="ID của bộ phim được gợi ý")
    title: str = Field(..., description="Tên phim")
    genres: str = Field(..., description="Các thể loại của phim")
    similarity_score: float = Field(..., description="Điểm tương đồng Cosine [0, 1]")
    rating_count: int = Field(..., description="Số lượt đánh giá trong tập train")
    rating_mean: float = Field(..., description="Điểm rating trung bình")
    reason: str = Field(..., description="Lý do gợi ý")


class RecommendationResponse(BaseModel):
    movie_id: int
    query_movie_title: str
    rating_count: int
    status: str
    warning: Optional[str] = None
    k: int
    filter_genre: Optional[str] = None
    recommendations: List[MovieRecommendationItem]


# Endpoints
@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Trang chủ ứng dụng Web với 3 màn hình tương tác."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"genres": GENRES_LIST}
    )


@app.get("/api/movies")
async def search_movies(
    query: Optional[str] = Query(None, description="Từ khóa tìm kiếm theo tên phim"),
    genre: Optional[str] = Query(None, description="Lọc theo thể loại"),
    limit: int = Query(20, ge=1, le=100)
):
    """Tìm kiếm phim hỗ trợ thanh tìm kiếm và autocomplete."""
    if MOVIES_DF is None:
        raise HTTPException(status_code=503, detail="Dữ liệu phim chưa sẵn sàng.")
    
    df = MOVIES_DF.copy()
    if query:
        df = df[df["title"].str.contains(query, case=False, na=False)]
    if genre and genre.lower() != "all":
        df = df[df["genres"].str.contains(genre, case=False, na=False)]
    
    results = df.head(limit).to_dict(orient="records")
    return {"total": len(results), "movies": results}


@app.get("/api/movies/{movie_id}")
async def get_movie_detail(movie_id: int):
    """Lấy chi tiết một bộ phim theo ID."""
    if MOVIES_DF is None:
        raise HTTPException(status_code=503, detail="Dữ liệu phim chưa sẵn sàng.")
    
    match = MOVIES_DF[MOVIES_DF["movieId"] == movie_id]
    if len(match) == 0:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phim với ID {movie_id}")
    
    movie_info = match.iloc[0].to_dict()
    # Thêm thống kê số rating nếu có
    if MODEL and movie_id in MODEL.movie_stats:
        stats = MODEL.movie_stats[movie_id]
        movie_info["rating_count"] = stats.get("rating_count", 0)
        movie_info["rating_mean"] = round(stats.get("rating_mean", 0.0), 2)
    else:
        movie_info["rating_count"] = 0
        movie_info["rating_mean"] = 0.0
        
    return movie_info


@app.get("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    movie_id: int = Query(..., description="ID của bộ phim người dùng đang quan tâm"),
    k: int = Query(10, ge=1, le=50, description="Số lượng phim tương tự cần gợi ý"),
    genre: Optional[str] = Query(None, description="Bộ lọc thể loại tùy chọn")
):
    """
    API chính: Gợi ý Top-K phim tương tự dựa trên Cosine Similarity.
    Xác thực đầu vào, kiểm tra phim hiếm/cold-start, trả về điểm tương đồng và lý do.
    """
    global MODEL
    if MODEL is None:
        load_assets()
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Mô hình gợi ý chưa được huấn luyện hoặc nạp.")

    if MOVIES_DF is not None:
        exists = len(MOVIES_DF[MOVIES_DF["movieId"] == movie_id]) > 0
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phim ID {movie_id} không tồn tại trong hệ thống.")

    result = MODEL.recommend(movie_id=movie_id, top_k=k, filter_genre=genre)
    
    return {
        "movie_id": movie_id,
        "query_movie_title": result.get("query_movie_title", ""),
        "rating_count": result.get("rating_count", 0),
        "status": result.get("status", "success"),
        "warning": result.get("warning"),
        "k": k,
        "filter_genre": genre,
        "recommendations": result.get("recommendations", [])
    }


@app.get("/api/model-card")
async def get_model_card():
    """Trả về Model Card và thông số đánh giá hệ thống."""
    card_path = os.path.join(MODELS_DIR, "model_card.json")
    if not os.path.exists(card_path):
        raise HTTPException(status_code=404, detail="Model Card chưa được tạo.")
    with open(card_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/failure-cases")
async def get_failure_cases():
    """Trả về danh sách 10 ca truy vấn thất bại / ngoại biên (Thí nghiệm 4)."""
    fail_path = os.path.join(REPORTS_DIR, "exp4_failure_analysis.json")
    if not os.path.exists(fail_path):
        raise HTTPException(status_code=404, detail="Dữ liệu phân tích lỗi chưa có.")
    with open(fail_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
