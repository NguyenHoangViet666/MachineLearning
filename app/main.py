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
from src.chatbot_engine import CineBotNLPEngine

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
CHATBOT: Optional[CineBotNLPEngine] = None


def load_assets():
    """Nạp Model và dữ liệu phim đã được lưu sẵn (Offline Precomputed)."""
    global MODEL, MOVIES_DF, GENRES_LIST, CHATBOT
    
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

    # 3. Nạp CineBot In-House NLP Engine
    if MOVIES_DF is not None:
        CHATBOT = CineBotNLPEngine(movies_df=MOVIES_DF)
        logger.info("CineBot In-House NLP Engine đã sẵn sàng phục vụ.")


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
    filter_genres: Optional[List[str]] = None
    genre_match_mode: Optional[str] = "any"
    recommendations: List[MovieRecommendationItem]


class ChatRequest(BaseModel):
    message: str = Field(..., description="Câu hỏi hoặc yêu cầu của người dùng")
    k: int = Field(5, ge=1, le=20, description="Số lượng phim gợi ý tối đa")


class ChatResponse(BaseModel):
    intent: str = Field(..., description="Ý định được phân loại bởi NLP Engine")
    reply: str = Field(..., description="Nội dung phản hồi Markdown tự nhiên")
    recommendations: List[Dict[str, Any]] = Field(default=[], description="Danh sách phim gợi ý đính kèm")
    referenced_movie: Optional[Dict[str, Any]] = Field(default=None, description="Thông tin phim được trích xuất")
    target_genre: Optional[str] = Field(default=None, description="Thể loại được trích xuất")


def parse_genres_param(
    genres: Optional[List[str]] = None, 
    genre: Optional[str] = None
) -> List[str]:
    """Chuẩn hóa danh sách thể loại từ query parameters (danh sách hoặc chuỗi phân tách dấu phẩy)."""
    result: List[str] = []
    if genres:
        for g in genres:
            if isinstance(g, str):
                for item in g.split(","):
                    clean = item.strip()
                    if clean and clean.lower() != "all" and clean not in result:
                        result.append(clean)
    if genre and genre.strip().lower() != "all":
        for item in genre.split(","):
            clean = item.strip()
            if clean and clean.lower() != "all" and clean not in result:
                result.append(clean)
    return result


# Endpoints
@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Trang chủ ứng dụng Web với 3 màn hình tương tác."""
    from collections import Counter
    counts = Counter()
    if MOVIES_DF is not None:
        for g_str in MOVIES_DF["genres"].dropna():
            for g in g_str.split("|"):
                g = g.strip()
                if g and g != "(no genres listed)":
                    counts[g] += 1
    genre_items = [{"name": g, "count": counts.get(g, 0)} for g in GENRES_LIST]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"genres": GENRES_LIST, "genre_items": genre_items}
    )


@app.get("/api/genres")
async def get_genres_list():
    """Lấy danh sách thể loại kèm số lượng phim tương ứng."""
    if MOVIES_DF is None:
        raise HTTPException(status_code=503, detail="Dữ liệu phim chưa sẵn sàng.")
    from collections import Counter
    counts = Counter()
    for g_str in MOVIES_DF["genres"].dropna():
        for g in g_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                counts[g] += 1
    genre_data = [{"genre": g, "count": counts.get(g, 0)} for g in GENRES_LIST]
    return {"genres": genre_data}


@app.get("/api/movies")
async def search_movies(
    query: Optional[str] = Query(None, description="Từ khóa tìm kiếm theo tên phim"),
    genre: Optional[str] = Query(None, description="Lọc theo thể loại (chuỗi đơn hoặc phân tách bởi dấu phẩy)"),
    genres: Optional[List[str]] = Query(None, description="Danh sách nhiều thể loại cần lọc"),
    match_mode: str = Query("any", description="Chế độ kết hợp: 'any' (khớp 1 thể loại), 'all' (đủ tất cả), 'exclude' (loại trừ)"),
    limit: int = Query(24, ge=1, le=100)
):
    """Tìm kiếm phim hỗ trợ thanh tìm kiếm, autocomplete và kết hợp lọc nhiều thể loại (ANY / ALL / EXCLUDE)."""
    if MOVIES_DF is None:
        raise HTTPException(status_code=503, detail="Dữ liệu phim chưa sẵn sàng.")
    
    genres_list = parse_genres_param(genres=genres, genre=genre)
    match_mode_clean = match_mode.lower() if match_mode in ("any", "all", "exclude") else "any"

    df = MOVIES_DF.copy()
    if query:
        df = df[df["title"].str.contains(query, case=False, na=False)]
    
    if genres_list:
        genres_lower = [g.lower() for g in genres_list]
        def match_genres(g_str):
            if not isinstance(g_str, str):
                return False
            m_set = {x.strip().lower() for x in g_str.split("|")}
            if match_mode_clean == "all":
                return all(req in m_set for req in genres_lower)
            elif match_mode_clean == "exclude":
                return not any(req in m_set for req in genres_lower)
            else:  # any
                return any(req in m_set for req in genres_lower)
        df = df[df["genres"].apply(match_genres)]
    
    records = df.head(limit).to_dict(orient="records")
    for r in records:
        m_id = r.get("movieId")
        if MODEL and m_id in MODEL.movie_stats:
            stats = MODEL.movie_stats[m_id]
            r["rating_count"] = stats.get("rating_count", 0)
            r["rating_mean"] = round(stats.get("rating_mean", 0.0), 2)
        else:
            r["rating_count"] = 0
            r["rating_mean"] = 0.0

    return {
        "total": len(records), 
        "movies": records,
        "filter_genres": genres_list,
        "match_mode": match_mode_clean
    }


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
    genre: Optional[str] = Query(None, description="Bộ lọc thể loại tùy chọn (chuỗi đơn hoặc phân tách bởi dấu phẩy)"),
    genres: Optional[List[str]] = Query(None, description="Danh sách nhiều thể loại cần kết hợp lọc"),
    match_mode: str = Query("any", description="Chế độ kết hợp: 'any' (khớp 1 thể loại), 'all' (đủ tất cả), 'exclude' (loại trừ)")
):
    """
    API chính: Gợi ý Top-K phim tương tự dựa trên Cosine Similarity.
    Xác thực đầu vào, kiểm tra phim hiếm/cold-start, trả về điểm tương đồng và lý do.
    Hỗ trợ kết hợp lọc nhiều thể loại: any (OR), all (AND), exclude (loại trừ).
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

    genres_list = parse_genres_param(genres=genres, genre=genre)
    match_mode_clean = match_mode.lower() if match_mode in ("any", "all", "exclude") else "any"

    result = MODEL.recommend(
        movie_id=movie_id, 
        top_k=k, 
        filter_genres=genres_list, 
        genre_match_mode=match_mode_clean
    )
    
    return {
        "movie_id": movie_id,
        "query_movie_title": result.get("query_movie_title", ""),
        "rating_count": result.get("rating_count", 0),
        "status": result.get("status", "success"),
        "warning": result.get("warning"),
        "k": k,
        "filter_genre": result.get("filter_genre"),
        "filter_genres": result.get("filter_genres", genres_list),
        "genre_match_mode": result.get("genre_match_mode", match_mode_clean),
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(payload: ChatRequest):
    """
    Endpoint Trợ lý ảo CineBot (In-House NLP Engine).
    Phân loại ý định bằng TF-IDF + Cosine Distance, trích xuất thực thể và
    gọi trực tiếp động cơ CosineItemItemRecommender để gợi ý phim và giải thích.
    Hoàn toàn offline, không gọi API bên thứ ba.
    """
    global CHATBOT, MODEL
    if CHATBOT is None:
        load_assets()
        if CHATBOT is None:
            raise HTTPException(status_code=503, detail="CineBot NLP Engine chưa sẵn sàng.")

    result = CHATBOT.process_message(
        message=payload.message,
        recommender=MODEL,
        top_k=payload.k
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
