import os
import sys
import io

# Đảm bảo in tiếng Việt chuẩn trên Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.chatbot_engine import CineBotNLPEngine
from src.models import CosineItemItemRecommender

print("Loading data...")
movies_df = pd.read_csv("data/processed/movies_clean.csv")
print(f"Loaded {len(movies_df)} movies.")

print("Initializing CineBotNLPEngine...")
bot = CineBotNLPEngine(movies_df)

print("Loading model...")
model = CosineItemItemRecommender.load("models/cosine_recommender.pkl")
print("Model loaded successfully.")

test_queries = [
    "xin chào bạn ơi",
    "gợi ý cho tôi phim giống Toy Story với",
    "tìm phim hoạt hình hay",
    "độ tương đồng cosine là gì",
    "hôm nay tôi buồn quá",
    "tập dữ liệu movielens gồm những gì"
]

for q in test_queries:
    res = bot.process_message(q, model, top_k=3)
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    print(f"INTENT: {res.get('intent')}")
    print(f"REPLY PREVIEW: {res.get('reply')[:120]}...")
    recs = res.get('recommendations', [])
    print(f"RECS COUNT: {len(recs)}")
    for r in recs[:2]:
        print(f"  -> {r.get('title')} (score: {r.get('similarity_score')})")

print("\nALL IN-HOUSE NLP ENGINE TESTS PASSED!")
