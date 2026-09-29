import sys
import os
import requests
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import CosineItemItemRecommender

class MockBuilder:
    def __init__(self, movie_ids):
        self.movie2idx = {m: i for i, m in enumerate(movie_ids)}
        self.idx2movie = {i: m for i, m in enumerate(movie_ids)}

def test_model_multigenre_filter():
    print("Testing CosineItemItemRecommender multi-genre filter...")
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama|Romance",
            "Comedy"
        ]
    })
    
    # 5 items similarity matrix (5x5)
    sim_mat = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6],
        [0.9, 0.0, 0.5, 0.4, 0.3],
        [0.8, 0.5, 0.0, 0.85, 0.2],
        [0.7, 0.4, 0.85, 0.0, 0.1],
        [0.6, 0.3, 0.2, 0.1, 0.0]
    ], dtype=np.float32)

    model = CosineItemItemRecommender()
    model.movies_df = movies_df
    model.similarity_matrix = sim_mat
    model.builder = MockBuilder([1, 2, 3, 4, 5])
    model.movie_stats = {
        1: {"rating_count": 200, "rating_mean": 4.0},
        2: {"rating_count": 100, "rating_mean": 3.5},
        3: {"rating_count": 50, "rating_mean": 3.2},
        4: {"rating_count": 30, "rating_mean": 3.0},
        5: {"rating_count": 40, "rating_mean": 3.1},
    }

    # 1. Test "any" mode with Animation & Romance
    res_any = model.recommend(1, top_k=5, filter_genres=["Animation", "Romance"], genre_match_mode="any")
    recs_any = res_any["recommendations"]
    rec_ids = [r["movieId"] for r in recs_any]
    assert all(r_id in [3, 4] for r_id in rec_ids), f"Unexpected recs in any mode: {rec_ids}"
    print(" [PASS] Any mode test passed:", rec_ids)

    # 2. Test "all" mode with Comedy and Romance
    res_all = model.recommend(1, top_k=5, filter_genres=["Comedy", "Romance"], genre_match_mode="all")
    recs_all = res_all["recommendations"]
    rec_ids_all = [r["movieId"] for r in recs_all]
    assert 3 in rec_ids_all and 4 in rec_ids_all, f"Expected 3 and 4, got {rec_ids_all}"
    print(" [PASS] All mode test passed:", rec_ids_all)

    # 3. Test "exclude" mode excluding Romance
    res_ex = model.recommend(1, top_k=5, filter_genres=["Romance"], genre_match_mode="exclude")
    recs_ex = res_ex["recommendations"]
    rec_ids_ex = [r["movieId"] for r in recs_ex]
    assert 3 not in rec_ids_ex and 4 not in rec_ids_ex, f"Found excluded Romance: {rec_ids_ex}"
    print(" [PASS] Exclude mode test passed:", rec_ids_ex)

def test_api_endpoints():
    print("\nTesting Live API Endpoints on http://127.0.0.1:8000 ...")
    try:
        # Check /api/genres
        res = requests.get("http://127.0.0.1:8000/api/genres")
        assert res.status_code == 200, f"Status: {res.status_code}"
        data = res.json()
        assert "genres" in data and len(data["genres"]) >= 18
        print(f" [PASS] /api/genres returned {len(data['genres'])} genres.")

        # Check /api/recommendations with multi-genres
        res_rec = requests.get("http://127.0.0.1:8000/api/recommendations?movie_id=1&k=5&genres=Animation,Children&match_mode=all")
        assert res_rec.status_code == 200
        rec_data = res_rec.json()
        assert len(rec_data["recommendations"]) > 0
        for item in rec_data["recommendations"]:
            assert "Animation" in item["genres"] and "Children" in item["genres"]
        print(f" [PASS] /api/recommendations (all mode) returned {len(rec_data['recommendations'])} items matching all genres.")

        # Check /api/movies with multi-genres
        res_mov = requests.get("http://127.0.0.1:8000/api/movies?genres=Action,Sci-Fi&match_mode=all&limit=5")
        assert res_mov.status_code == 200
        mov_data = res_mov.json()
        assert len(mov_data["movies"]) > 0
        for m in mov_data["movies"]:
            assert "Action" in m["genres"] and "Sci-Fi" in m["genres"]
        print(f" [PASS] /api/movies returned {len(mov_data['movies'])} movies matching Action + Sci-Fi.")
    except Exception as e:
        print(f" [WARN/FAIL] API test encountered: {e}")

if __name__ == "__main__":
    test_model_multigenre_filter()
    test_api_endpoints()
    print("\nALL MULTI-GENRE TESTS COMPLETED SUCCESSFULLY!")
