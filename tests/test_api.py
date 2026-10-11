import pandas as pd
import pytest
from fastapi.testclient import TestClient
import app.main as serving
from src.models import CosineItemItemRecommender


@pytest.fixture
def client(monkeypatch):
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4], "title": ["Seed (1995)", "Match (2001)", "Old (1980)", "Cold [test] (2002)"], "genres": ["Drama", "Drama|Comedy", "Comedy", "Drama"]})
    ratings = pd.DataFrame({"userId": [1, 2, 1, 2, 1, 2], "movieId": [1, 1, 2, 2, 3, 3], "rating": [4., 5., 4., 5., 3., 4.]})
    model = CosineItemItemRecommender(2, False)
    model.fit(ratings, movies)
    monkeypatch.setattr(serving, "MOVIES_DF", movies)
    monkeypatch.setattr(serving, "MODEL", model)
    monkeypatch.setattr(serving, "GENRES_LIST", ["Drama", "Comedy"])
    return TestClient(serving.app)


@pytest.mark.parametrize("url", ["/api/recommendations", "/api/recommendations?movie_id=abc", "/api/recommendations?movie_id=-1", "/api/recommendations?movie_id=1&k=0", "/api/recommendations?movie_id=1&k=51", "/api/recommendations?movie_id=1&match_mode=bad", "/api/movies?match_mode=bad", "/api/movies?genre=Unknown", "/api/recommendations?movie_id=1&genre=Unknown", "/api/movies?year_min=2001&year_max=1990", "/api/recommendations?movie_id=1&year_min=2001&year_max=1990", "/api/movies?year_min=1800", "/api/movies?year_min=2001.5"])
def test_invalid_input_returns_422(client, url):
    assert client.get(url).status_code == 422


def test_literal_search_never_interprets_regex(client):
    response = client.get("/api/movies", params={"query": "["})
    assert response.status_code == 200
    assert [m["movieId"] for m in response.json()["movies"]] == [4]


def test_year_filters_search_and_recommendation(client):
    search = client.get("/api/movies?year_min=2000&year_max=2001").json()
    assert [m["movieId"] for m in search["movies"]] == [2]
    rec = client.get("/api/recommendations?movie_id=1&year_min=2000&year_max=2001&genre=drama")
    assert rec.status_code == 200
    assert [r["movieId"] for r in rec.json()["recommendations"]] == [2]


def test_cold_start_preserves_warning(client):
    response = client.get("/api/recommendations?movie_id=4")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning" and data["warning"] and data["recommendations"] == []


def test_unknown_movie_and_unavailable_assets(client, monkeypatch):
    assert client.get("/api/recommendations?movie_id=9999").status_code == 404
    assert client.get("/api/movies/9999").status_code == 404
    monkeypatch.setattr(serving, "MODEL", None)
    monkeypatch.setattr(serving, "load_assets", lambda: pytest.fail("Request must not reload/train artifacts"))
    assert client.get("/api/recommendations?movie_id=1").status_code == 503
    monkeypatch.setattr(serving, "MOVIES_DF", None)
    assert client.get("/api/movies").status_code == 503


@pytest.mark.parametrize("message", ["", "   ", "a" * 2001])
def test_chat_rejects_empty_or_excessive_text(client, message):
    assert client.post("/api/chat", json={"message": message}).status_code == 422
