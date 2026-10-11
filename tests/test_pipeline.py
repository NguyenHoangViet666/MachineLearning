import json
import numpy as np
import pandas as pd
import pytest

from src.data import split_train_val_test, clean_and_document
from src.features import InteractionMatrixBuilder
from src.models import PopularityRecommender, CosineItemItemRecommender
from src.evaluate import evaluate_recommender
from src.train import select_candidate, analyze_failure_cases


def test_temporal_split_is_disjoint_and_ordered():
    ratings = pd.DataFrame([(u, m, 4.0, m) for u in (1, 2) for m in range(1, 21)], columns=["userId", "movieId", "rating", "timestamp"])
    train, val, test = split_train_val_test(ratings.sample(frac=1, random_state=9))
    assert (len(train), len(val), len(test)) == (28, 4, 8)
    pairs = [set(zip(part.userId, part.movieId)) for part in (train, val, test)]
    assert not pairs[0] & pairs[1] and not pairs[0] & pairs[2] and not pairs[1] & pairs[2]
    for user in (1, 2):
        assert train[train.userId == user].timestamp.max() <= val[val.userId == user].timestamp.min()
        assert val[val.userId == user].timestamp.max() <= test[test.userId == user].timestamp.min()


def test_preprocessing_uses_training_mean_before_movie_filter():
    ratings = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [5., 1., 3.]})
    builder = InteractionMatrixBuilder(min_movie_ratings=2)
    matrix, meta = builder.fit_transform(ratings, normalize_mean_centered=True)
    assert builder.user_means[1] == 3
    assert 20 not in builder.movie2idx
    assert matrix.toarray()[0, builder.user2idx[1]] == 2


def test_count_popularity_differs_from_rating_quality():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["A", "B"], "genres": ["Drama", "Drama"]})
    ratings = pd.DataFrame({"userId": [1, 2, 3, 1], "movieId": [1, 1, 1, 2], "rating": [1., 1., 1., 5.]})
    model = PopularityRecommender()
    model.fit(ratings, movies)
    assert model.recommend(top_k=1)[0]["movieId"] == 1


def test_quality_audit_records_each_exclusion(tmp_path):
    movies = pd.DataFrame({"movieId": [1], "title": ["A (2000)"], "genres": ["Drama"]})
    ratings = pd.DataFrame([(1, 1, 4., 1), (1, 1, 4., 2), (2, 1, 6., 3), (3, 99, 3., 4)], columns=["userId", "movieId", "rating", "timestamp"])
    (tmp_path / "ratings.csv").write_text("fixture", encoding="utf-8")
    clean, _ = clean_and_document(ratings, movies, tmp_path, tmp_path / "out")
    assert len(clean) == 1
    excluded = pd.read_csv(tmp_path / "out/excluded_ratings.csv")
    assert len(excluded) == 3 and excluded.reason.str.len().min() > 0
    manifest = json.loads((tmp_path / "out/data_manifest.json").read_text())
    assert len(manifest["files"]["ratings.csv"]["sha256"]) == 64


def test_validation_selection_prioritizes_precision():
    rows = [{"mean_centered": False, "min_ratings": 5, "k": 10, "metrics": {"Precision@10": .2, "Coverage": .05}}, {"mean_centered": True, "min_ratings": 1, "k": 20, "metrics": {"Precision@20": .1, "Coverage": .8}}]
    assert select_candidate(rows) is rows[0]


def test_final_test_guard_prevents_retraining(tmp_path, monkeypatch):
    import src.train as training
    import sys
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/experiment_results.json").write_text('{"status":"complete"}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train"])
    monkeypatch.setattr(training, "download_and_extract_data", lambda: pytest.fail("Frozen release must stop before data or training"))
    with pytest.raises(SystemExit, match="Final test already frozen"):
        training.main()


def test_metrics_use_heldout_items_and_exclude_seen():
    movies = pd.DataFrame({"movieId": [1, 2, 3], "title": ["A", "B", "C"], "genres": ["Drama"] * 3})
    train = pd.DataFrame({"userId": [1, 2, 3], "movieId": [1, 2, 2], "rating": [5., 5., 5.]})
    heldout = pd.DataFrame({"userId": [1], "movieId": [2], "rating": [5.]})
    model = PopularityRecommender()
    model.fit(train, movies)
    result = evaluate_recommender(model, train, heldout, k=1)
    assert result["Precision@1"] == result["HitRate@1"] == 1
    assert result["Coverage"] == .5
    assert result["ConfidenceIntervals95"]["Precision@1"] == [1., 1.]


def test_failure_audit_only_keeps_real_issues(tmp_path, monkeypatch):
    import src.train as training
    monkeypatch.setattr(training, "FIGURES_DIR", str(tmp_path))
    movies = pd.DataFrame({"movieId": range(1, 16), "title": [f"Film {i} (2000)" for i in range(1, 16)], "genres": ["Drama"] * 15})
    ratings = pd.DataFrame({"userId": [1, 2], "movieId": [1, 1], "rating": [4., 5.]})
    model = CosineItemItemRecommender(2, False)
    model.fit(ratings, movies)
    cases = analyze_failure_cases(model, movies)
    assert len(cases) == 10
    assert all(case["identified_issues"] for case in cases)
    assert all(case["title"].startswith("Film ") for case in cases)
