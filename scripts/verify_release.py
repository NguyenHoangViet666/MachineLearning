"""Verify saved artifacts and real startup/API without recomputing final test."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

config = json.loads(Path("models/config.json").read_text(encoding="utf-8"))
results = json.loads(Path("reports/experiment_results.json").read_text(encoding="utf-8"))
assert results["status"] == "complete"
assert hashlib.sha256(Path("models/cosine_recommender.pkl").read_bytes()).hexdigest() == config["artifact_sha256"]
manifest = json.loads(Path("data/processed/data_manifest.json").read_text(encoding="utf-8"))
for name, info in manifest["files"].items():
    assert hashlib.sha256(Path("data/raw/ml-latest-small", name).read_bytes()).hexdigest() == info["sha256"]
parts = [pd.read_csv(f"data/processed/{name}_ratings.csv") for name in ("train", "val", "test")]
pairs = [set(zip(p.userId, p.movieId)) for p in parts]
assert not pairs[0] & pairs[1] and not pairs[0] & pairs[2] and not pairs[1] & pairs[2]
with TestClient(app) as client:
    assert client.get("/").status_code == 200
    assert client.get("/api/movies", params={"query": "["}).status_code == 200
    rec = client.get("/api/recommendations?movie_id=1&k=5&year_min=1990&year_max=2010")
    assert rec.status_code == 200 and rec.json()["recommendations"]
    assert client.get("/api/recommendations?movie_id=1&k=0").status_code == 422
    failures = client.get("/api/failure-cases").json()
    assert len(failures) >= 10 and all(row["identified_issues"] for row in failures)
    cold = next(row for row in failures if row["case_type"] == "Cold-start")
    warning = client.get("/api/recommendations", params={"movie_id": cold["movieId"]}).json()
    assert warning["warning"] and warning["status"] == "warning"
    card = client.get("/api/model-card").json()
    assert card["technical_specifications"] == config["selected"]
    assert card["evaluation_summary"]["test"] == results["test"]
    assert client.post("/api/chat", json={"message": "xin chào"}).status_code == 200
    examples = []
    for request_path in ("/api/recommendations?movie_id=1&k=5", f"/api/recommendations?movie_id={cold['movieId']}&k=5", "/api/recommendations?movie_id=1&k=0", "/api/recommendations?movie_id=999999&k=5"):
        response = client.get(request_path)
        examples.append({"request": "GET " + request_path, "status_code": response.status_code, "response": response.json()})
    Path("reports/api_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
print("Release verified: data checksums, split isolation, saved model, startup, UI HTML, API, warnings and frozen metrics.")
