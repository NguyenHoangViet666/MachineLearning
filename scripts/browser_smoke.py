"""End-to-end browser smoke test; launch a temporary local server, then stop it."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--screenshots", action="store_true")
args = parser.parse_args()
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
try:
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            with urlopen(base + "/api/genres", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError("Temporary server exited before startup")
            time.sleep(.1)
    else:
        raise RuntimeError("Server startup timed out")
    config = json.loads((root / "models/config.json").read_text(encoding="utf-8"))
    failures = json.loads((root / "reports/figures/exp4_failure_analysis.json").read_text(encoding="utf-8"))
    cold = next(row for row in failures if row["case_type"] == "Cold-start")
    with sync_playwright() as playwright:
        installed = [Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe", Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe"]
        executable = next((str(path) for path in installed if path.is_file()), None)
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        page.locator('.seed-chip[data-id="1"]').click()
        expect(page.locator("#seed-title")).to_contain_text("Toy Story")
        expect(page.locator(".rec-card").first).to_be_visible()
        # Movie details: title click, focus restoration and recommendation action.
        first_title = page.locator(".rec-title .movie-detail-link").first
        expected_title = first_title.inner_text()
        first_title.click()
        expect(page.locator("#movie-detail-dialog")).to_be_visible()
        expect(page.locator("#movie-detail-title")).to_have_text(expected_title)
        expect(page.locator("#movie-detail-year")).not_to_have_text("")
        expect(page.locator("#movie-detail-rating")).to_contain_text("/ 5.0 sao")
        if args.screenshots:
            page.screenshot(path=str(root / "reports/figures/movie_detail_modal.png"))
        page.keyboard.press("Escape")
        expect(page.locator("#movie-detail-dialog")).not_to_be_visible()
        expect(first_title).to_be_focused()
        page.locator("#seed-title").click()
        expect(page.locator("#movie-detail-title")).to_contain_text("Toy Story")
        page.locator("#movie-detail-recommend").click()
        expect(page.locator("#movie-detail-dialog")).not_to_be_visible()
        # Full K domain is accepted; empty/fractional/out-of-range input never sends a request.
        k_input = page.locator("#select-k")
        for k in (1, 50):
            k_input.fill(str(k))
            with page.expect_response(lambda response: "/api/recommendations?" in response.url and f"k={k}&" in response.url or "/api/recommendations?" in response.url and response.url.endswith(f"k={k}")) as pending_k:
                page.locator("#btn-search-recommend").click()
            assert pending_k.value.status == 200 and pending_k.value.json()["k"] == k
        sent = []
        page.on("request", lambda request: sent.append(request.url) if "/api/recommendations?" in request.url else None)
        for value in ("", "0", "51", "1.5"):
            count_before = len(sent)
            k_input.fill(value)
            page.locator("#btn-search-recommend").click()
            expect(page.locator("#k-error")).to_be_visible()
            expect(k_input).to_have_value(value)
            assert len(sent) == count_before
        k_input.fill("10")
        expect(page.locator("#k-error")).not_to_be_visible()
        page.locator("#year-min").fill("1990")
        page.locator("#year-max").fill("2010")
        with page.expect_response(lambda response: "/api/recommendations?" in response.url and "year_min=1990" in response.url) as pending:
            page.locator("#btn-search-recommend").click()
        recommendations = pending.value.json()["recommendations"]
        assert recommendations and all(1990 <= int(re.search(r"\((\d{4})\)\s*$", item["title"])[1]) <= 2010 for item in recommendations)
        if args.screenshots:
            page.screenshot(path=str(root / "reports/figures/web_app_recommendation_ui.png"), full_page=True)
        page.locator("#year-min").fill("2020")
        page.locator("#btn-search-recommend").click()
        expect(page.locator("#recommendations-grid")).to_contain_text("Năm bắt đầu")
        page.locator("#btn-reset-filters").click()
        page.locator("#movie-search-input").fill(cold["title"])
        page.locator(f'.autocomplete-item[data-id="{cold["movieId"]}"]').click()
        expect(page.locator("#seed-warning-box")).to_be_visible()
        expect(page.locator("#seed-warning-msg")).not_to_have_text("")
        expect(page.locator("#movie-detail-dialog")).to_be_visible()
        expect(page.locator("#movie-detail-warning")).to_be_visible()
        expect(page.locator("#movie-detail-recommend")).to_be_disabled()
        page.locator("#movie-detail-close").click()
        page.locator('[data-tab="tab-dashboard"]').click()
        expect(page.locator("#comparison-table-body tr")).to_have_count(3)
        expect(page.locator("#failure-table-body tr")).to_have_count(10)
        expect(page.locator("#kpi-hit-label")).to_contain_text(f"@{config['selected']['k']}")
        if args.screenshots:
            page.screenshot(path=str(root / "reports/figures/web_app_dashboard_ui.png"), full_page=True)
        page.locator('[data-tab="tab-scope"]').click()
        expect(page.locator("#selected-algorithm")).to_contain_text(f"K={config['selected']['k']}")
        if args.screenshots:
            page.screenshot(path=str(root / "reports/figures/hinh_4_9_dac_ta_model_card.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('[data-tab="tab-recommender"]').click()
        expect(page.locator("#year-min")).to_be_visible()
        page.locator("#seed-title").click()
        expect(page.locator("#movie-detail-dialog")).to_be_visible()
        bounds = page.locator("#movie-detail-dialog").bounding_box()
        assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
        page.locator("#movie-detail-close").click()
        assert not errors, errors
        browser.close()
    print("Browser passed: movie details/actions/focus, K 1-50/errors, search, years, cold-start, dashboard and mobile dialog.")
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
