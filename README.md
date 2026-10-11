# MovieLens — Project 01: Gợi ý phim bằng cosine

Học máy cơ bản, lớp 12523W.1 — PGS.TS. Nguyễn Văn Hậu.
Nhóm: Nguyễn Hoàng Việt và Vũ Thành Minh. Nguồn: GroupLens MovieLens latest-small.

Hệ thống biểu diễn mỗi phim bằng vector rating của người dùng, so sánh cosine thô và mean-centered, trả Top-K phim cùng điểm tương đồng/lý do. Title/genres chỉ dùng hiển thị và lọc. Không dự đoán rating hoặc suy diễn sở thích nhạy cảm.

## Cài đặt trên máy mới

Đã kiểm chứng với Python 3.11.9 trên Windows. Chạy từ thư mục gốc repo:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/verify_release.py
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS dùng `.venv/bin/python` thay `.venv/Scripts/python.exe`. Mở http://127.0.0.1:8000; OpenAPI và ví dụ schema tại `/docs`. Model, catalog small và kết quả đã lưu được bàn giao trong repo, nên serving không cần train lại. Thiếu model/catalog trả HTTP 503. Không nạp lại hoặc train trong request.

`requirements.txt` mô tả dependency trực tiếp; `requirements.lock.txt` khóa phiên bản trực tiếp và phụ thuộc. Chỉ tái sinh lock sau khi kiểm thử bằng `python scripts/lock_environment.py`.

Kiểm thử trình duyệt tùy chọn: `python -m pip install -r requirements-browser.txt`, rồi `python scripts/browser_smoke.py --screenshots`. Script dùng Chrome/Edge sẵn có; máy chưa có trình duyệt cần `python -m playwright install chromium`. Script tự mở server tạm, kiểm tra UI desktop/mobile và tắt server; ảnh mới được lưu vào `reports/figures/`.

## Pipeline và cách tái tạo

```powershell
python -m src.data
python -m src.train --development
python -m src.evaluate --split validation
python -m src.evaluate --split test
```

- `src.data`: tải HTTPS nếu thiếu dữ liệu; kiểm tra schema/khóa/rating; lưu dòng loại kèm lý do, checksum và split.
- `src.train --development`: EDA trên train; baseline đếm lượt rating; khảo sát cosine raw/mean-centered, K = 5/10/20 và min_ratings = 1/5/10 trên validation. Không tính metric test, không thay model serving.
- `src.evaluate --split validation`: nạp model đã lưu và đánh giá riêng, không train; lưu `reports/evaluation.json`.
- `src.evaluate --split test`: đọc kết quả test đã đóng băng, không đánh giá test lại.

`python -m src.train` dùng cho **một bản phát hành chưa có** `reports/experiment_results.json`. Lệnh từ chối nếu test đã bắt đầu/hoàn thành để tránh vô tình đánh giá lại. Để tái tạo toàn bộ từ đầu, dùng bản sao riêng của repo, giữ hồ sơ bản phát hành hiện tại và bỏ tệp kết quả đóng băng trong bản sao. Việc tái tạo kỹ thuật không được dùng làm vòng tuning mới trên cùng test. Không tuyên bố các kết quả test lịch sử của phiên bản cũ chưa từng được xem.

Split cố định 70/10/20 theo từng user, sắp timestamp rồi movieId; tỷ lệ làm tròn theo số rating. Timestamp bằng nhau có thể nằm hai bên ranh giới. Không phải cutoff thời gian chung cho mọi user. Seed 42 dùng cho bootstrap; không tạo split ngẫu nhiên mới. Vector, lọc tần suất và trung bình user chỉ học từ train; user mean tính trước khi lọc phim.

Chọn bằng **Precision@K validation**, sau đó coverage, K nhỏ hơn, raw và ngưỡng nhỏ hơn khi hòa. Cấu hình/model được lưu trước test. So sánh test tại K=10 dùng ngưỡng được validation chọn riêng cho từng biểu diễn; so sánh có kiểm soát cùng min_ratings=5/K=10 nằm trong biểu đồ validation. Test cuối chạy một đợt cho các cấu hình đã chốt, trên toàn bộ user có rating giữ lại >=3.5.

Đơn vị đánh giá là user: cộng cosine từ tối đa 5 phim user chấm cao nhất trong train, loại phim đã xem rồi lấy Top-K. Web phục vụ một phim đầu vào; latency offline là truy vấn tổng hợp theo user, không phải độ trễ HTTP của một phim. Coverage dùng mẫu số là toàn bộ phim có rating train. Khoảng tin cậy 95% dùng 1.000 bootstrap user; không tuyên bố nhiều seed/fold cho split thời gian cố định.

## Kết quả hiện tại

Nguồn có thể kiểm tra: `reports/experiment_results.json`, `reports/validation_grid.csv`, `models/config.json`, `models/model_card.json`.

Validation chọn **cosine raw, min_ratings=10, K=5**. Test có 594 user đủ điều kiện:

| Mô hình đối chứng tại K=10 | HitRate@10 | Precision@10 | Coverage |
| --- | ---: | ---: | ---: |
| Popularity theo số rating | 0.3215 | 0.0545 | 0.0113 |
| Cosine raw | 0.3906 | 0.0640 | 0.1052 |
| Cosine mean-centered | 0.2896 | 0.0407 | 0.1329 |

Cấu hình serving đã chọn tại K=5 có Precision@5=0.0653, HitRate@5=0.2559 và Coverage=0.0726. Không so trực tiếp điểm ở K=5 với baseline K=10. Mean-centering tăng coverage nhưng giảm precision ở thí nghiệm này; không khẳng định luôn tốt hơn raw hoặc có quan hệ nhân quả. Có metric theo độ dài lịch sử và khoảng tin cậy trong JSON.

Thí nghiệm 4 quét movieId tăng dần để tìm ca thực sự có vấn đề: không đủ dữ liệu, không đủ kết quả, cosine thấp hoặc ít nhất 80% kết quả cùng thể loại. Lưu ít nhất 10 ca gồm Top-5 thật, lý do và hướng xử lý; ưu tiên giữ cả ca thiếu đa dạng lẫn cold-start. Tiêu chí đa dạng là dấu hiệu cần kiểm tra thủ công, không đồng nghĩa phim gợi ý sai.

## API và đầu vào

`GET /api/movies`, `/api/movies/{movie_id}`, `/api/genres`, `/api/recommendations`, `/api/model-card`, `/api/failure-cases`; `POST /api/chat`.

Ví dụ request:

```text
GET /api/recommendations?movie_id=1&k=5&genres=Animation,Children&match_mode=all&year_min=1990&year_max=2010
```

Response thành công gồm `movie_id`, `query_movie_title`, `rating_count`, `status`, `warning`, `k`, thông tin bộ lọc và `recommendations`. Mỗi phim gồm `movieId`, `title`, `genres`, `similarity_score`, `rating_count`, `rating_mean`, `reason`. Để xem response thật trên model bàn giao:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/recommendations?movie_id=1&k=5' | ConvertTo-Json -Depth 6
```

`reports/api_examples.json` lưu request/response thật cho gợi ý thành công, cold-start, K ngoài miền và ID không tồn tại; được sinh bởi `scripts/verify_release.py`.

K: 1–50; movie_id: số nguyên dương; năm: 1870–2100 và từ <= đến. `match_mode`: `any`, `all`, `exclude`; thể loại phải có trong catalog, không âm thầm đổi giá trị sai. Tìm tên phim là chuỗi literal, không phải regex. Năm không rõ bị loại khi lọc năm. API trả 422 cho input sai, 404 cho phim không tồn tại, 503 khi thiếu artifact. Phim có thật nhưng dưới ngưỡng rating trả 200, `status="warning"`, thông báo rõ và danh sách rỗng. Cosine có miền lý thuyết [-1,1]; serving chỉ trả điểm dương [0,1]; rating [0.5,5.0].

Dashboard và phạm vi/model card đọc model card đã lưu, không ghi cố định KPI. UI có tìm kiếm, chọn phim, Top-K, lọc nhiều thể loại/năm và cảnh báo. Tên phim nhập tự do không bị tự động thay bằng kết quả đầu tiên.

## Hồ sơ bàn giao kỹ thuật

- `data/README.md`, `data/processed/data_manifest.json`: nguồn/điều khoản/dictionary/checksum. Ngày tải gốc của dữ liệu có sẵn chưa biết được ghi rõ.
- `data/processed/quality_report.json`, `excluded_ratings.csv`: kiểm tra chất lượng và dòng bị loại.
- `models/cosine_recommender.pkl`, `config.json`, `model_card.json`: artifact, lựa chọn, môi trường và SHA-256.
- `reports/experiment_results.json`, `validation_grid.csv`, `reports/figures/`: kết quả validation/test và hình có nguồn từ pipeline.
- `tests/`: split, preprocessing, baseline, metric, phân tích lỗi, API đầu vào/lỗi, lọc năm/thể loại và chatbot.
- `reports/nhat_ky_6_tuan_phan_cong.md`: nhật ký nhóm; báo cáo PDF/nguồn và slide là sản phẩm bàn giao riêng, không được các script tự xác nhận đã hoàn thành.

Nguồn dữ liệu: [GroupLens](https://grouplens.org/datasets/movielens/latest/), Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, DOI: 10.1145/2827872. Giữ điều khoản trong `data/raw/ml-latest-small/README.txt`.
