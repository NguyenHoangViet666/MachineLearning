# HỆ THỐNG GỢI Ý PHIM THEO SỞ THÍCH NGƯỜI DÙNG (MOVIELENS)

> **Học máy cơ bản — Đề tài Project 01 (Bài 1: Vector và Độ tương đồng Cosine)**  
> **Lớp:** 12523W.1 | **Giảng viên hướng dẫn:** PGS.TS. Nguyễn Văn Hậu  
> **Hình thức:** Nhóm 02 sinh viên

---

## 1. Giới thiệu tổng quan

Dự án xây dựng hệ thống gợi ý phim tương đồng (Item–Item Collaborative Retrieval) trên nền tảng dữ liệu thực tế **MovieLens latest-small** (100.836 ratings, 9.742 phim, 610 users).

Mỗi bộ phim được mô hình hóa thành một vector trong không gian đánh giá của người dùng ($N$-chiều, $N=610$). Khi người dùng quan tâm đến một bộ phim, hệ thống sử dụng thuật toán **Cosine Similarity điều chỉnh (Adjusted Cosine)** để truy xuất Top-K bộ phim tương tự nhất và giải thích lý do tương đồng.

---

## 2. Nguyên tắc khoa học & Chống rò rỉ dữ liệu (Anti-Leakage)

Dự án tuân thủ nghiêm ngặt các tiêu chuẩn khoa học dữ liệu nhằm đảm bảo tính khách quan và khả năng tái lập:
1. **Phân chia tập dữ liệu (Stratified Temporal / Per-user Split):**
   - Tập dữ liệu được phân chia theo từng người dùng: **Train (70%)**, **Validation (10%)**, và **Test (20%)**.
   - Các đánh giá mới nhất theo thời gian của từng người dùng được giữ lại độc lập trong tập Test.
2. **Không rò rỉ dữ liệu (Zero Leakage):**
   - Ma trận tương tác Item–User và toàn bộ vector đặc trưng của các bộ phim **CHỈ ĐƯỢC XÂY DỰNG TỪ TẬP TRAIN**.
   - Tập Test hoàn toàn độc lập, được đóng băng và chỉ dùng một lần duy nhất để đo lường các metric cuối cùng (`HitRate@K`, `Precision@K`, `Recall@K`, `Coverage`).
3. **Tách biệt Offline Training & Online Serving:**
   - Ứng dụng Web / API chỉ nạp các artifact và metadata đã được tiền tính toán sẵn (`.pkl`, `.json`), tuyệt đối không huấn luyện lại mô hình trong thời gian phục vụ request.

---

## 3. Cấu trúc thư mục mã nguồn

```text
Học máy/
├── data/
│   ├── raw/                       # Dữ liệu gốc MovieLens (ml-latest-small)
│   ├── processed/                 # Dữ liệu đã split (train_ratings.csv, val, test, movies_clean)
│   └── README.md                  # Data dictionary, bản quyền GroupLens & nguyên tắc split
├── src/
│   ├── data.py                    # Script tự động tải, kiểm tra chất lượng & chia tập
│   ├── features.py                # Xây dựng ma trận thưa CSR, khử thiên vị mean-centering
│   ├── models.py                  # PopularityRecommender (Baseline) & CosineItemItemRecommender
│   ├── train.py                   # Huấn luyện mô hình, chạy 4 thí nghiệm, vẽ biểu đồ
│   └── evaluate.py                # Đo lường HitRate@K, Precision@K, Coverage, Latency
├── models/
│   ├── cosine_recommender.pkl     # Trọng số mô hình đã lưu
│   └── model_card.json            # Đặc tả kỹ thuật Model Card chuẩn mực
├── reports/
│   └── figures/                   # Biểu đồ EDA & 4 biểu đồ thí nghiệm khoa học
│       ├── eda_rating_distribution.png
│       ├── eda_long_tail_movies.png
│       ├── exp1_baseline_vs_cosine.png
│       ├── exp2_raw_vs_mean_centered.png
│       ├── exp3_k_and_min_ratings.png
│       └── exp4_failure_analysis.json
├── app/                           # Ứng dụng Web Serving & REST API
│   ├── main.py                    # FastAPI server & endpoints
│   ├── templates/index.html       # Giao diện 3 màn hình (Gợi ý, Dashboard, Model Card)
│   └── static/
│       ├── css/style.css          # Giao diện hiện đại (Dark Glassmorphism)
│       └── js/app.js              # Tương tác tìm kiếm live-search & rendering
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── Project_01_goi_y_phim_movielens.md # Đề bài giao nhiệm vụ kết thúc học phần
└── README.md                      # Hướng dẫn cài đặt và sử dụng
```

---

## 4. Hướng dẫn cài đặt & Chạy từ máy mới

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Tải dữ liệu, chia tập và huấn luyện mô hình (Chạy 4 thí nghiệm)
Chỉ cần chạy một lệnh duy nhất:
```bash
python -m src.train
```
Lệnh trên sẽ tự động:
1. Tải tập dữ liệu MovieLens từ GroupLens và giải nén.
2. Kiểm tra chất lượng dữ liệu (missing, duplicates, schema).
3. Chia tập Train/Val/Test chống rò rỉ dữ liệu.
4. Xuất biểu đồ EDA và thực thi **4 Thí nghiệm bắt buộc**.
5. Lưu model `models/cosine_recommender.pkl` và `models/model_card.json`.

### Bước 3: Khởi chạy ứng dụng Web & API
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Truy cập trình duyệt tại: **<http://127.0.0.1:8000>**

---

## 5. Kết quả 4 Thí nghiệm Bắt buộc

### Thí nghiệm 1: So sánh Baseline Phổ Biến (Popularity) vs Cosine Item-Item
| Metric (K=10) | Baseline (Popularity) | Cosine (Raw Vector) | Cải thiện |
| :--- | :---: | :---: | :---: |
| **HitRate@10** | 0.2700 | **0.3700** | **+37.0%** |
| **Precision@10** | 0.0485 | **0.0690** | **+42.3%** |
| **Recall@10** | 0.0287 | **0.0553** | **+92.7%** |
| **Catalog Coverage** | 0.0061 (0.6%) | **0.0725 (7.25%)** | **Gấp 11.9 lần** |
| **Query Latency** | 0.28 ms | 86.88 ms | Đáp ứng thời gian thực (< 100ms) |

*Nhận xét:* Baseline tuy cho thời gian truy vấn cực nhanh nhưng độ phủ danh mục cực kỳ nghèo nàn (chỉ 0.6% kho phim được gợi ý), hoàn toàn không thể hiện được tính cá nhân hóa theo từng sở thích. Mô hình Cosine vượt trội rõ rệt trên tất cả các tiêu chí đánh giá.

### Thí nghiệm 2: Khử thiên vị người dùng (Mean-Centering / Adjusted Cosine)
* Trừ điểm trung bình của từng user ($r'_{ui} = r_{ui} - \mu_u$) giúp mở rộng độ phủ kho phim lên **10.96%** (so với 7.25% của Raw Cosine), loại bỏ hiệu ứng các bộ phim chỉ được điểm cao vì người xem quá dễ tính.

### Thí nghiệm 3: Khảo sát siêu tham số K và ngưỡng lọc `min_ratings`
* Khảo sát với $K \in \{5, 10, 20\}$ và $min\_ratings \in \{1, 5, 10\}$ trên tập Validation:
  * Khi tăng $K$ từ 5 lên 20, `HitRate` tăng từ 0.17 lên 0.38, độ phủ tăng từ 3.8% lên 9.6%.
  * Chọn ngưỡng $min\_ratings = 5$ giúp cân bằng giữa độ chính xác và khả năng phục vụ được nhiều phim trong danh mục.

### Thí nghiệm 4: Phân tích 10 ca truy vấn ngoại biên & thất bại
Chi tiết được lưu tại `reports/figures/exp4_failure_analysis.json` và hiển thị trực tiếp trên tab **Dashboard** của ứng dụng Web:
* **Các ca tương đồng xuất sắc:** *The Godfather (1972)* $\to$ *The Godfather: Part II (1974)* (Điểm cosine 0.6731); *Toy Story (1995)* $\to$ *Toy Story 2 (1999)* (Điểm cosine 0.3301).
* **Các ca Cold-start & Dữ liệu thưa:** *Black Butler (2017)*, *Camera Buff (1979)* chỉ có 1 tương tác trong tập train $\to$ Hệ thống kích hoạt cảnh báo `Warning` và từ chối gợi ý ảo để bảo vệ độ tin cậy.

---

## 6. Đặc tả REST API

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/api/movies?query={name}&limit={n}` | Tìm kiếm phim theo tên phục vụ Autocomplete |
| `GET` | `/api/movies/{id}` | Lấy chi tiết thông tin phim và số lượt rating |
| `GET` | `/api/recommendations?movie_id={id}&k={k}&genre={genre}` | Gợi ý Top-K phim tương đồng theo Cosine |
| `GET` | `/api/model-card` | Xuất thông tin Model Card dưới dạng JSON |
| `GET` | `/api/failure-cases` | Danh sách dữ liệu phân tích 10 ca ngoại biên (TN4) |

---

## 7. Phân công đóng góp (Nhóm 02 sinh viên)

| Thành viên | Trách nhiệm chính |
| :--- | :--- |
| **Thành viên 1** | Xây dựng pipeline dữ liệu (`src/data.py`, `src/features.py`), kiểm tra Data Leakage, thiết kế thí nghiệm 1 & 2. |
| **Thành viên 2** | Triển khai mô hình Cosine (`src/models.py`, `src/evaluate.py`), khảo sát tham số thí nghiệm 3 & 4, xây dựng Web App & API (`app/`). |
| **Cả nhóm** | Cùng thảo luận lựa chọn metric, phân tích 10 ca lỗi, viết báo cáo tổng kết và chuẩn bị slide thuyết trình. |
