# NHẬT KÝ THỰC HIỆN 06 TUẦN & BẢNG PHÂN CÔNG ĐÓNG GÓP ĐỒ ÁN

> **HỌC PHẦN:** HỌC MÁY CƠ BẢN — NĂM HỌC 2026 – 2027  
> **TRƯỜNG:** ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN (HYUTE)  
> **LỚP:** 12523W.1  
> **GIẢNG VIÊN HƯỚNG DẪN:** PGS.TS. NGUYỄN VĂN HẬU  
> **ĐỀ TÀI:** PROJECT 01 — HỆ THỐNG GỢI Ý PHIM THEO SỞ THÍCH NGƯỜI DÙNG (MOVIELENS)  
> **CHỦ ĐỀ:** BÀI 1 — VECTOR VÀ ĐỘ TƯƠNG ĐỒNG COSINE  

---

## 1. Thông tin nhóm sinh viên thực hiện

| STT | Họ và tên | Mã sinh viên | Lớp | Vai trò | Email / Liên hệ | Tỷ lệ đóng góp |
| :-: | :--- | :-: | :-: | :---: | :---: | :-: |
| 1 | **Nguyễn Hoàng Việt** | **10123357** | 12523W.1 | Nhóm trưởng | 10123357@hyute.edu.vn | **50%** |
| 2 | **Vũ Thành Minh** | *10123xxx* (bổ sung) | 12523W.1 | Thành viên | 10123xxx@hyute.edu.vn | **50%** |

---

## 2. Bảng ma trận phân công nhiệm vụ (Responsibility Matrix)

Tuân thủ nghiêm ngặt quy định rubric: *Hai thành viên đều tham gia sâu vào cả 2 mảng: Dữ liệu/Mô hình hóa và Web Serving/Báo cáo; không chia cứng một người chỉ làm lý thuyết.*

| Hạng mục công việc | Người phụ trách chính | Người phối hợp / Review | Mức độ hoàn thành |
| :--- | :---: | :---: | :-: |
| **1. Khảo sát dữ liệu & Chống rò rỉ (Leakage)** | Nguyễn Hoàng Việt | Vũ Thành Minh | 100% |
| **2. Phân tích khám phá dữ liệu (EDA & Figures)** | Vũ Thành Minh | Nguyễn Hoàng Việt | 100% |
| **3. Xây dựng ma trận thưa CSR & Mean-centering** | Nguyễn Hoàng Việt | Vũ Thành Minh | 100% |
| **4. Xây dựng mô hình Baseline & Cosine Recommender** | Vũ Thành Minh | Nguyễn Hoàng Việt | 100% |
| **5. Thực hiện 4 thí nghiệm khoa học bắt buộc** | Cả hai cùng thực hiện | Cả hai cùng thực hiện | 100% |
| **6. Thiết kế REST API Serving (FastAPI)** | Nguyễn Hoàng Việt | Vũ Thành Minh | 100% |
| **7. Thiết kế Giao diện Web (Dark Glassmorphism)** | Vũ Thành Minh | Nguyễn Hoàng Việt | 100% |
| **8. Phát triển module mở rộng CineBot In-House NLP** | Nguyễn Hoàng Việt | Vũ Thành Minh | 100% |
| **9. Soạn thảo Báo cáo khoa học & Model Card** | Vũ Thành Minh | Nguyễn Hoàng Việt | 100% |
| **10. Thiết kế Slide thuyết trình & Diễn tập bảo vệ** | Cả hai cùng thực hiện | Cả hai cùng thực hiện | 100% |

---

## 3. Nhật ký chi tiết tiến độ 06 tuần (Weekly Worklog)

### TUẦN 1: Khởi động đề tài, Nghiên cứu MovieLens & Thiết lập Môi trường
* **Thời gian:** 18/08/2026 – 24/08/2026
* **Mục tiêu tuần:** Thống nhất bài toán, đọc tài liệu MovieLens, tạo Git repository, cấu hình môi trường Python và chốt tiêu chí thành công.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Đọc đặc tả đề tài Bài 1 và hướng dẫn GroupLens Research.<br>- Lập Project Brief (1-2 trang), xác định bài toán là Item-Item Retrieval.<br>- Viết tài liệu `data/README.md` (nguồn gốc, trích dẫn bản quyền, schema ban đầu). | 14h | `data/README.md`, Project Brief |
| **Vũ Thành Minh** | - Khởi tạo Git repository, thiết lập `.gitignore`, quy ước commit message.<br>- Cấu hình môi trường ảo Python 3.11, tạo file `requirements.txt` ban đầu.<br>- Thiết lập backlog công việc 6 tuần trên bảng quản trị nhóm. | 13h | Repo Git, `requirements.txt`, Git project board |

* **Vấn đề phát sinh & Cách giải quyết:**
  - *Vấn đề:* Phân vân giữa bài toán dự đoán điểm số (Rating Regression) và gợi ý danh sách Top-K (Item Retrieval).
  - *Giải quyết:* Đối chiếu rubric của Thầy Hậu, bài toán cốt lõi của Bài 1 là Vector và Cosine Similarity, do đó thống nhất tập trung vào Item-Item Retrieval, đánh giá bằng HitRate@K và Precision@K.

---

### TUẦN 2: Tiền xử lý dữ liệu, Kiểm tra Chất lượng & Phân chia Chống rò rỉ (Zero Leakage)
* **Thời gian:** 25/08/2026 – 31/08/2026
* **Mục tiêu tuần:** Tải dữ liệu tự động, kiểm tra missing/outlier, thực hiện Stratified Per-User Temporal Split (70/10/20).

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Viết script `src/data.py` tự động tải `ml-latest-small.zip`.<br>- Lập trình hàm chia tập dữ liệu theo thời gian của từng người dùng (Per-user temporal split): Train (70%), Val (10%), Test (20%).<br>- Đóng băng hoàn toàn tập Test, lưu trữ tại `data/processed/`. | 16h | `src/data.py`, `train_ratings.csv`, `test_ratings.csv` |
| **Vũ Thành Minh** | - Thực hiện EDA phân tích chất lượng dữ liệu: 100.836 ratings, kiểm tra trùng lặp.<br>- Lập biểu đồ phân bố điểm số đánh giá (`reports/figures/eda_rating_distribution.png`).<br>- Lập biểu đồ phân bố đuôi dài Long-Tail của các bộ phim (`reports/figures/eda_long_tail_movies.png`). | 15h | `eda_rating_distribution.png`, `eda_long_tail_movies.png`, Data dictionary |

* **Vấn đề phát sinh & Cách giải quyết:**
  - *Vấn đề:* Có nguy cơ rò rỉ dữ liệu (Data Leakage) nếu dùng chung ma trận ratings toàn bộ để tính tương đồng rồi mới test.
  - *Giải quyết:* Nhóm tuân thủ nghiêm ngặt nguyên tắc: **Chỉ dựng ma trận đặc trưng từ tập Train**. Tập Test độc lập 100% và chỉ được đọc một lần duy nhất lúc đo metric cuối cùng.

---

### TUẦN 3: Xây dựng Ma trận Thưa CSR & Mô hình Cosine Item-Item
* **Thời gian:** 01/09/2026 – 07/09/2026
* **Mục tiêu tuần:** Lập trình chuyển đổi ma trận thưa Item-User, thuật toán Mean-centering, xây dựng Baseline và Cosine Recommender.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Viết `src/features.py`: Lập trình `InteractionMatrixBuilder` chuyển dữ liệu sang `scipy.sparse.csr_matrix` ($N=610$ users).<br>- Lập trình kỹ thuật Mean-centering (Adjusted Cosine) trừ điểm trung bình từng user.<br>- Viết hàm chuẩn hóa L2 từng vector hàng để tăng tốc độ tính tích vô hướng ma trận. | 18h | `src/features.py`, pipeline tạo ma trận CSR |
| **Vũ Thành Minh** | - Viết `src/models.py`: Xây dựng `PopularityRecommender` làm Baseline dựa trên công thức Bayesian weighted rating (IMDb formula).<br>- Xây dựng khung `CosineItemItemRecommender`, tích hợp bộ lọc thể loại (Genre filter) và hàm lưu/tải mô hình (`.pkl`). | 17h | `src/models.py` (Baseline & Cosine Model) |

* **Vấn đề phát sinh & Cách giải quyết:**
  - *Vấn đề:* Khi thử nghiệm với ma trận Dense numpy, bộ nhớ RAM bị tiêu tốn lớn và tốc độ tính ma trận tương đồng $2770 \times 2770$ bị chậm.
  - *Giải quyết:* Chuyển 100% sang ma trận thưa `csr_matrix`, dùng `normalize(..., norm="l2")` rồi nhân ma trận thưa `(X @ X.T)`. Tốc độ tính toán giảm từ 15 giây xuống còn chưa đầy 0.3 giây.

---

### TUẦN 4: Chạy 4 Thí nghiệm Khoa học, Tuning Tham số & Phân tích Lỗi
* **Thời gian:** 08/09/2026 – 14/09/2026
* **Mục tiêu tuần:** Viết module đánh giá `evaluate.py`, chạy 4 thí nghiệm bắt buộc theo yêu cầu đề tài, lập biểu đồ so sánh và Model Card.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Viết `src/evaluate.py`: Đo lường HitRate@K, Precision@K, Recall@K, Catalog Coverage, và Query Latency.<br>- Viết kịch bản `src/train.py` chạy Thí nghiệm 1 (Baseline vs Cosine) và Thí nghiệm 2 (Raw vs Mean-centered Cosine). | 18h | `src/evaluate.py`, `exp1_baseline_vs_cosine.png`, `exp2_raw_vs_mean_centered.png` |
| **Vũ Thành Minh** | - Chạy Thí nghiệm 3 (Khảo sát tham số K và ngưỡng số lượt đánh giá tối thiểu `min_ratings`).<br>- Chạy Thí nghiệm 4: Phân tích 10 ca truy vấn thất bại / ngoại biên, phân loại nhóm lỗi.<br>- Biên soạn đặc tả kỹ thuật `models/model_card.json`. | 19h | `exp3_k_and_min_ratings.png`, `exp4_failure_analysis.json`, `models/model_card.json` |

* **Vấn đề phát sinh & Cách giải quyết:**
  - *Vấn đề (Phát hiện khoa học quan trọng):* Thí nghiệm 2 cho thấy sau khi trừ trung bình (Mean-centering), `Precision@10` giảm từ 6.9% xuống 3.4%, nhưng `Coverage` lại tăng từ 7.25% lên 10.96%.
  - *Giải quyết:* Nhóm không che giấu số liệu mà ghi nhận trung thực vào báo cáo: Mean-centering biến rating thấp thành số âm, làm giảm điểm của các phim bom tấn và mở rộng độ đa dạng cho các phim ngách (catalog coverage).

---

### TUẦN 5: Đóng gói API Serving & Thiết kế Giao diện Web 3 Màn hình
* **Thời gian:** 15/09/2026 – 21/09/2026
* **Mục tiêu tuần:** Tách biệt Offline Training và Online Serving, xây dựng REST API bằng FastAPI, thiết kế UI hiện đại Dark Glassmorphism.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Viết `app/main.py` bằng FastAPI: Thiết kế Pydantic schemas, endpoints `/api/recommendations`, `/api/model-card`, `/api/failure-cases`.<br>- Tách biệt Serving hoàn toàn: Chỉ nạp artifact `.pkl` và `.json`, không train lại khi có request.<br>- Phát triển mở rộng `src/chatbot_engine.py`: Xây dựng In-House NLP Engine (TF-IDF + Cosine Intent Classifier). | 19h | `app/main.py`, `src/chatbot_engine.py`, API specs |
| **Vũ Thành Minh** | - Thiết kế giao diện Web 3 màn hình (`app/templates/index.html`): Tab Gợi Ý Phim, Tab Dashboard Thí Nghiệm, Tab Model Card & Giới Hạn.<br>- Viết `app/static/css/style.css` (Dark Glassmorphism, Neon glow, responsive).<br>- Viết `app/static/js/app.js`: Xử lý live-search autocomplete, render biểu đồ, tích hợp widget chat CineBot. | 20h | `index.html`, `style.css`, `app.js`, Giao diện Web hoàn chỉnh |

* **Vấn đề phát sinh & Cách giải quyết:**
  - *Vấn đề:* MovieLens lưu tiêu đề phim bị đảo ngược mạo từ (VD: *"Matrix, The (1999)"* thay vì *"The Matrix"*), khiến người dùng gõ tìm kiếm tự nhiên không thấy.
  - *Giải quyết:* Viết hàm tự động tháo đảo ngữ mạo từ (The, A, An) trong `chatbot_engine.py` và `app.js`, sinh ra hơn 22.000 khóa ánh xạ tự nhiên.

---

### TUẦN 6: Hoàn thiện Báo cáo Khoa học, Thiết kế Slide & Diễn tập Bảo vệ
* **Thời gian:** 22/09/2026 – 28/09/2026 (hoàn tất chuẩn bị cho buổi báo cáo)
* **Mục tiêu tuần:** Hoàn thiện báo cáo kỹ thuật 15–20 trang, hoàn thành slide trình chiếu 12 trang, rehearsal diễn tập bảo vệ và đóng gói bàn giao.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Hoàng Việt** | - Soạn thảo báo cáo kỹ thuật: Mục 3 (Thiết kế pipeline & Chống rò rỉ), Mục 4 (Công thức toán học Vector Cosine), Mục 5 (Kết quả thí nghiệm).<br>- Viết tài liệu hướng dẫn cài đặt chạy từ máy mới trong `README.md`.<br>- Soạn kịch bản trả lời các câu hỏi vấn đáp lý thuyết (Euclid vs Cosine, ma trận thưa). | 16h | Bản thảo Báo cáo chương 3, 4, 5; `README.md` |
| **Vũ Thành Minh** | - Soạn thảo báo cáo kỹ thuật: Mục 1 (Bối cảnh), Mục 2 (Khám phá dữ liệu EDA), Mục 6 (Phân tích lỗi & Ứng dụng Web).<br>- Thiết kế bộ Slide thuyết trình 12 trang chuẩn mực (Google Slides / PowerPoint).<br>- Kiểm thử chạy lại toàn bộ mã nguồn trên môi trường ảo sạch. | 16h | Bản thảo Báo cáo chương 1, 2, 6; Bộ Slide thuyết trình |
| **Cả hai thành viên** | - Hợp nhất bản Báo cáo hoàn chỉnh, rà soát lỗi chính tả, công thức toán và trích dẫn.<br>- Diễn tập thuyết trình (Rehearsal) đúng khung thời gian 12–15 phút.<br>- Phân vai demo sản phẩm và trả lời phản biện trước Thầy. | 8h (chung) | Báo cáo hoàn chỉnh, Slide hoàn chỉnh, Kịch bản bảo vệ |

---

## 4. Bảng kiểm tra cổng chất lượng trước khi nộp (Submission Checklist)

| Cổng kiểm tra | Tiêu chuẩn theo Rubric | Tình trạng của Nhóm | Minh chứng trong mã nguồn |
| :--- | :--- | :---: | :--- |
| **1. Cổng Dữ liệu** | Nguồn gốc rõ ràng, có data dictionary, script tải tự động, không commit dữ liệu thô quá lớn. | **ĐẠT (PASS)** | `src/data.py`, `data/README.md` |
| **2. Cổng Chống rò rỉ** | Train/Val/Test chia đúng; ma trận vector chỉ fit trên Train; Test độc lập 100%. | **ĐẠT (PASS)** | `src/data.py` (temporal split), `src/features.py` |
| **3. Cổng Đánh giá** | Có baseline so sánh; đủ 4 thí nghiệm bắt buộc; phân tích 10 ca lỗi thực tế. | **ĐẠT (PASS)** | `src/train.py`, `reports/figures/`, `model_card.json` |
| **4. Cổng Sản phẩm Web** | Web chạy local mượt mà; API có schema Pydantic; hiển thị 3 màn hình và cảnh báo cold-start. | **ĐẠT (PASS)** | `app/main.py`, `app/templates/index.html` |
| **5. Cổng Tính tái lập** | Có file `requirements.txt`; một lệnh chạy được server; không hardcode đường dẫn tuyệt đối. | **ĐẠT (PASS)** | `README.md`, `requirements.txt` |
| **6. Cổng Đóng góp Nhóm** | Cả hai thành viên đóng góp cân bằng 50% - 50%; hiểu trọn vẹn toàn bộ pipeline. | **ĐẠT (PASS)** | Nhật ký 6 tuần, lịch sử commit Git |

---

*Hưng Yên, ngày 29 tháng 09 năm 2026*  

**XÁC NHẬN CỦA CÁC THÀNH VIÊN NHÓM:**

| Nhóm trưởng | Thành viên |
| :---: | :---: |
| *(Ký và ghi rõ họ tên)* | *(Ký và ghi rõ họ tên)* |
| <br><br>**Nguyễn Hoàng Việt** | <br><br>**Vũ Thành Minh** |
