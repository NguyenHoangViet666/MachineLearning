# HỒ SƠ DỮ LIỆU — MOVIELENS LATEST-SMALL

## 1. Nguồn dữ liệu & Giấy phép sử dụng
* **Tên tập dữ liệu:** MovieLens Latest Datasets (Small version - `ml-latest-small.zip`).
* **Nguồn chính thức:** GroupLens Research, Đại học Minnesota (Hoa Kỳ).
* **URL:** <https://grouplens.org/datasets/movielens/latest/>
* **Tài liệu đặc tả (README gốc):** <https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html>
* **Quyền sử dụng & Bản quyền:** 
  * Được phép sử dụng tự do cho mục đích nghiên cứu học thuật và giáo dục.
  * Không sử dụng cho mục đích thương mại nếu chưa có sự đồng ý bằng văn bản của GroupLens.
  * Trích dẫn yêu cầu:
    > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

## 2. Quy mô & Cấu trúc tệp dữ liệu
* **Quy mô tổng thể:** 100.836 lượt đánh giá (ratings) và 3.688 tag ứng dụng trên 9.742 bộ phim bởi 610 người dùng.
* **Thời gian ghi nhận:** Từ 29/03/1996 đến 24/09/2018. Tất cả người dùng được chọn ngẫu nhiên, mỗi người đã đánh giá ít nhất 20 bộ phim.
* **Các tệp thành phần:**
  * `ratings.csv`: Chứa toàn bộ lịch sử chấm điểm (`userId`, `movieId`, `rating`, `timestamp`).
  * `movies.csv`: Thông tin phim (`movieId`, `title`, `genres`).
  * `tags.csv`: Các thẻ do người dùng tự gán cho phim (`userId`, `movieId`, `tag`, `timestamp`).
  * `links.csv`: Mã liên kết định danh sang IMDb và TMDb (`movieId`, `imdbId`, `tmdbId`).

## 3. Data Dictionary (Từ điển dữ liệu)

### Bảng `ratings.csv` (Dữ liệu cốt lõi cho mô hình)
| Tên cột | Kiểu dữ liệu | Đơn vị / Miền giá trị | Vai trò | Thời điểm có sẵn | Mô tả |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `userId` | Integer | [1, 610] | Entity ID | Lịch sử | Mã định danh duy nhất của người dùng (đã được ẩn danh) |
| `movieId` | Integer | [1, 193609] | Entity ID | Lịch sử | Mã định danh duy nhất của bộ phim |
| `rating` | Float | [0.5, 5.0], bước 0.5 | Interaction / Value | Lịch sử | Điểm đánh giá mức độ yêu thích phim |
| `timestamp` | Integer | Unix epoch (giây) | Temporal feature | Lịch sử | Thời điểm người dùng thực hiện đánh giá |

### Bảng `movies.csv` (Dữ liệu bổ trợ & hiển thị)
| Tên cột | Kiểu dữ liệu | Vai trò | Mô tả |
| :--- | :--- | :--- | :--- |
| `movieId` | Integer | Primary Key | Mã định danh khớp với `ratings.csv` |
| `title` | String | Metadata | Tên phim kèm năm phát hành trong ngoặc, ví dụ: *Toy Story (1995)* |
| `genres` | String | Metadata | Danh sách thể loại phân tách bởi dấu `\|` (Action, Comedy, Drama,...) |

## 4. Nguyên tắc chống rò rỉ dữ liệu (Anti-Leakage Rules)
1. **Chia tập Train/Validation/Test:** 
   * Dùng chiến lược **Leave-K-Out** hoặc **Temporal Split** theo từng người dùng: giữ lại một số tương tác đánh giá cuối cùng của mỗi người dùng cho tập Test.
2. **Phạm vi tính toán Vector:**
   * Ma trận User–Item và vector đặc trưng của mỗi bộ phim **CHỈ ĐƯỢC XÂY DỰNG TỪ TẬP TRAIN**.
   * Tập Test hoàn toàn độc lập và chỉ được dùng ở bước cuối cùng để đo lường độ chính xác Top-K (Precision@K, Hit-Rate@K) của danh sách gợi ý.
   * Tuyệt đối không tính toán Cosine Similarity trên toàn bộ dữ liệu trước khi split.
