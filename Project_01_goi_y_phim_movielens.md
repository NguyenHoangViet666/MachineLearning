HỌC MÁY CƠ BẢN

ĐỀ TÀI PROJECT KẾT THÚC HỌC PHẦN

**PROJECT 01**

# HỆ THỐNG GỢI Ý PHIM THEO SỞ THÍCH NGƯỜI DÙNG

| **THÔNG TIN**  | **NỘI DUNG**                           |
| -------------- | -------------------------------------- |
| Chủ đề học máy | Bài 1 — Vector và độ tương đồng cosine |
| Lớp            | 12523W.1                               |
| Giảng viên     | PGS.TS. NGUYỄN VĂN HẬU                 |
| Hình thức      | Nhóm 02 sinh viên                      |
| Thời gian      | 06 tuần                                |
| Mức độ         | Trung bình — có phần mở rộng tùy chọn  |
| Dữ liệu        | MovieLens latest-small                 |

## Mục tiêu ngắn gọn

Một nền tảng xem phim cần đề xuất nhanh các phim tương tự với phim người dùng đang quan tâm và giải thích được vì sao chúng giống nhau.

Tài liệu giao nhiệm vụ • Phiên bản dùng cho Kết thúc học phần

# 1\. Bối cảnh, mục tiêu và câu hỏi nghiên cứu

## Bối cảnh thực tiễn

Một nền tảng xem phim cần đề xuất nhanh các phim tương tự với phim người dùng đang quan tâm và giải thích được vì sao chúng giống nhau.

## Câu hỏi nghiên cứu

Từ lịch sử chấm điểm, có thể biểu diễn mỗi phim bằng một vector người dùng và dùng cosine để tìm các phim tương tự đến mức nào?

## Mục tiêu học tập bắt buộc

- Vận dụng đúng kiến thức của Bài 1 — Vector và độ tương đồng cosine; giải thích được giả định, ưu và nhược điểm của phương pháp.
- Xây dựng pipeline có thể chạy lại từ dữ liệu thô đến kết quả; giữ tập kiểm thử độc lập và báo cáo trung thực.
- So sánh với baseline đơn giản, phân tích lỗi thay vì chỉ nêu một con số điểm số.
- Đóng gói kết quả thành ứng dụng web có API, kiểm tra đầu vào và thông tin giới hạn của mô hình.
- Phân công cân bằng cho hai thành viên, thể hiện qua Git history, nhật ký và phần vấn đáp cá nhân.

## Đặc tả bài toán học máy

| **Thành phần**  | **Yêu cầu**                                                                               |
| --------------- | ----------------------------------------------------------------------------------------- |
| Loại bài toán   | Gợi ý theo nội dung tương tác (item–item retrieval), không phải dự đoán điểm số bắt buộc. |
| Đơn vị quan sát | Một phim; vector là điểm đánh giá của các người dùng.                                     |
| Đầu vào         | movieId, userId, rating; title/genres chỉ dùng hiển thị và lọc.                           |
| Đầu ra          | Top-N phim tương tự kèm điểm cosine và lý do ngắn.                                        |
| Chỉ số chính    | Precision@K hoặc hit-rate trên lượt đánh giá giữ lại; coverage; thời gian truy vấn.       |

## Tiêu chí thành công tối thiểu

Nhóm phải chạy được toàn bộ pipeline trên máy khác, vượt hoặc giải thích rõ khi không vượt baseline, và chứng minh việc chia tập/tiền xử lý không làm rò rỉ thông tin.

# 2\. Dữ liệu thực tế và quy tắc sử dụng

| **Thuộc tính**  | **Thông tin đã xác minh**                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Tên dữ liệu     | MovieLens latest-small                                                                                                                     |
| Quy mô          | 100.836 ratings, 9.742 phim, 610 người dùng                                                                                                |
| Quyền sử dụng   | Dùng cho nghiên cứu/giáo dục; ghi nguồn GroupLens, không dùng thương mại nếu chưa xin phép.                                                |
| Phạm vi đề xuất | Dùng bản latest-small; chỉ giữ người/phim có đủ tương tác; tạo ma trận thưa và chia giữ lại một số rating của từng người dùng để đánh giá. |

## Nguồn chính thức

**Trang dữ liệu:** <https://grouplens.org/datasets/movielens/latest/>

**DOI/tài liệu mô tả:** <https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html>

## Hồ sơ dữ liệu phải nộp

- README dữ liệu: nguồn, ngày tải, tệp sử dụng, checksum hoặc phiên bản, giấy phép/điều khoản và trích dẫn.
- Data dictionary: tên biến, kiểu, đơn vị, vai trò feature/target/ID, thời điểm biến trở nên có sẵn.
- Báo cáo chất lượng: missing, trùng lặp, ngoại lệ, phân bố lớp/target và mọi dòng bị loại kèm lý do.
- Không đưa dữ liệu lớn hoặc dữ liệu có hạn chế vào Git; cung cấp script tải hoặc hướng dẫn tái tạo.

## Mốc thời gian và rò rỉ dữ liệu

Không dùng rating cần dự đoán khi tạo vector đánh giá cho chính lần kiểm thử. Việc chuẩn hóa hoặc lọc tần suất phải học từ tập train.

## Quy tắc không thương lượng

- Tách train/validation/test trước mọi bước học từ dữ liệu; mọi imputer, scaler, encoder, feature selection, PCA hoặc resampling phải fit đúng phần train.
- Test chỉ dùng một lần để báo cáo cuối. Không chọn mô hình, ngưỡng hoặc tham số dựa trên test.
- Ghi random_state và môi trường; nếu dữ liệu theo thời gian/người/phiên/hóa đơn, chia theo đúng nhóm tương ứng.

# 3\. Quy trình kỹ thuật và thí nghiệm bắt buộc

## Pipeline tối thiểu

1. Đóng băng định nghĩa bài toán, thời điểm dự đoán và tiêu chí đánh giá trước khi xem test.
2. Tải dữ liệu bằng script; kiểm tra schema, số dòng, khóa và chất lượng dữ liệu.
3. Khám phá dữ liệu trên train; lập biểu đồ phục vụ quyết định tiền xử lý, không chỉ trang trí.
4. Xây dựng baseline: Danh sách phim phổ biến theo số lượt đánh giá.
5. Xây dựng mô hình chính: Cosine item–item trên vector rating thô và vector đã trừ trung bình người dùng.
6. Chọn tham số bằng validation/CV phù hợp; lưu toàn bộ Pipeline và cấu hình.
7. Đánh giá một lần trên test; báo khoảng biến thiên qua seed/fold khi phù hợp.
8. Phân tích lỗi, giới hạn, trường hợp ngoài miền và chuyển mô hình vào API/web.

## Bốn thí nghiệm bắt buộc

| **#** | **Thí nghiệm / bằng chứng cần có**                         |
| ----- | ---------------------------------------------------------- |
| 1     | So sánh phổ biến với cosine.                               |
| 2     | So sánh có/không trừ trung bình người dùng.                |
| 3     | Khảo sát K và ngưỡng số lượt đánh giá tối thiểu.           |
| 4     | Phân tích ít nhất 10 truy vấn thất bại hoặc thiếu đa dạng. |

## Chuẩn so sánh

- Dùng cùng split và cùng đơn vị đánh giá cho các mô hình.
- Báo cả metric chính lẫn ít nhất một biểu đồ/ma trận lỗi phù hợp.
- Tách rõ kết quả validation (dùng chọn) và test (dùng kết luận).
- Không khẳng định quan hệ nhân quả từ hệ số, feature importance, cụm hoặc tương quan.

## Cấu trúc mã nguồn gợi ý

data/README.md • src/data.py • src/features.py • src/train.py • src/evaluate.py • app/ • models/ • reports/figures/ • requirements.txt • README.md

# 4\. Sản phẩm web và hồ sơ bàn giao

## Yêu cầu ứng dụng web

Trang tìm phim, trang chi tiết và danh sách Top-N; bộ lọc thể loại/năm; API GET /api/recommendations?movie_id=&k=; hiển thị điểm tương đồng và cảnh báo khi dữ liệu quá thưa.

- Có ít nhất 03 màn hình: giới thiệu/phạm vi; thao tác chính; dashboard đánh giá & model card.
- API trả JSON, có schema/validation, mã lỗi hợp lý, ví dụ request/response và kiểm thử tối thiểu.
- Giao diện hiển thị đơn vị, miền giá trị và thông báo khi đầu vào thiếu/ngoài miền; không âm thầm thay thế giá trị.
- Tách train offline khỏi serving. Ứng dụng chỉ nạp model/pipeline đã lưu, không huấn luyện lại ở mỗi request.
- Stack khuyến nghị: Python + pandas + scikit-learn; Flask/FastAPI; HTML/CSS/JS hoặc React/Vue. SQLite chỉ khi thật sự cần.

## Các tệp/sản phẩm phải nộp

| **Hạng mục** | **Yêu cầu tối thiểu**                                                                 |
| ------------ | ------------------------------------------------------------------------------------- |
| Kho mã nguồn | Code sạch; README chạy từ máy mới; requirements/lock; commit của cả hai thành viên.   |
| Dữ liệu      | README, data dictionary, script tải/làm sạch, không vi phạm giấy phép.                |
| Mô hình      | Pipeline đã lưu, config/seed, script train và evaluate tách biệt.                     |
| Ứng dụng web | Chạy local; API; validation; dashboard/model card; có dữ liệu demo an toàn.           |
| Báo cáo      | 15–25 trang không tính phụ lục; PDF và tệp nguồn DOCX/LaTeX; biểu đồ đọc được.        |
| Trình bày    | 10–12 slide; demo trực tiếp 5–7 phút; tổng 12–15 phút và vấn đáp.                     |
| Bàn giao     | Link repo, hướng dẫn, model/data card, nhật ký 6 tuần, bảng phân công và tự đánh giá. |

## Phần mở rộng (chỉ chấm khi phần bắt buộc hoàn chỉnh)

Kết hợp điểm cosine với độ phổ biến hoặc thể loại để tăng đa dạng; thêm hồ sơ người dùng mới.

## Nguyên tắc phạm vi

Không bắt buộc cloud, mobile app, deep learning ngoài bài hoặc thu thập thêm dữ liệu. Một hệ thống nhỏ nhưng tái lập, trung thực và phân tích sâu được đánh giá cao hơn sản phẩm nhiều tính năng nhưng thiếu bằng chứng.

# 5\. Kế hoạch thực hiện trong 06 tuần

| **Tuần** | **Công việc chính**                                                    | **Minh chứng đầu ra**                                            |
| -------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1        | Chốt câu hỏi, thời điểm dự đoán, nguồn/giấy phép; tạo repo và backlog. | Project brief 1–2 trang; data README; schema; baseline plan.     |
| 2        | Tải/làm sạch; EDA trên train; hoàn thiện split và baseline.            | Script dữ liệu; notebook EDA; baseline chạy được; checkpoint GV. |
| 3        | Pipeline mô hình chính; validation/CV; theo dõi thí nghiệm.            | Bảng thí nghiệm vòng 1; model candidate; kiểm thử preprocessing. |
| 4        | Hoàn tất thí nghiệm, chọn mô hình/ngưỡng; test cuối; phân tích lỗi.    | Bảng kết quả đóng băng; figures; model card nháp.                |
| 5        | Đóng gói API và giao diện; tích hợp model; test đầu vào và lỗi.        | Web demo end-to-end; API docs; test; checkpoint demo.            |
| 6        | Hoàn thiện báo cáo/slide; rehearsal; tái lập trên môi trường sạch.     | Release cuối; báo cáo; slide; biên bản phân công.                |

## Phân công hai thành viên

- Mỗi người phải có đóng góp đáng kể ở cả phần dữ liệu/mô hình và phần web/báo cáo; không chia cứng một người chỉ viết báo cáo.
- Pull request/commit message phải mô tả công việc. Nhật ký tuần ghi người thực hiện, giờ ước lượng, kết quả và vấn đề.
- Hai thành viên cùng hiểu toàn bộ pipeline và đều có thể trả lời câu hỏi về leakage, metric, mô hình và API.

## Cổng kiểm tra trước khi nộp

| **Cổng**  | **Đạt khi**                                                                               |
| --------- | ----------------------------------------------------------------------------------------- |
| Dữ liệu   | Nguồn truy cập được; giấy phép/trích dẫn rõ; script tái tạo; không dữ liệu nhạy cảm thật. |
| Đánh giá  | Split hợp lệ; baseline; validation/test tách; bảng metric và phân tích lỗi.               |
| Kỹ thuật  | Một lệnh hoặc hướng dẫn ngắn chạy train/evaluate/app; không đường dẫn tuyệt đối cá nhân.  |
| Web       | Luồng chính và API chạy; input xấu không làm sập; model card/giới hạn hiện rõ.            |
| Học thuật | Mọi bảng/hình có nguồn; không sao chép; ghi rõ công cụ AI và cách kiểm chứng nếu có.      |

# 6\. Rubric đánh giá và yêu cầu báo cáo

| **Tiêu chí**                  | **Điểm** | **Mô tả đạt yêu cầu**                                                                        |
| ----------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| 1\. Bài toán & phạm vi        | 8        | Câu hỏi đo được; đơn vị quan sát, thời điểm, đầu ra và tiêu chí thành công rõ.               |
| 2\. Dữ liệu & EDA             | 12       | Nguồn/giấy phép; data dictionary; chất lượng; EDA dẫn tới quyết định.                        |
| 3\. Split, leakage & tái lập  | 15       | Chia đúng; Pipeline đúng; seed/version; test độc lập; không lối tắt.                         |
| 4\. Mô hình & thí nghiệm      | 18       | Baseline; mô hình đúng kiến thức chủ đề; thiết kế so sánh công bằng; ablation/tuning hợp lý. |
| 5\. Đánh giá & phân tích lỗi  | 18       | Metric đúng; biểu đồ/ma trận; sai số theo nhóm/thời gian; kết luận không phóng đại.          |
| 6\. Web/API                   | 12       | Luồng end-to-end; validation; API; UX rõ; model card và cảnh báo.                            |
| 7\. Mã nguồn & bàn giao       | 7        | Cấu trúc, README, môi trường, test cơ bản, model/data artifact có thể tái tạo.               |
| 8\. Báo cáo, trình bày & nhóm | 10       | Logic, trích dẫn, slide/demo, trả lời cá nhân, phân công và Git history.                     |

**Tổng: 100 điểm. Phần mở rộng chỉ được cộng trong phạm vi tiêu chí liên quan, không bù cho sai split, leakage hoặc sản phẩm không chạy.**

## Điều kiện giới hạn điểm

- Không tái lập được kết quả, không có test độc lập hoặc có leakage nghiêm trọng chưa sửa: tối đa 50/100.
- Không có ứng dụng web/API chạy được: mất toàn bộ mục 6; báo cáo không thay thế sản phẩm.
- Không trích nguồn dữ liệu/vi phạm điều khoản hoặc sao chép: xử lý theo quy chế học vụ.
- Một thành viên không chứng minh được đóng góp/hiểu biết: điểm cá nhân có thể khác điểm nhóm.

## Đề cương báo cáo bắt buộc

Tóm tắt • Bối cảnh & câu hỏi • Dữ liệu/giấy phép • Thời điểm & leakage • Phương pháp • Thiết kế thí nghiệm • Kết quả • Phân tích lỗi • Web/API • Đạo đức & giới hạn • Kết luận • Tài liệu tham khảo • Phụ lục tái lập.

## Câu hỏi vấn đáp gợi ý

- Vì sao cosine phù hợp hơn khoảng cách Euclid cho vector rating thưa?
- Bạn đã tránh dùng rating kiểm thử trong vector như thế nào?
- Coverage và độ chính xác Top-K có thể xung đột ra sao?

## Sử dụng có trách nhiệm

Không suy diễn sở thích nhạy cảm. Nêu rõ đây là gợi ý từ dữ liệu lịch sử, không phải đánh giá chất lượng tuyệt đối.

## Tài liệu tham khảo tối thiểu

**\[1\] Slide học phần:** Bài 1 — Vector và độ tương đồng cosine.  
**\[2\] Nguồn dữ liệu:** <https://grouplens.org/datasets/movielens/latest/**\[3\>] DOI/mô tả:**<https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html**\[4\>] Tài liệu phương pháp scikit-learn:** <https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity>