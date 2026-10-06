# Reflection — Lab 19

**Tên:** Đỗ Đình Hoàn (02377)  
**Cohort:** AICB-P2T2 · Track 2 Day 19  
**Path đã chạy:** Lite (fastembed + Qdrant in-memory + SQLite Feast + FastAPI)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` / `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact queries (96.7%):** BM25 và Hybrid hòa nhau. Keyword search vượt trội nhờ các thuật ngữ kỹ thuật verbatim (như `Kubernetes`, `PostgreSQL`, `FastAPI`).
- **Paraphrase queries (32.0%):** Vector và Hybrid đều hạ do mô hình `bge-small-en` (tiếng Anh) chưa tối ưu cho diễn đạt tiếng Việt phức tạp; tuy nhiên trên model đa ngữ (`bge-m3`), Vector/Hybrid sẽ áp đảo hoàn toàn.
- **Mixed queries (100.0%):** Hybrid thắng tuyệt đối (100% vs 97% BM25 & 98.5% Vector) nhờ kết hợp tín hiệu khớp từ khóa chính xác với hiểu ngữ cảnh tổng thể qua Reciprocal Rank Fusion (RRF $k=60$).

**Khi nào KHÔNG dùng Hybrid?**
1. **Pure BM25:** Tra cứu mã định danh/SKU/lỗi hệ thống exact (e.g. `ERR_502_BAD_GATEWAY`, `UUID-9901`), nơi ngữ cảnh vector gây nhiễu và tốn latency/RAM không cần thiết.
2. **Pure Vector:** Tra cứu đa phương tiện (image-to-text), đa ngôn ngữ không có keyword chung, hoặc câu hỏi hoàn toàn mô tả ý tưởng/cảm xúc mà không chứa thuật ngữ cố định.

---

## Điều ngạc nhiên nhất khi làm lab này

1. **RRF đơn giản nhưng cực kỳ mạnh mẽ:** Chỉ với công thức $\frac{1}{k + rank}$, RRF tự động chuẩn hóa điểm số không tương thích giữa BM25 (chưa chặn) và Vector Cosine (0-1) mà không cần chuẩn hóa thủ công.
2. **Cái bẫy Post-filtering:** Khi filter đạt độ chọn lọc cao (selectivity ~4%), post-filter kéo recall sập thẳng về 0.00, khẳng định filtered-ANN là giải pháp bắt buộc trong production.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: Self-driven
