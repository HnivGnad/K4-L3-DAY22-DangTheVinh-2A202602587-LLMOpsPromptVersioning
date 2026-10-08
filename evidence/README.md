# Bằng chứng Day 22 — DangTheVinh / 2A202602587

Các log trong thư mục được ghi trực tiếp khi chạy script. Screenshot để bổ sung sau theo yêu cầu.

| File | Cách tạo |
|---|---|
| 02_ab_routing_log.txt | python src/run_all.py --step 2 |
| 03_ragas_report.json | python src/run_all.py --step 3 |
| 04_pii_demo_log.txt | python src/run_all.py --step 4 |
| 04_json_demo_log.txt | python src/run_all.py --step 4 |
| 03_analysis.md | Tự tạo cùng báo cáo RAGAS từ điểm thực đo |

V1 trả lời ngắn, trực tiếp, chỉ dựa trên context. V2 trình bày định nghĩa và facts/cơ chế có cấu trúc, cũng chỉ dựa trên context. V1 có ít mệnh đề cần kiểm chứng hơn; V2 có thể bao phủ câu hỏi tốt hơn nhưng thêm mệnh đề có thể làm giảm faithfulness. Đây là giả thuyết thiết kế. Phân tích điểm thực tế nằm trong [03_analysis.md](03_analysis.md) sau khi chạy RAGAS thành công.

Hai phiên bản dùng cùng dữ liệu, chunk 500/overlap 50, top-k=3. Các điểm mẫu ở data/ragas_v1_per_sample.json và data/ragas_v2_per_sample.json giúp kiểm tra trường hợp bất thường.

Còn bổ sung:
- 01_langsmith_traces.png: danh sách ≥50 traces bước 1, mở một trace có context.
- 02_prompt_hub.png: hai prompt dangthevinh-2a202602587-rag-prompt-v1/v2.
- 03_ragas_scores.png: bảng 4 metric V1/V2 sau khi chạy thật.
- URL project LangSmith và GitHub đã được ghi bên dưới/README gốc.
## Project LangSmith

[day22-dangthevinh-2a202602587](https://smith.langchain.com/o/080078b4-7f91-44a6-8306-dab808064625/projects/p/10197647-1ce8-431c-bf21-7aa6e17e9313)

Đã xác minh trên server: 50 rag-query và 100 ab-rag-query thành công (hai lần A/B), mỗi trace đều có context và answer. Chi tiết và thời điểm xác minh nằm trong [01_langsmith_verification.json](01_langsmith_verification.json). Quyền truy cập phụ thuộc cấu hình tài khoản LangSmith; chưa thay đổi chế độ chia sẻ.

Chạy xác minh lại: python src/verify_langsmith.py.
Kết quả cuối: V1 faithfulness 0.9508, V2 0.9935; cả hai đạt bonus ≥0.9. 10 tests và audit bỏ screenshot đều PASS. [Phân tích thực đo](03_analysis.md), [100 mẫu kèm điểm](03_ragas_samples.json).
