# Phân tích V1 và V2

Lần chạy UTC: 2026-10-08T05:01:57.556146+00:00

| Metric | V1 | V2 | Chênh lệch V2−V1 |
|---|---:|---:|---:|
| faithfulness | 0.9508 | 0.9935 | +0.0427 |
| answer_relevancy | 0.9172 | 0.8742 | -0.0430 |
| context_recall | 1.0000 | 1.0000 | +0.0000 |
| context_precision | 0.9450 | 0.9417 | -0.0033 |

V2 có faithfulness cao hơn hoặc bằng phiên bản còn lại.
V1 giới hạn câu trả lời ngắn và trực tiếp, giảm số mệnh đề cần kiểm chứng. V2 tổ chức Definition và Key facts, yêu cầu từng câu có bằng chứng đầy đủ, nên có thể bao phủ câu hỏi tốt hơn, nhưng cũng tạo thêm mệnh đề cần được context hỗ trợ. Đây là giả thuyết giải thích, không phải bằng chứng nhân quả từ một lần chạy.

Hai phiên bản dùng cùng 50 QA, embeddings, chunking và top-k=3. Context recall/precision chủ yếu phản ánh truy xuất; sai khác giữa hai bản có thể do LLM judge. Cần xem điểm từng mẫu và chạy lặp trước khi kết luận ổn định.

Đạt faithfulness ≥ 0.8: True. Đạt bonus cả hai ≥ 0.9: True.

## Đối chiếu từng mẫu

V1: câu trả lời trung bình 42.0 từ; 45/50 mẫu có faithfulness = 1.

Chú ý V1 mẫu 1: answer trùng nguyên văn reference và câu đó có trong retrieved context, nhưng judge chấm 0. Đây là trường hợp điểm judge không khớp bằng chứng; không nên kết luận answer hallucinate chỉ từ điểm này.

### V1 — mẫu 1, faithfulness 0.0000
What are the three main types of machine learning?

> The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.

### V1 — mẫu 8, faithfulness 0.4000
How do LSTM networks address the vanishing gradient problem?

> LSTM networks address the vanishing gradient problem by using gating mechanisms that control information flow through the network. These gates allow gradients to flow unchanged over many time steps, effectively preserving information and mitigating the issue.

V2: câu trả lời trung bình 66.7 từ; 48/50 mẫu có faithfulness = 1.

### V2 — mẫu 1, faithfulness 0.8000
What are the three main types of machine learning?

> - **Definition**: The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.
> - **Key facts**: In supervised learning, models are trained on labeled input-output pairs to learn a mapping function. In unsupervised learning, models discover hidden patterns or structure in unlabeled data. In reinforcement learning, an agent learns to make decisions based on feedback from its environment.

### V2 — mẫu 30, faithfulness 0.8750
What is hybrid search in vector databases?

> - **Definition**: Hybrid search combines the strengths of both neural and keyword-based retrieval in vector databases.
> 
> - **Key facts**: Hybrid search integrates dense vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion. Dense search excels at semantic matching but may miss exact keyword matches, while sparse search (BM25) excels at precise keyword matching but misses semantic variants. Reciprocal Rank Fusion merges the ranked lists from both approaches by summing reciprocal ranks, resulting in a more robust combined ranking.

50/50 cặp dùng contexts giống hệt nhau. Số từ là mô tả độ dài, không phải thước đo chất lượng. Các mẫu điểm thấp cần đối chiếu passage gốc trong data/rag_outputs_*.json; không xem mọi bất đồng của LLM judge là lỗi thực tế đã chứng minh.

## Cải thiện prompt qua hai lần chạy
V2 ban đầu: 0.8695; sau chỉnh prompt: 0.9935 (chênh lệch +0.1240). Prompt mới bỏ phần giới hạn không có bằng chứng và yêu cầu không hoàn tất passage bị cắt từ kiến thức ngoài context.
V1 không đổi prompt nhưng faithfulness thay từ 0.9786 thành 0.9508, cho thấy biến động giữa các lần chạy. Không quy toàn bộ thay đổi điểm cho prompt khi chưa có nhiều lần đo.

V1 có answer relevancy cao hơn trong lần cuối, V2 có faithfulness cao hơn. Vì vậy cần cân nhắc cả tính trực tiếp lẫn grounding, không chọn một phiên bản chỉ dựa trên một metric.

Dữ liệu đối chiếu đủ 100 mẫu và điểm từng mẫu: [03_ragas_samples.json](03_ragas_samples.json).
