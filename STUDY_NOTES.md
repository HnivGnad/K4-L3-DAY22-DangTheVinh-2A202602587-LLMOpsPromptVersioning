# Giải thích bài làm

## Luồng dữ liệu và tracing

Knowledge base là tài liệu duy nhất dùng cho câu trả lời. RecursiveCharacterTextSplitter chia theo đoạn/câu rồi ký tự, chunk tối đa 500 ký tự và overlap 50. Embeddings ánh xạ mỗi chunk thành vector; FAISS tìm ba vector gần câu hỏi nhất. Reference trong qa_pairs.py chỉ dùng lúc chấm, không được đưa vào prompt sinh answer.

Trong LCEL, pipe nối đầu ra bước trước vào đầu vào bước sau. Dict đầu tiên tạo context bằng retriever và chuyển nguyên question bằng RunnablePassthrough. RunnablePassthrough.assign giữ hai trường này rồi thêm answer do prompt → LLM → StrOutputParser tạo ra. Vì vậy trace gốc có question, context và answer; trace con vẫn có hoạt động retriever, prompt, model và parser.

@traceable bọc một lần gọi hàm thành run. Tracing bật từ config.py trước khi import LangChain. Chạy 50 câu bước 1 và 50 câu bước 2 tạo ít nhất 100 root queries nếu API và upload thành công. Dòng console chỉ xác nhận gọi xong; kiểm tra server LangSmith mới xác nhận có traces.

## Prompt Hub và routing

prompts.py là nguồn chung của V1/V2. V1 yêu cầu câu trả lời ngắn và trực tiếp. V2 yêu cầu định nghĩa và facts/cơ chế có cấu trúc. Cả hai dùng context, từ chối suy đoán và trả lời bằng ngôn ngữ câu hỏi.

Push lưu template thành commit trên Hub. Pull lấy template từ server khi chạy. Bước 2/3 không tự dùng local khi pull hỏng: việc đó có thể che lỗi key và tạo evidence sai yêu cầu. Hàm có fallback tường minh cho debug nhưng main không bật.

MD5(request_id) chuyển thành số nguyên rồi lấy modulo 2. Một ID luôn có cùng bucket giữa nhiều lần chạy và nhiều Python process. Đây là hash phân nhóm, không dùng cho mật mã hay bảo vệ dữ liệu. Hash không bảo đảm phân phối chính xác 25/25, nên kiểm tra cả hai version đều nhận câu hỏi.

## RAGAS

Mỗi SingleTurnSample gồm user_input, response, retrieved_contexts và reference. Contexts phải là list các passage riêng, để chấm từng passage.

- Faithfulness: các mệnh đề trong response được context hỗ trợ đến đâu.
- Answer relevancy: response đáp ứng question đến đâu; RAGAS sinh câu hỏi từ response và so embeddings.
- Context recall: reference được retrieved contexts bao phủ đến đâu.
- Context precision: các passages hữu ích được xếp hạng tốt đến đâu, so với reference.

Bước 3 chấm cả 50 QA cho mỗi prompt. LLM và embeddings cho evaluator được truyền tường minh. Báo cáo lưu mean của đủ 50 điểm cho mỗi metric; thiếu/NaN làm bước thất bại thay vì bỏ mẫu để tăng điểm. Điểm từng mẫu được lưu riêng để debug. Target là ít nhất một faithfulness ≥0.8; bonus khi cả hai ≥0.9 chỉ xác nhận sau chạy thật.

V1 có ít mệnh đề nên có thể giảm hallucination. V2 có thể đầy đủ hơn nhưng cần grounding cho thêm mệnh đề. Cùng truy xuất nên recall/precision dự kiến tương tự; LLM judge có nhiễu. Phân tích tự tạo nêu chênh lệch số đo và giới hạn, không khẳng định nhân quả từ một lần chạy.

## Guardrails

register_validator đăng ký hai class tự viết với Guardrails. validate trả PassResult nếu input sạch/hợp lệ. Khi lỗi, FailResult có fix_value; OnFailAction.FIX được đặt trong constructor của validator để Guard thay output bằng fix_value.

PIIDetector dùng regex cho email, credit card, SSN và phone. Card được xử lý trước phone để tránh một phần số thẻ bị che thành phone. Regex này là demo bài lab, không phát hiện mọi PII trong thực tế. Dữ liệu demo là giả.

JSONFormatter thử parse trước. Nếu lỗi, nó gỡ markdown fences, chuyển từng string nháy đơn bằng ast.literal_eval và json.dumps, xóa comma thừa ngoài string. Bộ quét bỏ qua ký tự escape và bảo toàn apostrophe/comma trong string nháy đôi. ast.literal_eval chỉ đọc literal, không thực thi code. Nếu vẫn không parse được, trả object error/raw (raw tối đa 200 ký tự). NaN/Infinity bị từ chối vì không thuộc JSON chuẩn.

## Kiểm tra và evidence

Unit tests chạy FAISS thật với fixture embeddings/model, kiểm tra context và schema. Tracing tắt trong tests để không tạo root traces giả. Demo bước 4 gọi Guard thật và kiểm tra output trước khi ghi PASS. Hai log tách riêng; A/B log và report được lưu khi chạy bước thật.

run_all.py dừng ở lỗi đầu tiên và trả exit code 1. Có thể chạy lại một bước bằng --step. verify_submission.py kiểm tra file, cấu trúc report, đủ request ID, số demo và .env không bị track. --skip-screenshots dùng trong giai đoạn hiện tại; khi nộp cần bỏ cờ và bổ sung ba ảnh.