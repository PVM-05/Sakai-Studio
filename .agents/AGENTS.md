# Quy định Bắt buộc cho Agent

## 1. Tuân thủ Quy tắc Thiết kế
- Agent phải luôn đọc và tuân thủ các quy tắc trong tệp `.agents/rules/design-rules.md` trước khi viết mã nguồn hoặc thiết kế giao diện.
- Tất cả các văn bản hiển thị trên giao diện, thông báo và tài liệu hướng dẫn phải sử dụng tiếng Việt chuẩn mực.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.

## 2. Ép buộc Kích hoạt Kỹ năng (Skills & Workflows Enforcement)
Agent **BẮT BUỘC** nhận diện ngữ cảnh và áp dụng ngay lập tức TẤT CẢ các kỹ năng (skills) có trong `.agents/skills/` theo đúng quy trình (workflow). Không chờ người dùng nhắc.

**A. Quy trình Kế hoạch & Thiết kế (Planning)**
- **`brainstorming`**: Bắt buộc khởi chạy hỏi đáp, lập 2-3 phương án và chốt thiết kế trước khi code.
- **`writing-plans` / `executing-plans`**: Bắt buộc viết tài liệu kế hoạch (plan) chi tiết và bám sát thực thi từng bước.
- **`using-superpowers` / `writing-skills`**: Bắt buộc đọc và sử dụng các năng lực hệ thống, quản lý quy trình đúng chuẩn.

**B. Quy trình Code & Quản lý Nhánh (Git/Code Workflow)**
- **`requesting-code-review` / `receiving-code-review`**: Tự động đánh giá, rà soát code trước khi hoàn thành task.
- **`using-git-worktrees` / `finishing-a-development-branch`**: Tự động quản lý branch an toàn khi phát triển và merge tính năng.

**C. Quy trình Debug & Kiểm thử (Testing)**
- **`test-driven-development`**: Bắt buộc viết kịch bản test trước, code sau.
- **`systematic-debugging`**: Phải debug có hệ thống, không đoán mò khi gặp lỗi.
- **`verification-before-completion`**: Phải tự chạy code xác minh thành công trước khi báo hoàn tất.

**D. Phân Việc & Tối ưu Bối cảnh (Delegation & Caveman)**
- **`caveman` (và họ caveman-*)**: Mặc định giao tiếp siêu ngắn gọn để tối ưu token. Tự động nén file, đo đếm cost và review.
- **`cavecrew` / `dispatching-parallel-agents` / `subagent-driven-development`**: Tự động phân rã task lớn thành nhiều subagent để chạy song song.

**E. Chuyên Môn Cụ Thể (Domain Specific)**
- **`thiet-ke-giao-dien`**: Ép Word-Wrap, không emoji, 100% tiếng Việt, icon Fluent.
- **`video-ai-engineering`**: Ép quy chuẩn khi đụng AI (STTN, Whisper, SAM2) và FFmpeg.

> **Lưu ý Cốt Lõi:** Agent phải điền ĐẦY ĐỦ checklist của từng skill tương ứng trước khi chuyển trạng thái. Việc bỏ qua bất kỳ quy trình nào (đặc biệt là planning & verification) bị xem là vi phạm nghiêm trọng.
