# PAR resolution-degradation benchmark

Đánh giá checkpoint cố định trên clean/×2/×4/×8; không train, không SR và không gọi phép thử này là ZSSR.

## Protocol

Giải mã ảnh bằng Pillow → RGB, không EXIF transpose. Mỗi scale sinh trực tiếp từ ảnh gốc: bicubic xuống `(max(1,W//s),max(1,H//s))`, bicubic trở lại `(W,H)`, lưu PNG lossless. Sau đó dùng validation transform gốc của checkpoint. Threshold chính là `p > 0.5`.

Dữ liệu server: `/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition/data/robustness/par_resolution_v1/`. Có 6.002 ảnh validation và 23.822 ảnh test, tổng 119.296 PNG. Manifest giữ labels, ID, source/output SHA-256, kích thước gốc/trung gian và cờ clamp. Split chính thức giữ nguyên; tập loại trùng chỉ dùng báo cáo bổ sung.

## Đường chạy

1. `python -m experiments.par_resolution.generate` sinh dữ liệu; `--resume` xác minh/reuse, `--repair` sửa output hỏng một cách tường minh.
2. `python -m unittest discover -s experiments/par_resolution/tests -v` kiểm thử CPU, gồm evaluator production với model nhỏ.
3. `registry` chỉ nhận checkpoint hoàn thành; `freeze` tạo snapshot riêng và kiểm thử snapshot đó, sau đó nạp strict trọng số vào năm kiến trúc thật trên CPU.
4. `worker` chờ GPU trống và dùng cooperative lock chung với EOPAR. Smoke 128 ảnh validation ở bốn scale phải đạt trước khi tạo test gate.
5. Worker đánh giá val/test cho từng checkpoint, một job/GPU. Không dừng process khác; lỗi làm worker dừng, không retry vô hạn. Checksum/source/environment thay đổi sẽ chặn chạy.
6. `analyze` dựng CSV, Markdown, đường mA theo scale, bảng từng thuộc tính/support, kích thước ảnh và bootstrap ghép cặp từ prediction đã kiểm hash. Chỉ xuất số thực đo.

## Lượt đang triển khai

Snapshot server: `experiments/releases/par_resolution_v5/repo/`.
Báo cáo kiểm thử/registry: `experiments/par_resolution/reports/release_v5/`.
Tiến trình, log và kết quả: `experiments/par_resolution/outputs/release_v5/`.

Hai launcher đã được khởi động; chúng đợi `release_ready.json` trước khi chạy worker. GPU 1 chạy ngay khi đạt gate tài nguyên; GPU 0 chờ tiến trình hiện có kết thúc. Xem `worker0.json`, `worker1.json` và `supervisor*.log` để biết trạng thái thực tế.

9 checkpoint: C0; C3/C3_MS seed 42,43; C4b/C4b_MS batch16×4 seed42,43. Không đưa seed44 đang thay đổi vào registry. Đây là **baseline comparison**. MambaPAR, PromptPAR, LLM-PAR, FOCUS chưa có checkpoint MSP60K57 phù hợp được xác minh. EOPAR chưa có checkpoint đủ điều kiện; việc đánh giá này không mở khóa train EOPAR.

Giới hạn provenance: checkpoint cũ không chứa hash nhãn tại thời điểm train; binding dựa vào saved pickle path, thứ tự thuộc tính hiện tại và strict architecture loading. Lịch sử degradation augmentation chưa xác minh, nên không tuyên bố unseen/zero-shot robustness. Khoảng tin cậy bootstrap theo ảnh không thay thế độ lệch giữa seed hoặc cluster identity/camera.
