# Nguồn tái tạo báo cáo PAR

PDF chính: `../bao_cao_PAR.pdf`. Đây là bản 24 trang, 960×540 pt, tiếng Việt, chốt snapshot ngày 02/11/2026. Font DejaVu Sans thay cho Tahoma/Arial thiếu trên máy; font và giấy phép nằm trong assets/fonts. Logo được trích từ mẫu do người dùng cung cấp; background, sơ đồ và biểu đồ được dựng bằng vector.

## Dựng lại

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python build_report.py
python export_analysis.py
```

`content.json` chứa mục lục, thông tin bìa, tài liệu tham khảo và 100 khối nội dung có thể sửa. `build_report.py` chứa quy tắc bố cục và các bảng/biểu đồ được tính từ dữ liệu. `data/` giữ snapshot nhỏ và prediction, không chứa checkpoint hay bộ dữ liệu lớn. `PROVENANCE.csv` ánh xạ trang/khẳng định tới nguồn và SHA-256. `metric_reference.py` là bản chụp evaluator thuần NumPy.

Bản này dùng test baseline lịch sử (chỉ mA) và full validation C3 (mA, mean-instance F1). Bảng baseline đối chiếu mA với trung bình 57 thuộc tính; do không có prediction lịch sử trong gói, không tuyên bố đã tính lại toàn bộ metric lịch sử. Tám artifact NPZ của validation C3 và smoke C3_MS đã kiểm checksum và tính lại metric. Điểm smoke không dùng làm kết quả test. Không có số EOPAR hoặc phương pháp ngoài dự án được bịa thêm.

`attribute_resolution.csv` có mA từng thuộc tính, số mẫu dương/âm và mức giảm ×8. `resolution_strata.csv` phân tích theo chiều nhỏ nhất của ảnh gốc/trung gian. Đây là phân tích validation một seed, không thay thế kiểm định nhiều seed, test hoặc ảnh che tự nhiên.

## Cập nhật kết quả

1. Lấy artifact đánh giá đã hoàn tất ở một mốc cố định; giữ marker, JSON, NPZ và registry/checksum tương ứng trong thư mục snapshot mới.
2. Kiểm split, checkpoint, seed, attribute order và checksum trước khi dùng số mới; không ghi đè dữ liệu làm nguồn của bản cũ.
3. Sửa đường dữ liệu và mô tả rõ validation/test, ngày chốt và trạng thái trong content.json. Giữ bảng lịch sử tách biệt với metric chuẩn.
4. Dựng lại PDF; dùng `render_verify.py` khi chạy trong repo để render cả mẫu và báo cáo; đọc toàn bộ 24 trang, đặc biệt bảng và công thức. Sau render, cập nhật verification.json với kết quả kiểm tra trực quan.

Lưu ý: `render_verify.py` mặc định tìm mẫu `Slide (6).pdf` ở root repo và lưu hình trung gian vào tmp/pdfs/par_report. Khi chuyển gói nguồn ra nơi khác, sửa đường mẫu trong script nếu muốn đối chiếu mẫu. Không cần mẫu để chạy build_report.py khi các code_evidence đã đóng gói sẵn; script ưu tiên code gốc nếu chạy trong repo.

Các phần đang chờ tại mốc chốt: test resolution đủ 9 checkpoint; so sánh nhiều seed; checkpoint SOTA ngoài dự án tương thích MSP60K57; supervision vùng hợp lệ và GPU/train gates EOPAR; nhãn che tự nhiên/cross-domain. Báo cáo không mở khóa bất kỳ job train nào.
