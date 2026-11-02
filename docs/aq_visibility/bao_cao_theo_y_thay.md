# Báo cáo nhánh AQ + Visibility theo định hướng của thầy

**Mục tiêu nhánh:** kiểm tra PAR có xét **khả năng quan sát (visibility)** của từng thuộc tính khi bị che khuất hay không, và đo đóng góp từng thành phần bằng ablation C0–C4.

**Phạm vi:** ma trận thesis trong `aq_visibility/` (tách khỏi nhánh improve/DeepMAR/KD).  
**Dữ liệu eval:** MSP60k `dataset_random.pkl`, test **n = 23822**.  
**Nguồn số:** `aq_visibility/output/eval_json/{C0..C4}_test.json` (đo trên sol1, 2026-10-13). Chỉ ghi số đã đo.

---

## 1. Mapping ý thầy → việc đã làm

| Ý thầy | Trạng thái | Cách thực hiện trong repo |
|--------|------------|---------------------------|
| Giữ vision encoder + text encoder | Đã làm | VMamba (vision) + Mamba text; không thay backbone |
| Thay gộp 57 attr / 11 nhóm → **57 attribute queries** | Đã làm | `AttributeQueryDecoder`, C2–C4 |
| Head riêng từng query (thay head sau global pooling) | Đã làm | Predictor per-query; C0/C1 vẫn GAP+MLP (`_my_model_8`) |
| Thêm **visibility head** | Đã làm | VisHead trên C3/C4; `λ_v = 0.5` |
| Ban đầu token VMamba hiện có; đa tỉ lệ chỉ khi mất chi tiết attr nhỏ | Đã làm / deferred | Single-scale; Phase 6 giữ single-scale (xem §5) |
| Nhánh nền ảnh = ablation tùy chọn | Đã scaffold | C5 `use_background=True` — **chưa train** (không vào bảng chính) |
| Chia train/val/test **trước** khi tạo biến thể | Đã làm | Split cố định `dataset_random.pkl` → rồi mới build pairs |
| Cặp gốc–che trong cùng partition | Đã làm | `occlusion_pairs_{train,val,test}.pkl` |
| Vùng bằng chứng: đầu, áo, quần, giày | Đã làm | Regions `head / upper / lower / shoes` |
| Ước lượng che bằng body/pose; uncertain loại khỏi `L_vis` | Đã làm | Soft target + `vis_valid`; uncertain skip |
| Không mặc định ảnh gốc `vis=1`; audit thủ công | Một phần | Rule `vis_valid=0` khi unpaired; CSV audit còn **template** (chưa gắn đủ nhãn tự nhiên) |
| Loss `L_attr + λ_v L_vis + λ_c L_consist` | Đã làm | C3: `λ_v=0.5, λ_c=0`; C4: `λ_v=0.5, λ_c=0.2` |
| So sánh 5 mô hình + in metrics | Đã làm | C0–C4 train + eval đầy đủ |

---

## 2. Năm cấu hình so sánh (đúng checklist thầy)

| ID | Tên cụ thể | Khớp ý thầy |
|----|------------|-------------|
| **C0** | Base (`_my_model_8` GAP+MLP) | Mô hình base trên git |
| **C1** | Base + Region Occlusion Aug | Base + augmentation che khuất có kiểm soát |
| **C2** | Attribute Query (57 queries) | Attribute query |
| **C3** | AQ + Visibility Head | Query + visibility |
| **C4** | AQ + Visibility + Consistency | Queries + visibility + consistency |

Ghi chú: C2–C4 dùng package `aq_visibility` (không phải `_my_model_8`). C0/C1 giữ Base để ablation “chỉ đổi head/query”.

---

## 3. Dữ liệu occlusion đã xây

| Split | Số cặp gốc–che | Phân bố vùng (ước lượng) |
|-------|----------------|--------------------------|
| train | 30298 | head 7583 · upper 7602 · lower 7422 · shoes 7691 |
| val | 6002 | head 1476 · upper 1491 · lower 1505 · shoes 1530 |
| test | 23822 | head 5981 · upper 6019 · lower 5798 · shoes 6024 |

- Uncertain / invalid (loại hoặc không vào `L_vis`): train 323 · val 50 · test 272.
- Body–attr map: `data/occlusion/body_region_map.json` (57 thuộc tính → region / global).
- Soft target: attr vùng bị che ≈ 0.15; attr vùng còn bằng chứng ≈ 0.85; attr global/`vis_valid=0` không vào `L_vis`.
- **Audit thủ công:** `manual_visibility.csv` hiện mới là template (report: “Label 50–100 natural occlusion images…”). Đây là hạng mục còn mở theo đúng yêu cầu thầy.

---

## 4. Loss (theo thầy)

\[
L = L_{\text{attr}} + \lambda_v L_{\text{vis}} + \lambda_c L_{\text{consist}}
\]

| Thành phần | Định nghĩa đã implement |
|------------|-------------------------|
| \(L_{\text{attr}}\) | BCE có trọng số (+ ASL) trên attribute logits |
| \(L_{\text{vis}}\) | BCE có mask trên `vis_pred` với `vis_valid=1`; dự đoán từ **ảnh bị che** |
| \(L_{\text{consist}}\) | \(\mathrm{mean}\|\sigma(\text{clean})-\sigma(\text{occ})\|\) chỉ trên attr có vùng bằng chứng còn nguyên (`consist_mask`) |

Preset thesis: C3 `λ_v=0.5, λ_c=0` · C4 `λ_v=0.5, λ_c=0.2`. Contrastive / CLIP fuse / KD **tắt** trong ma trận này.

---

## 5. Kết quả đo được (test)

### 5.1 Overall

| Variant | mA | Acc | Prec | Rec | F1 | occ_mA | vis_mae | consist_gap |
|---------|------|------|------|------|------|--------|---------|-------------|
| C0 Base | 0.7751 | 0.9195 | 0.7006 | 0.8898 | 0.7839 | — | — | — |
| C1 Base+OccAug | 0.7675 | 0.9189 | 0.6977 | 0.8893 | 0.7819 | — | — | — |
| C2 AQ | 0.8012 | 0.9172 | 0.6931 | 0.9048 | 0.7849 | — | — | — |
| C3 AQ+Vis | **0.8024** | 0.9248 | 0.7184 | 0.8996 | 0.7989 | **0.7443** | 0.0085 | 0.1048 |
| C4 AQ+Vis+Consist | 0.8011 | 0.9248 | 0.7177 | 0.9011 | **0.7990** | 0.7220 | 0.0089 | **0.0468** |

### 5.2 Region mA (clean) — chỉ C2–C4 (eval Base chưa xuất region)

| Variant | head | upper | lower | shoes |
|---------|------|-------|-------|-------|
| C2 | 0.8197 | 0.8420 | 0.8457 | 0.7843 |
| C3 | 0.8221 | 0.8420 | 0.8521 | 0.7832 |
| C4 | 0.8174 | 0.8418 | 0.8512 | 0.7830 |

### 5.3 Đọc kết quả theo ý thầy

1. **Có xét visibility khi che khuất không?**  
   Có — VisHead + `L_vis` trên cặp che (C3/C4). vis_mae ≈ 0.0085–0.0089 trên nhãn hợp lệ; C3 đạt occ_mA tốt nhất **0.7443**.

2. **Attribute Query có giúp không?**  
   Có — C2 mA **0.8012** (+2.61pp so với C0 0.7751). Đây là bước nhảy lớn nhất trong ma trận.

3. **Visibility thêm trên AQ?**  
   C3 clean mA **0.8024** (nhỉnh C2) + có occ/vis metrics; F1 tăng rõ hơn (0.7849 → 0.7989).

4. **Consistency?**  
   C4 giảm consist_gap gần một nửa (0.1048 → **0.0468**), đúng mục tiêu “ổn định attr còn nguyên”. Đổi lại occ_mA giảm (0.7443 → 0.7220); clean mA gần như giữ nguyên.

5. **Occ-aug thông thường trên Base (C1)?**  
   Không cải thiện overall mA so với C0 (0.7675 < 0.7751) trong setup này.

6. **Đa tỉ lệ?**  
   Shoes vẫn yếu nhất (~0.78 vs lower ~0.85). Gate protocol `(C0.shoes − C4.shoes) > 0.04` **chưa kích hoạt** vì C0 chưa có region mA cùng pipeline → **giữ single-scale VMamba** (đúng “chỉ thêm khi có bằng chứng mất chi tiết”). Có thể bổ sung region eval cho C0/C1 nếu thầy muốn gate chặt hơn.

7. **Background branch**  
   Để tùy chọn (C5), chưa chạy — đúng “ablation riêng, không lẫn bảng chính”.

---

## 6. Metrics đã in để kiểm tra (theo thầy)

Mỗi run C2–C4 (và C3/C4 đầy đủ) xuất:

- Clean: mA, Acc, Precision, Recall, F1  
- Occluded (cặp): cùng 5 metrics (`occ_mA` …)  
- Visibility: `vis_mae`, `vis_acc`  
- Consistency: `consist_gap`  
- Region mA: head / upper / lower / shoes  

C0/C1 dùng eval Base → clean 5 metrics (+ `val_best_mA`); chưa có occ/vis/region trong JSON hiện tại.

File tổng hợp: `docs/aq_visibility/results.md`, `docs/aq_visibility_results.md`, `docs/aq_visibility/gap_analysis.md`.

---

## 7. Việc còn mở (nên báo thầy)

1. **Gán nhãn thủ công** 50–100 ảnh che tự nhiên → chạy `tools/occlusion/audit_manual_visibility.py` (CSV hiện mới template).  
2. **C5 background** nếu cần ablation đóng góp nhánh nền.  
3. (Tuỳ chọn) Eval region cho C0/C1 cùng pipeline để quyết định đa tỉ lệ theo đúng ngưỡng protocol.  
4. Nhánh improve/DeepMAR **không** lẫn vào bảng này (đã tách).

---

## 8. Kết luận ngắn cho thầy

Nhánh đã triển khai đủ checklist: giữ encoder, 57 attribute queries + head per-query, VisHead, loss ba thành phần, dữ liệu cặp theo partition (head/áo/quần/giày), uncertain loại khỏi visibility loss, và chạy đủ 5 mô hình so sánh.

Trên test MSP60k đo được: **AQ là đóng góp chính** (C2 ≈ +2.6pp mA vs Base); **VisHead (C3)** tốt nhất clean mA và occ_mA; **Consistency (C4)** cải thiện rõ khoảng cách clean–occ trên attr còn nguyên nhưng không tăng thêm mA sạch. Đa tỉ lệ và background giữ đúng tinh thần thầy: chưa bắt buộc / tùy chọn ablation.
