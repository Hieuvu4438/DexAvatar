# Báo Cáo Lỗi Fitting Mesh và Render (DPoser-X Pipeline)

Báo cáo này tài liệu hóa nguyên nhân tại sao khi áp dụng DPoser-X vào pipeline DexAvatar (NLF + WiLoR + Sapiens + SignHPoser + DPoser-X) thì không thể fitting được bất kỳ mesh nào (hoặc mesh không xuất hiện trên ảnh render).

---

## 1. Tóm tắt Hai Nguyên Nhân Cốt Lõi

Có hai lỗi độc lập cùng xảy ra đồng thời khiến mesh hoàn toàn biến mất khỏi camera:
1. **NLF Init của DPoser-X bị sai lệch camera translation (Depth Z quá gần):** Giá trị khởi tạo `transl` bị lỗi `Z = 0.085m` (khoảng 8.5 cm) thay vì `17.8m`. Vì translation bị đóng băng (frozen) trong suốt quá trình tối ưu hóa, mesh bị vẽ cực kỳ to và văng khỏi khung hình (off-screen).
2. **Logic lưu Mesh/Render bỏ qua tư thế cơ thể đã tinh chỉnh:** Trong `fit_single_frame.py`, khi chạy ở chế độ post-fit refinement (`use_dposerx_refine: True` và `use_dposerx_body: False`), phần logic lưu mesh và render đặt `body_pose = None`. Điều này ép mô hình SMPL-X về tư thế T-pose mặc định thay vì dùng tư thế NLF hay tư thế đã được DPoser-X tinh chỉnh.

---

## 2. Chi Tiết Lỗi 1: Tịnh Tiến Camera (Z = 0.085m) Lệch Chuẩn

### So sánh Dữ liệu Khởi tạo (low_153.pkl)
Khi đọc trực tiếp tham số camera từ hai thư mục khởi tạo NLF:

*   **Pipeline hoạt động đúng (`method_nlf_wilor`):**
    *   `transl`: `[-0.0028, 0.748, 17.819]` (Depth Z = **`17.8` mét** - Đúng chuẩn camera của SMPLer-X).
    *   `focal`: `[6695.373, 6695.373]`
    *   `princpt`: `[236.308, 187.839]`
*   **Pipeline DPoser-X bị lỗi (`method_nlf_vqvae_dposerx`):**
    *   `transl`: `[0.0103, 0.411, 0.0858]` (Depth Z = **`0.085` mét** - 8.5 cm trước camera!).
    *   `focal`: `[5000.0, 5000.0]`
    *   `princpt`: `[257.0, 150.0]`

### Lý do và Cơ chế
1. Trong file [Full_running_command_nlf_dposerx.sh](file:///home/haipd/DexAvatar/methods/Full_running_command_nlf_dposerx.sh#L10), pipeline trỏ nguồn khởi tạo NLF (`NLF_SOURCE`) sang:
   ```bash
   NLF_SOURCE="/home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx/${SIGN_NAME}"
   ```
2. Thư mục `method_nlf_vqvae_dposerx` này trước đó trích xuất NLF không tìm thấy thư mục camera của SMPLer-X trong `shared/` nên đã tự động tính toán ra Depth Z cực gần.
3. Vì trong [fit_single_frame.py](file:///home/haipd/DexAvatar/dexavatar_fitting/smplifyx/fit_single_frame.py#L653-L659), các tham số `transl` và `global_orient` **không được đưa vào danh sách tham số cần tối ưu (`final_params`)** mà bị đóng băng theo giá trị khởi tạo, khoảng cách 8.5 cm này được giữ nguyên suốt quá trình fitting. Mesh bị phóng đại hàng trăm lần do khoảng cách quá gần và bay ra ngoài camera.

---

## 3. Chi Tiết Lỗi 2: Logic Render/Mesh Saving Đặt `body_pose = None`

### Đoạn Code Lỗi
Trong [fit_single_frame.py](file:///home/haipd/DexAvatar/dexavatar_fitting/smplifyx/fit_single_frame.py#L962-L970), phần logic chuẩn bị dữ liệu xuất mesh và render:

```python
962:     if save_meshes or visualize:
963:         if use_signbposer:
964:             body_pose = signbposer.decode(
965:                 pose_embedding,
966:                 output_type='aa').view(1, -1)
967:         elif use_motionbert_prior or use_phd_prior or use_dposerx_body or use_vqvae_hand:
968:             body_pose = pose_embedding.view(1, -1)
969:         else:
970:             body_pose = None
```

### Lý do và Cơ chế
Khi chạy cấu hình `cfg_files/fit_smplx_vposer_x_dposerx.yaml`:
*   `use_dposerx_body: False`
*   `use_dposerx_refine: True`
*   `use_signbposer: False`
*   `use_vqvae_hand: False`

Điều này khiến toàn bộ điều kiện `if/elif` từ dòng 963-967 nhận giá trị `False`, đi thẳng vào nhánh `else` và đặt `body_pose = None`. 

Mặc dù trước đó ở dòng 791, kết quả tinh chỉnh DPoser-X đã được giải mã và ghi nhận vào dictionary kết quả:
```python
791:                         bp_refined = _dpr_prior.decode_to_pose(bp_tensor, num_steps=10)
792:                         if not torch.isnan(bp_refined).any():
793:                             result['body_pose'] = bp_refined.detach().cpu().numpy()
```
Nhưng đoạn code lưu mesh và render hoàn toàn không đọc từ `result['body_pose']` mà chỉ đọc từ `pose_embedding`. Do đó, mesh vẽ ra luôn bị ép về tư thế T-pose mặc định.

---

## 4. Hướng Dẫn Sửa Lỗi Chi Tiết (Step-by-Step Fix Guide)

### Bước 1: Thay đổi NLF_SOURCE để dùng Init đúng
Trong file [Full_running_command_nlf_dposerx.sh](file:///home/haipd/DexAvatar/methods/Full_running_command_nlf_dposerx.sh), đổi nguồn `NLF_SOURCE` sang pipeline Wilor để tái sử dụng NLF init đã khớp camera chuẩn (Z ~ 17.8m):

```diff
-NLF_SOURCE="/home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx/${SIGN_NAME}"
+NLF_SOURCE="/home/haipd/DexAvatar/outputs/method_nlf_wilor/${SIGN_NAME}"
```

### Bước 2: Cập nhật Logic lưu Mesh và Render trong `fit_single_frame.py`
Sửa đổi [fit_single_frame.py](file:///home/haipd/DexAvatar/dexavatar_fitting/smplifyx/fit_single_frame.py) từ dòng 962 để ưu tiên lấy các giá trị pose (`body_pose`, `left_hand_pose`, `right_hand_pose`) đã được tối ưu hóa hoặc refine từ `result` dictionary nếu chúng tồn tại:

```python
    if save_meshes or visualize:
        body_pose = None
        # Ưu tiên lấy body_pose đã tối ưu hóa/tinh chỉnh từ dictionary kết quả
        if result is not None and 'body_pose' in result:
            body_pose = torch.from_numpy(result['body_pose']).to(device=device, dtype=dtype)
        elif use_signbposer:
            body_pose = signbposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1)
        elif use_motionbert_prior or use_phd_prior or use_dposerx_body or use_vqvae_hand:
            body_pose = pose_embedding.view(1, -1)

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_signbposer
        if append_wrists and body_pose is not None:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        lhand_pose = None
        rhand_pose = None
        if result is not None and 'left_hand_pose' in result:
            lhand_pose = torch.from_numpy(result['left_hand_pose']).to(device=device, dtype=dtype)
        if result is not None and 'right_hand_pose' in result:
            rhand_pose = torch.from_numpy(result['right_hand_pose']).to(device=device, dtype=dtype)

        if use_hposer3d or use_vqvae_hand:
            if indp_sign_class != "0":
                    if lhand_pose is None:
                        if use_hposer3d:
                            lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                        else:
                            lhand_pose = hposer3d_vqvae.decode_aa(lhand_embedding3d).view(1, -1)
                    if rhand_pose is None:
                        if use_hposer3d:
                            rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                        else:
                            rhand_pose = rhand_pose = hposer3d_vqvae.decode_aa(rhand_embedding3d).view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, right_hand_pose=rhand_pose, left_hand_pose=lhand_pose)
            else:
                if hand_label == 'right_hand':
                    if rhand_pose is None:
                        if use_hposer3d:
                            rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                        else:
                            rhand_pose = hposer3d_vqvae.decode_aa(rhand_embedding3d).view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, right_hand_pose=rhand_pose)
                
                elif hand_label == 'left_hand':
                    if lhand_pose is None:
                        if use_hposer3d:
                            lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                        else:
                            lhand_pose = hposer3d_vqvae.decode_aa(lhand_embedding3d).view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, left_hand_pose=lhand_pose)
        else:
            model_output = body_model(body_pose=body_pose, return_verts=True)
```
