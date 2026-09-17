package com.example.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

/** Thêm hoặc sửa một người nhận hàng đã lưu. */
@Getter
@Setter
public class RecipientRequest {

    @NotBlank(message = "Tên người nhận không được để trống")
    @Size(max = 120, message = "Tên người nhận tối đa 120 ký tự")
    private String name;

    // Cùng một pattern với OrderCreateRequest: cùng một dữ liệu thì phải cùng một
    // quy tắc, nếu không có những số điện thoại lưu được vào sổ mà không đặt hàng được.
    @NotBlank(message = "Số điện thoại không được để trống")
    @Pattern(regexp = "^\\+?\\d[\\d .-]{6,18}$", message = "Số điện thoại không hợp lệ")
    @Size(max = 20, message = "Số điện thoại tối đa 20 ký tự")
    private String phone;

    @NotBlank(message = "Địa chỉ không được để trống")
    @Size(max = 255, message = "Địa chỉ tối đa 255 ký tự")
    private String address;

    @Size(max = 30, message = "Nhãn tối đa 30 ký tự")
    private String label;

    /** null = không nói gì; true = đặt làm mặc định, ưu tiên cho đơn sau. */
    private Boolean isDefault;
}
