package com.example.backend.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class OrderCreateRequest {

    // KHONG co truong userId o day. Truoc day client tu gui len id va server chi
    // kiem no co ton tai, tuc la bat ky ai cung gan duoc don cua minh vao tai
    // khoan cua nguoi khac. Bay gio don nhan user_id tu token (xem
    // ShopOrderController), va mot field thura trong DTO la mot loi phong ve sau.

    // Ba field dưới đây là NGƯỜI NHẬN HÀNG, không phải chủ tài khoản: mua quà
    // tặng thì họ tên ở đây khác hoàn toàn người đang đăng nhập.
    @NotBlank(message = "Họ tên không được để trống")
    @Size(max = 120, message = "Họ tên tối đa 120 ký tự")
    private String customerName;

    // Số điện thoại Việt Nam: 0xxxxxxxxx, +84xxxxxxxxx, có thể có khoảng trắng
    // hoặc gạch khi người dùng gõ cho dễ đọc.
    @NotBlank(message = "Số điện thoại không được để trống")
    @Pattern(regexp = "^\\+?\\d[\\d .-]{6,18}$", message = "Số điện thoại không hợp lệ")
    @Size(max = 20, message = "Số điện thoại tối đa 20 ký tự")
    private String phone;

    @NotBlank(message = "Địa chỉ không được để trống")
    @Size(max = 255, message = "Địa chỉ tối đa 255 ký tự")
    private String address;

    @Size(max = 500, message = "Ghi chú tối đa 500 ký tự")
    private String note;

    @NotEmpty(message = "Đơn hàng phải có ít nhất một sản phẩm")
    @Size(max = 50, message = "Một đơn hàng tối đa 50 dòng sản phẩm")
    @Valid
    private List<OrderItemRequest> items;
}
