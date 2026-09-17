package com.example.backend.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

/**
 * Một dòng giỏ gửi lên. CHỈ có id và số lượng: giá, tên sản phẩm đều lấy từ DB,
 * nếu cho client gửi giá thì ai cũng sửa được đơn hàng về 0 đồng.
 */
@Getter
@Setter
public class OrderItemRequest {

    @NotNull(message = "Sản phẩm không được để trống")
    private Integer productId;

    @NotNull(message = "Số lượng không được để trống")
    @Min(value = 1, message = "Số lượng phải lớn hơn 0")
    @Max(value = 999, message = "Số lượng tối đa 999 sản phẩm mỗi dòng")
    private Integer quantity;
}
