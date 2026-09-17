package com.example.backend.dto;

import com.example.backend.enums.OrderStatus;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

/**
 * Đề nghị chuyển một đơn sang trạng thái khác, từ trang quản trị.
 *
 * Chỉ có đúng một trường bắt buộc là {@code toStatus}: người bấm nói lên ý định
 * "muốn sang trạng thái nào", còn được hay không là luật của OrderStatusRules ở
 * server. Không nhận nguyên entity hoặc cả cột statusUpdatedAt: client không được
 * quyền tự đóng dấu thời gian cho hành động của mình.
 */
@Getter
@Setter
public class AdminOrderStatusRequest {

    @NotNull(message = "Phải chọn trạng thái muốn chuyển tới")
    private OrderStatus toStatus;

    @Size(max = 500, message = "Ghi chú tối đa 500 ký tự")
    private String note;

    /** Chỉ có nghĩa ở bước bàn giao hàng — xem AdminOrderService#changeStatus. */
    @Size(max = 100, message = "Tên đơn vị vận chuyển tối đa 100 ký tự")
    private String shippingUnit;

    @Size(max = 60, message = "Mã vận đơn tối đa 60 ký tự")
    private String trackingCode;
}
