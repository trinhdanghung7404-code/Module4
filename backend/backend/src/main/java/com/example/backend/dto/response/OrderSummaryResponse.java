package com.example.backend.dto.response;

import com.example.backend.enums.OrderStatus;
import lombok.Builder;
import lombok.Getter;

import java.util.Map;

/**
 * Bộ đếm đơn hàng theo trạng thái, cho các thẻ trên trang chủ quản trị.
 *
 * byStatus luôn đủ cả năm trạng thái, kể cả trạng thái đang bằng 0: thiếu hẳn một
 * khoá thì thẻ "Đang vận chuyển" biến mất khỏi màn hình, và admin không phân biệt
 * được "chưa có đơn nào" với "server quên tính".
 */
@Getter
@Builder
public class OrderSummaryResponse {

    private final long total;
    private final Map<OrderStatus, Long> byStatus;

    /** Nhãn tiếng Việt đi kèm để UI khỏi giữ một bản dịch thứ hai. */
    private final Map<OrderStatus, String> labels;
}
