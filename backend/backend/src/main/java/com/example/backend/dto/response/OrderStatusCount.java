package com.example.backend.dto.response;

import com.example.backend.enums.OrderStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Một dòng "trạng thái -> bao nhiêu đơn", đích đến của câu group by trong
 * {@code OrderRepository#countGroupedByStatus}.
 *
 * Tồn tại vì JPA constructor expression cần một type đích; dùng List&lt;Object[]&gt;
 * thì mọi chỗ đọc phải nhớ index 0 là gì, index 1 là gì.
 */
@Getter
@AllArgsConstructor
public class OrderStatusCount {

    private final OrderStatus status;
    private final Long count;
}
