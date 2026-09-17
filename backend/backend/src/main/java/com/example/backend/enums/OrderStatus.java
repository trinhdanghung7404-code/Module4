package com.example.backend.enums;

/**
 * Trạng thái đơn hàng. Lưu trong DB dạng chữ (EnumType.STRING) để thứ tự khai báo
 * có thay đổi cũng không làm hỏng dữ liệu cũ.
 */
public enum OrderStatus {
    PENDING,
    CONFIRMED,
    SHIPPING,
    DELIVERED,
    CANCELLED
}
