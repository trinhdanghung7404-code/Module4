package com.example.backend.enums;

/**
 * Ai là người làm thay đổi trạng thái đơn hàng, ghi lại trong
 * {@code order_status_history.actor_type}.
 *
 * Ba giá trị đóng khung đúng bằng CHECK constraint trong DB: thêm giá trị ở Java
 * mà quên sửa constraint thì lần ghi đầu tiên sẽ bị PostgreSQL chặn, còn thêm ở DB
 * mà không có trong enum thì đọc phải một dòng không phân tích được. Đặt cạnh nhau
 * ở đây để sửa DB là phải nhìn thấy file này.
 */
public enum OrderStatusActorType {
    /** Nhân viên bấm trên trang quản trị. */
    ADMIN,

    /** Chính khách hàng tác động (ví dụ hủy đơn còn chờ xác nhận). */
    USER,

    /** Không có người thật: backfill dữ liệu cũ, hoặc hệ thống tự ghi. */
    SYSTEM
}
