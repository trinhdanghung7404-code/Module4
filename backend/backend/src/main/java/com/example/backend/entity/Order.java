package com.example.backend.entity;

import com.example.backend.enums.OrderStatus;
import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Một đơn hàng của khách. Bảng là "orders" vì ORDER là từ khóa dành riêng
 * trong PostgreSQL. Xem thêm db/manual/02_orders.sql.
 */
@Entity
@Table(name = "orders")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Order {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /** Mã đơn in cho khách, dạng DH-000001, sinh từ id sau khi insert. */
    @Column(name = "order_code", nullable = false, unique = true, length = 30)
    private String orderCode;

    /**
     * Tài khoản đã đặt hàng. Null với đơn đặt không đăng nhập.
     *
     * Để số nguyên trơ thay vì @ManyToOne User: đơn chỉ cần id để liệt kê
     * "Đơn hàng của tôi", còn khai báo quan hệ thì mỗi lần đọc đơn lại kéo theo
     * một lần load user. DB buộc FK thật và ON DELETE SET NULL, nên khách xoá tài
     * khoản thì đơn vẫn còn.
     */
    @Column(name = "user_id")
    private Integer userId;

    /** Tên người nhận hàng, không nhất thiết là chủ tài khoản. */
    @Column(name = "customer_name", nullable = false, length = 120)
    private String customerName;

    @Column(nullable = false, length = 20)
    private String phone;

    @Column(nullable = false, length = 255)
    private String address;

    @Column(length = 500)
    private String note;

    /** Tổng tiền do server tính từ giá trong DB, không lấy từ request. */
    @Column(name = "total_amount", nullable = false, precision = 12, scale = 2)
    private BigDecimal totalAmount;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private OrderStatus status;

    /**
     * Lúc đơn bước vào {@code status} hiện tại. Null với đơn tạo trước khi có luồng
     * theo dõi (đã được backfill về created_at trong 06_order_fulfillment.sql).
     */
    @Column(name = "status_updated_at")
    private LocalDateTime statusUpdatedAt;

    /**
     * Đơn vị vận chuyển admin nhập tay khi bàn giao hàng — KHÔNG phải tích hợp API
     * của GHN/GHTK: không có hợp đồng, không có callback, và gọi API thật thì demo
     * chết khi mất mạng.
     */
    @Column(name = "shipping_unit", length = 100)
    private String shippingUnit;

    @Column(name = "tracking_code", length = 60)
    private String trackingCode;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Builder.Default
    @OneToMany(cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @JoinColumn(name = "order_id", nullable = false)
    private List<OrderItem> items = new ArrayList<>();

    @PrePersist
    void applyDefaults() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
        if (status == null) {
            status = OrderStatus.PENDING;
        }
    }
}
