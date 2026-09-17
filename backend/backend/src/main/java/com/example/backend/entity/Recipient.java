package com.example.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Một người nhận hàng đã lưu của một tài khoản: tên + SĐT + địa chỉ.
 *
 * Đây là thứ khách chọn ở màn thanh toán để khỏi phải gõ lại. Đơn hàng vẫn
 * LƯU SNAPSHOT (orders.customer_name/phone/address) chứ không lưu FK tới bảng
 * này, vì xoá hoặc sửa người nhận thì đơn cũ không được phép đổi nơi giao.
 */
@Entity
@Table(name = "recipient")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Recipient {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /** Xoá tài khoản thì xoá luôn người nhận của nó: không còn ai dùng lại. */
    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false, length = 120)
    private String name;

    @Column(nullable = false, length = 20)
    private String phone;

    @Column(nullable = false, length = 255)
    private String address;

    /** Nhãn gọi nhanh kiểu "Nhà", "Cơ quan" để phân biệt khi chọn. */
    @Column(length = 30)
    private String label;

    /**
     * Kiểu wrapper chứ không phải boolean nguyên sinh: Lombok sinh getIsDefault()
     * nên JSON ra key "isDefault" rõ ràng, không thành "default".
     */
    @Column(name = "is_default", nullable = false)
    private Boolean isDefault;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
}
