package com.example.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Tài khoản khách hàng (dùng để đăng nhập và sở hữu đơn hàng).
 *
 * Tách khỏi "người nhận hàng" (Recipient): một tài khoản giao được cho vô số
 * địa chỉ khác nhau, và người nhận thì không cần có tài khoản.
 *
 * Bảng tên là "user_account" chứ không phải "user": USER là từ khóa dành riêng
 * trong PostgreSQL (đồng nghĩa của current_user), tạo bảng "user" thì mọi lần
 * tham chiếu đều phải bọc ngoặc kép. Xem thêm db/manual/03_users.sql.
 */
@Entity
@Table(name = "user_account")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(nullable = false, unique = true, length = 50)
    private String username;

    /** Đã được lowercase lúc lưu, nên "A@b.com" và "a@b.com" không qua được unique. */
    @Column(nullable = false, unique = true, length = 100)
    private String email;

    /** Hash BCrypt. Không bao giờ rò ra ngoài qua response DTO. */
    @Column(nullable = false)
    private String password;

    /** Họ tên chủ tài khoản, dùng để gợi ý sẵn khi tạo người nhận đầu tiên. */
    @Column(name = "full_name", nullable = false, length = 100)
    private String fullName;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
}
