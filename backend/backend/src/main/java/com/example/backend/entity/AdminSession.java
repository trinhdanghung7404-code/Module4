package com.example.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Một phiên đăng nhập của quản trị.
 *
 * Cùng kiểu với {@link UserSession}: chỉ lưu SHA-256 của token, userId để dạng số
 * nguyên trơ thay vì @ManyToOne Admin. Hai bảng tách riêng chứ không dồn vào một
 * bảng "session" dùng chung rồi thêm cột role: khách và admin là hai bảng người
 * dùng khác nhau (user_account / admin), một cột id trỏ được tới hai bảng là thứ
 * DB không kiểm tra được và ứng dụng thì rất dễ quên kiểm tra.
 */
@Entity
@Table(name = "admin_session")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AdminSession {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "token_hash", nullable = false, unique = true, length = 64)
    private String tokenHash;

    @Column(name = "admin_id", nullable = false)
    private Integer adminId;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    /** Qua mốc này là token bị từ chối; dòng cũ nằm lại cho tới khi bị dọn. */
    @Column(name = "expires_at", nullable = false)
    private LocalDateTime expiresAt;
}
