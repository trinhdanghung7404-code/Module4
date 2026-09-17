package com.example.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Một phiên đăng nhập của khách.
 *
 * Chỉ lưu SHA-256 của token, không lưu token gốc: token là thứ duy nhất chứng minh
 * "đúng người này đang gọi", nên rò DB (kèm read-only cũng đủ) không được phép cho
 * kẻ tấn công dựng lại token hợp lệ. Token gốc chỉ tồn tại một lần lúc đăng ký /
 * đăng nhập và đi thẳng ra client.
 *
 * userId để là số nguyên trơ, giống Order: mỗi lần gọi API chỉ cần biết id nào đang
 * đăng nhập, khai báo @ManyToOne User thì mỗi request lại thêm một lần load user.
 * DB đã buộc FK thật, nên xoá tài khoản là mọi phiên của nó hết hiệu lực theo.
 */
@Entity
@Table(name = "user_session")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserSession {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "token_hash", nullable = false, unique = true, length = 64)
    private String tokenHash;

    @Column(name = "user_id", nullable = false)
    private Integer userId;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    /** Qua mốc này là token bị từ chối; dòng cũ nằm lại cho tới khi bị dọn. */
    @Column(name = "expires_at", nullable = false)
    private LocalDateTime expiresAt;
}
