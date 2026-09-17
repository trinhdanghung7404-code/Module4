package com.example.backend.service;

import com.example.backend.entity.UserSession;
import com.example.backend.repository.UserSessionRepository;
import com.example.backend.security.TokenCodec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDateTime;

/**
 * Phát hành và thẩm định token đăng nhập của khách.
 *
 * Token opaque, không phải JWT: hết hạn là xoá dòng, không có cảnh "token vẫn sống
 * sau khi khách đăng xuất". Đổi lại mọi request phải tra DB một lần — ở quy mô của
 * dự án này cái giá đó rẻ hơn nhiều so với việc giữ một danh sách token thu hồi.
 *
 * DB chỉ nhìn thấy SHA-256 của token. Đọc trộm bảng user_session vì vậy không dùng
 * được gì: muốn giả mạo một phiên thì phải đoán đúng 256 bit ngẫu nhiên, chứ không
 * phải copy một chuỗi có sẵn trong bảng.
 *
 * Sinh token và tính hash đặt ở {@link TokenCodec} để AdminSessionService dùng
 * chung đúng một luật, không có bản thứ hai lệch nhau.
 */
@Service
public class SessionService {

    private final UserSessionRepository sessionRepository;
    private final Duration ttl;

    public SessionService(
            UserSessionRepository sessionRepository,
            @Value("${app.user.session-ttl-days:30}") long ttlDays
    ) {
        this.sessionRepository = sessionRepository;
        this.ttl = Duration.ofDays(ttlDays);
    }

    /**
     * Mở một phiên mới và trả về token GỐC — giá trị duy nhất không bao giờ được
     * lưu lại, nên gọi đúng một lần rồi gửi cho client.
     *
     * Một tài khoản có nhiều phiên cùng lúc (hai trình duyệt, hai tab trình duyệt
     * riêng) là chủ đích: đăng xuất ở máy này không được đá khách ra khỏi máy khác.
     */
    @Transactional
    public String issue(Integer userId) {
        LocalDateTime now = LocalDateTime.now();

        // Tiện tay dọn lúc phát hành: không có scheduler nào trong dự án, và nếu
        // đợi đến khi bảng phình lên mới dọn thì không ai gọi cả.
        sessionRepository.deleteExpired(now);

        String token = TokenCodec.randomToken();
        sessionRepository.save(UserSession.builder()
                .tokenHash(TokenCodec.sha256Hex(token))
                .userId(userId)
                .expiresAt(now.plus(ttl))
                .build());

        return token;
    }

    /**
     * Token -> id tài khoản đang đăng nhập. Null nghĩa là "không tin cái này".
     *
     * Không ném lỗi ở đây: việc báo 401 thuộc về interceptor, còn caller nội bộ
     * (ví dụ POST /api/orders cho khách lẻ) cần một câu trả lời im lặng.
     */
    @Transactional(readOnly = true)
    public Integer resolveUserId(String token) {
        if (token == null) {
            return null;
        }

        if (!TokenCodec.looksUsable(token)) {
            return null;
        }

        return sessionRepository
                .findByTokenHashAndExpiresAtAfter(
                        TokenCodec.sha256Hex(token.trim()), LocalDateTime.now())
                .map(UserSession::getUserId)
                .orElse(null);
    }

    /** Thu hồi theo token. Token lạ hoặc đã hết hạn thì im lặng: logout luôn "thành công". */
    @Transactional
    public void revoke(String token) {
        if (token == null) {
            return;
        }

        if (!TokenCodec.looksUsable(token)) {
            return;
        }

        sessionRepository.findByTokenHash(TokenCodec.sha256Hex(token.trim()))
                .ifPresent(sessionRepository::delete);
    }
}
