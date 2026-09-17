package com.example.backend.service;

import com.example.backend.entity.AdminSession;
import com.example.backend.repository.AdminSessionRepository;
import com.example.backend.security.TokenCodec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDateTime;

/**
 * Phát hành và thẩm định token đăng nhập của quản trị — bản sao có chủ đích của
 * {@link SessionService}.
 *
 * Vì sao không gộp một service: hai loại phiên trỏ tới hai bảng người dùng khác
 * nhau và có thời hạn khác nhau. Phiên admin đặt ngắn hơn nhiều (mặc định 12 giờ
 * so với 30 ngày của khách) vì một token admin bị rơi là mất cả cửa hàng, còn
 * token khách rơi thì cùng lắm lộ sổ địa chỉ của một người. Ghep chung thi doi
 * lai thanh mot tham so "ttl" dung cho ca hai, tuc la bo mat ly do ngan hon.
 *
 * Phần dễ lệch nhau (sinh token, tính hash, chặn độ dài) đã nằm ở
 * {@link TokenCodec}, nên ở đây chỉ còn đúng việc tra bảng.
 */
@Service
public class AdminSessionService {

    private final AdminSessionRepository sessionRepository;
    private final Duration ttl;

    public AdminSessionService(
            AdminSessionRepository sessionRepository,
            @Value("${app.admin.session-ttl-hours:12}") long ttlHours
    ) {
        this.sessionRepository = sessionRepository;
        this.ttl = Duration.ofHours(ttlHours);
    }

    /** Mở một phiên mới, trả về token GỐC — chỉ tồn tại một lần, DB không giữ nó. */
    @Transactional
    public String issue(Integer adminId) {
        LocalDateTime now = LocalDateTime.now();
        sessionRepository.deleteExpired(now);

        String token = TokenCodec.randomToken();
        sessionRepository.save(AdminSession.builder()
                .tokenHash(TokenCodec.sha256Hex(token))
                .adminId(adminId)
                .expiresAt(now.plus(ttl))
                .build());

        return token;
    }

    /** Token -> id admin đang đăng nhập. Null nghĩa là "không tin cái này". */
    @Transactional(readOnly = true)
    public Integer resolveAdminId(String token) {
        if (!TokenCodec.looksUsable(token)) {
            return null;
        }

        return sessionRepository
                .findByTokenHashAndExpiresAtAfter(
                        TokenCodec.sha256Hex(token.trim()), LocalDateTime.now())
                .map(AdminSession::getAdminId)
                .orElse(null);
    }

    /** Thu hồi đúng phiên đang dùng. Token lạ hoặc đã hết hạn thì im lặng. */
    @Transactional
    public void revoke(String token) {
        if (!TokenCodec.looksUsable(token)) {
            return;
        }

        sessionRepository.findByTokenHash(TokenCodec.sha256Hex(token.trim()))
                .ifPresent(sessionRepository::delete);
    }
}
