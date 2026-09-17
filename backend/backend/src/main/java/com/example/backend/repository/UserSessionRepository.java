package com.example.backend.repository;

import com.example.backend.entity.UserSession;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.Optional;

public interface UserSessionRepository extends JpaRepository<UserSession, Integer> {

    /**
     * Chỉ trả về phiên còn hiệu lực: điều kiện expires_at nằm trong chính câu
     * truy vấn để hết hạn là token biến mất, không có nhánh "quên kiểm tra hạn"
     * ở tầng service.
     */
    Optional<UserSession> findByTokenHashAndExpiresAtAfter(String tokenHash, LocalDateTime now);

    /** Logout cần tìm cả dòng đã hết hạn để xoá hẳn, không để rác nằm lại. */
    Optional<UserSession> findByTokenHash(String tokenHash);

    /**
     * Dọn các phiên đã hết hạn. Dự án này không có scheduler nào, nên lệnh dọn
     * được gọi tiện tay lúc phát hành token mới — cứ đăng nhập là dọn quá khứ,
     * không thể tích lại vô hạn.
     */
    @Modifying
    @Query("delete from UserSession s where s.expiresAt < :now")
    int deleteExpired(@Param("now") LocalDateTime now);
}
