package com.example.backend.repository;

import com.example.backend.entity.AdminSession;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.Optional;

public interface AdminSessionRepository extends JpaRepository<AdminSession, Integer> {

    /** Chỉ trả về phiên còn hiệu lực — hết hạn là token biến mất, không có nhánh "quên kiểm tra hạn". */
    Optional<AdminSession> findByTokenHashAndExpiresAtAfter(String tokenHash, LocalDateTime now);

    /** Logout cần tìm cả dòng đã hết hạn để xoá hẳn, không để rác nằm lại. */
    Optional<AdminSession> findByTokenHash(String tokenHash);

    /**
     * Dọn mọi phiên của một admin.
     *
     * Chưa có chỗ gọi trong luồng hiện tại, nhưng đây đúng là câu phải chạy khi
     * admin đổi mật khẩu: không xoá thì token cũ vẫn sống tới hết hạn, tức là kẻ
     * đánh cắp token vẫn vào được dù nạn nhân vừa "đổi mật khẩu cho an toàn".
     */
    @Modifying
    @Query("delete from AdminSession s where s.adminId = :adminId")
    int deleteByAdminId(@Param("adminId") Integer adminId);

    /** Xem ghi chú ở UserSessionRepository#deleteExpired: dọn tiện tay lúc phát hành token. */
    @Modifying
    @Query("delete from AdminSession s where s.expiresAt < :now")
    int deleteExpired(@Param("now") LocalDateTime now);
}
