package com.example.backend.repository;

import com.example.backend.entity.Recipient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface RecipientRepository extends JpaRepository<Recipient, Integer> {

    /** Mặc định nên nổi lên đầu danh sách; id tăng dần để thứ tự không đổi mỗi lần tải. */
    List<Recipient> findByUserIdOrderByIsDefaultDescIdAsc(Integer userId);

    Optional<Recipient> findByIdAndUserId(Integer id, Integer userId);

    /**
     * Chặn lưu trùng một người nhận. Thứ tự tham số BẮT BUỘC khớp thứ tự tên
     * property trong tên method: Spring Data ghép theo vị trí chứ không theo tên
     * biến, đảo ngược là query chạy sai mà không báo lỗi.
     */
    boolean existsByUserIdAndNameAndPhoneAndAddress(
            Integer userId, String name, String phone, String address);

    /**
     * JPQL UPDATE chạy thẳng xuống DB ngay lúc gọi, không chờ Hibernate flush.
     * Phải dùng bản này thay vì sửa từng entity trong bộ nhớ, vì lệnh INSERT người
     * nhận mới có thể bị đẩy xuống TRƯỚC các lệnh UPDATE unset mặc định, và index
     * unique "một mặc định mỗi tài khoản" sẽ từ chối cả giao dịch.
     */
    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query("update Recipient r set r.isDefault = false "
            + "where r.user.id = :userId and r.isDefault = true")
    int clearDefaults(@Param("userId") Integer userId);
}
