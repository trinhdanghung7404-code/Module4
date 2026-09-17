package com.example.backend.repository;

import com.example.backend.dto.response.OrderStatusCount;
import com.example.backend.entity.Order;
import com.example.backend.enums.OrderStatus;
import jakarta.persistence.LockModeType;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface OrderRepository extends JpaRepository<Order, Integer> {

    Optional<Order> findByOrderCode(String orderCode);

    boolean existsByOrderCode(String orderCode);

    /**
     * "Đơn hàng của tôi" — see OrderService.myOrders for why the list is capped
     * and why the filter lives here instead of in the caller.
     *
     * Id đi kèm để hai đơn tạo cùng một giây không đổi chỗ cho nhau giữa hai lần
     * tải: created_at chỉ chính xác tới microsecond và vẫn có thể trùng.
     */
    List<Order> findTop50ByUserIdOrderByCreatedAtDescIdDesc(Integer userId);

    /**
     * Đọc đơn kèm KHOÁ DÒNG (SELECT ... FOR UPDATE), dùng riêng cho bước đổi trạng
     * thái.
     *
     * Vì sao phải khoá: hai admin cùng bấm "Hủy" trên một đơn thì cả hai đọc được
     * status = PENDING, cả hai thấy phép chuyển là hợp lệ, và cả hai cộng trả tồn
     * kho — kho tự nhiều thêm đúng số hàng của một đơn mà không ai đặt hàng cả.
     *
     * Có thể chặn bằng UPDATE ... WHERE status = :from giống decreaseStock, nhưng ở
     * đây service còn phải đọc items để trả hàng và ghi lịch sử, nên khoá cả dòng
     * cho một transaction là cách nói thẳng với vấn đề: người đến sau chờ, rồi nhìn
     * thấy trạng thái mới và bị chặn bằng thông báo rõ ràng thay vì âm thầm ghi 0
     * dòng.
     */
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select o from Order o where o.id = :id")
    Optional<Order> findByIdForUpdate(@Param("id") Integer id);

    /**
     * Danh sách đơn cho trang quản trị: lọc theo trạng thái, tìm theo mã đơn / tên
     * khách / số điện thoại, mới nhất lên đầu.
     *
     * status = null nghĩa là "không lọc" — để nguyên tham số null thay vì sentinel 0
     * kiểu findShopCards vì ở đây tham số là enum, không có giá trị 0 tự nhiên.
     *
     * pattern do service dựng theo đúng cách ProductService vẫn làm: "%" + từ khoá +
     * "%", nên để trống thành "%%" và khớp mọi dòng — vì vậy không cần nhánh
     * "search = '' thì bỏ qua điều kiện" trong câu truy vấn. ba cột đều NOT NULL
     * nên LIKE không âm thầm loại bỏ đơn nào.
     *
     * Tìm bằng LIKE nên chỉ khớp dấu bình thường: gõ "giao hang" không ra "Giao hàng".
     *
     * ORDER BY nằm sẵn trong câu truy vấn nên caller phải truyền Pageable KHÔNG kèm
     * sort, nếu không Spring sẽ nối thêm ORDER BY thứ hai vào sau cái này.
     */
    @Query(
            value = """
                    SELECT o
                    FROM Order o
                    WHERE (:status IS NULL OR o.status = :status)
                      AND (LOWER(o.orderCode) LIKE :pattern
                           OR LOWER(o.customerName) LIKE :pattern
                           OR LOWER(o.phone) LIKE :pattern)
                    ORDER BY o.createdAt DESC, o.id DESC
                    """,
            countQuery = """
                    SELECT COUNT(o)
                    FROM Order o
                    WHERE (:status IS NULL OR o.status = :status)
                      AND (LOWER(o.orderCode) LIKE :pattern
                           OR LOWER(o.customerName) LIKE :pattern
                           OR LOWER(o.phone) LIKE :pattern)
                    """
    )
    Page<Order> searchForAdmin(
            @Param("status") OrderStatus status,
            @Param("pattern") String pattern,
            Pageable pageable
    );

    /**
     * Đếm theo trạng thái cho các thẻ trên trang chủ admin: MỘT câu group by thay vì
     * chạy năm câu count, mỗi câu một round-trip.
     *
     * Trạng thái không có đơn nào sẽ VẮNG MẶT khỏi kết quả (group by chỉ trả về nhóm
     * có dòng), nên service phải tự khởi tạo đủ 5 trạng thái = 0.
     */
    @Query("""
            SELECT new com.example.backend.dto.response.OrderStatusCount(o.status, COUNT(o))
            FROM Order o
            GROUP BY o.status
            """)
    List<OrderStatusCount> countGroupedByStatus();
}
