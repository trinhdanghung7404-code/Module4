package com.example.backend.entity;

import com.example.backend.enums.OrderStatus;
import com.example.backend.enums.OrderStatusActorType;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Một bước đổi trạng thái của đơn hàng.
 *
 * Insert-only: không update, không delete. Đó là toàn bộ lý do tồn tại của bảng —
 * cột {@code orders.status} chỉ giữ giá trị CUỐI, nên "đơn này bị hủy lúc mấy giờ,
 * từ trạng thái nào, ai hủy" là thứ không thể suy ra lại nếu không ghi lại từng lần.
 * Muốn sửa được dòng lịch sử thì muốn nó thành bảng trạng thái, và một bảng có thể
 * bị viết lại thì không còn là bằng chứng nữa.
 *
 * actorId KHÔNG phải FK: admin hoặc khách bị xoá tài khoản thì dòng lịch sử phải còn
 * nguyên, nên kèm actorLabel là tên chép tại thời điểm hành động. Nhờ vậy đọc lại
 * sau hai năm vẫn biết "hung.nt" nào đã bấm xác nhận, kể cả khi tài khoản đó không
 * còn tồn tại.
 */
@Entity
@Table(name = "order_status_history")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrderStatusHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /**
     * Đơn nào. Để số nguyên trơ như Order.userId: mỗi lần mở danh sách đơn không
     * cần load nguyên entity Order, và bảng này được nạp theo kiểu "một câu truy
     * vấn cho cả trang đơn" (xem AdminOrderService#attachHistories).
     */
    @Column(name = "order_id", nullable = false)
    private Integer orderId;

    /** Null ở dòng đầu tiên của một đơn: chưa có trạng thái nào để đi từ đó. */
    @Enumerated(EnumType.STRING)
    @Column(name = "from_status", length = 20)
    private OrderStatus fromStatus;

    @Enumerated(EnumType.STRING)
    @Column(name = "to_status", nullable = false, length = 20)
    private OrderStatus toStatus;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 10)
    private OrderStatusActorType actorType;

    @Column(name = "actor_id")
    private Integer actorId;

    /** Tên người hành động tại thời điểm đó — bản chép, không tra lại theo id. */
    @Column(name = "actor_label", length = 120)
    private String actorLabel;

    @Column(length = 500)
    private String note;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
}
