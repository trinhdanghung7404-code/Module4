package com.example.backend.repository;

import com.example.backend.entity.OrderStatusHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public interface OrderStatusHistoryRepository extends JpaRepository<OrderStatusHistory, Integer> {

    /**
     * Toàn bộ lineage của MỘT đơn, theo đúng thứ tự xảy ra.
     *
     * id là khoá phụ để hai bản ghi ghi cùng microsecond vẫn giữ được thứ tự —
     * created_at chỉ chính xác tới mức đó, mà lộn thứ tự "đang giao" với "đã giao"
     * thì timeline thành chuyện kể ngược.
     */
    List<OrderStatusHistory> findByOrderIdOrderByCreatedAtAscIdAsc(Integer orderId);

    /**
     * Lineage của NHIỀU đơn trong MỘT câu truy vấn, cho danh sách admin và trang
     * "Đơn hàng của tôi".
     *
     * Nếu gọi findByOrderId... trong vòng lặp thì 20 đơn trên một trang là 20 lệnh
     * đọc; ở đây service tự gom id về một lần rồi nhóm lại trong bộ nhớ.
     */
    List<OrderStatusHistory> findByOrderIdInOrderByCreatedAtAscIdAsc(Collection<Integer> orderIds);

    /**
     * Gom lịch sử của nhiều đơn thành một map — dùng cho mọi danh sách.
     *
     * LinkedHashMap để nhóm nào cũng giữ đúng thứ tự mà câu truy vấn đã sắp (theo
     * createdAt, rồi id): hai bản ghi cùng microsecond mà rơi vào HashMap rồi đảo
     * trước sau thì cái stepper trên màn hình sẽ kể ngược chuyện đã xảy ra.
     */
    default Map<Integer, List<OrderStatusHistory>> mapByOrderId(Collection<Integer> orderIds) {
        if (orderIds.isEmpty()) {
            return Map.of();
        }

        return findByOrderIdInOrderByCreatedAtAscIdAsc(orderIds).stream()
                .collect(Collectors.groupingBy(
                        OrderStatusHistory::getOrderId,
                        LinkedHashMap::new,
                        Collectors.toList()));
    }
}
