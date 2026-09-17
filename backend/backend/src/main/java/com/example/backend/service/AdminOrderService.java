package com.example.backend.service;

import com.example.backend.dto.AdminOrderStatusRequest;
import com.example.backend.dto.response.AdminOrderPageResponse;
import com.example.backend.dto.response.OrderResponse;
import com.example.backend.dto.response.OrderStatusCount;
import com.example.backend.dto.response.OrderSummaryResponse;
import com.example.backend.entity.Admin;
import com.example.backend.entity.Order;
import com.example.backend.entity.OrderItem;
import com.example.backend.entity.OrderStatusHistory;
import com.example.backend.entity.Product;
import com.example.backend.enums.OrderStatus;
import com.example.backend.enums.OrderStatusActorType;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.AdminRepository;
import com.example.backend.repository.OrderRepository;
import com.example.backend.repository.OrderStatusHistoryRepository;
import com.example.backend.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Hồ sơ đơn hàng phía quản trị: xem danh sách, xem chi tiết, và BƯỚC TIẾP THEO của
 * luồng PENDING -> CONFIRMED -> SHIPPING -> DELIVERED (hoặc CANCELLED).
 *
 * Ba nguyên tắc xuyên suốt file này:
 *
 * 1) Luật dịch chuyển nằm ở OrderStatusRules, không nằm ở nút bấm. Client gửi được
 *    bất kỳ toStatus nào; ở đây từ chối những bước không hợp lệ.
 *
 * 2) Mọi lần đổi trạng thái đều để lại một dòng order_status_history. Không có dòng
 *    đó thì "đơn này hủy lúc mấy giờ, ai hủy" là câu hỏi không trả lời được, và
 *    khách chỉ còn thấy chữ "Đã hủy" không kèm bằng chứng.
 *
 * 3) Hủy đơn là nghiệp vụ DUY NHẤT chạm vào tồn kho sau lúc đặt. Hàng đã trừ khỏi
 *    kho lúc đặt (OrderService#create) nên phải được cộng lại ở đây, trong cùng một
 *    transaction với việc đổi trạng thái — tách ra hai lần ghi thì tiến trình chết
 *    giữa chừng để lại đơn đã hủy mà hàng không về kho.
 */
@Service
@RequiredArgsConstructor
public class AdminOrderService {

    private static final int DEFAULT_PAGE_SIZE = 20;

    /** Chặn một request hỏi toàn bộ DB về bộ nhớ chỉ bằng ?size=100000. */
    private static final int MAX_PAGE_SIZE = 50;

    /** Đúng bằng độ dài cột order_status_history.note varchar(500). */
    private static final int MAX_NOTE_LENGTH = 500;

    private final OrderRepository orderRepository;
    private final OrderStatusHistoryRepository statusHistoryRepository;
    private final ProductRepository productRepository;
    private final AdminRepository adminRepository;

    @Transactional(readOnly = true)
    public AdminOrderPageResponse list(OrderStatus status, String search, int page, int size) {
        int safeSize = size <= 0 ? DEFAULT_PAGE_SIZE : Math.min(size, MAX_PAGE_SIZE);
        Pageable pageable = PageRequest.of(Math.max(page, 0), safeSize);

        Page<Order> found = orderRepository.searchForAdmin(
                status, buildSearchPattern(search), pageable);

        Map<Integer, List<OrderStatusHistory>> histories = statusHistoryRepository
                .mapByOrderId(found.getContent().stream().map(Order::getId).toList());

        List<OrderResponse> content = found.getContent().stream()
                .map(order -> OrderResponse.from(
                        order, histories.getOrDefault(order.getId(), List.of())))
                .toList();

        return AdminOrderPageResponse.builder()
                .content(content)
                .page(found.getNumber())
                .size(found.getSize())
                .totalElements(found.getTotalElements())
                .totalPages(found.getTotalPages())
                .last(found.isLast())
                .build();
    }

    @Transactional(readOnly = true)
    public OrderResponse detail(Integer orderId) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new BusinessException("Không tìm thấy đơn hàng #" + orderId));

        return OrderResponse.from(order,
                statusHistoryRepository.findByOrderIdOrderByCreatedAtAscIdAsc(orderId));
    }

    @Transactional(readOnly = true)
    public OrderSummaryResponse summary() {
        // Khởi tạo đủ 5 trạng thái = 0: group by chỉ trả về nhóm có dòng, mà thiếu
        // khoá thì thẻ "Đang vận chuyển" biến mất khỏi màn hình và admin không phân
        // biệt được "chưa có đơn nào" với "server quên tính".
        Map<OrderStatus, Long> byStatus = new EnumMap<>(OrderStatus.class);
        for (OrderStatus status : OrderStatus.values()) {
            byStatus.put(status, 0L);
        }

        long total = 0;
        for (OrderStatusCount row : orderRepository.countGroupedByStatus()) {
            byStatus.put(row.getStatus(), row.getCount());
            total += row.getCount();
        }

        return OrderSummaryResponse.builder()
                .total(total)
                .byStatus(byStatus)
                .labels(OrderStatusRules.labels())
                .build();
    }

    /**
     * Một bước chuyển trạng thái.
     *
     * adminId đến từ token (AdminSessionInterceptor), không đến từ body: đã ghi nhận
     * "ai làm việc này" thì phải ghi người mà server biết, không phải người mà
     * request tự xưng.
     */
    @Transactional
    public OrderResponse changeStatus(Integer orderId,
                                      AdminOrderStatusRequest request,
                                      Integer adminId) {
        // FOR UPDATE: hai admin cùng bấm một lúc thì người đến sau chờ, rồi đọc được
        // trạng thái MỚI và bị luật chặn lại bằng thông báo rõ ràng, thay vì cùng
        // cộng trả một lần hàng vào kho.
        Order order = orderRepository.findByIdForUpdate(orderId)
                .orElseThrow(() -> new BusinessException("Không tìm thấy đơn hàng #" + orderId));

        OrderStatus from = order.getStatus();
        OrderStatus to = request.getToStatus();
        OrderStatusRules.assertAllowed(from, to);

        String shippingUnit = blankToNull(request.getShippingUnit());
        String trackingCode = blankToNull(request.getTrackingCode());
        String note = blankToNull(request.getNote());

        if (to == OrderStatus.SHIPPING) {
            // Bắt buộc ghi rõ: "đã bàn giao cho đơn vị vận chuyển" mà không biết là
            // đơn vị nào thì cột này chỉ còn là một mốc thời gian vô nghĩa.
            if (shippingUnit == null) {
                throw new BusinessException(
                        "Phải ghi rõ đơn vị vận chuyển trước khi bàn giao hàng.");
            }
            order.setShippingUnit(shippingUnit);
            order.setTrackingCode(trackingCode);
        } else if (shippingUnit != null || trackingCode != null) {
            // Làm ngơ dữ liệu người dùng vừa gõ là cách nhanh nhất để họ tin hệ thống
            // đã lưu trong khi nó không lưu gì.
            throw new BusinessException("Chỉ nhập đơn vị vận chuyển và mã vận đơn ở bước "
                    + OrderStatusRules.label(OrderStatus.SHIPPING) + ".");
        }

        if (OrderStatusRules.returnsStock(to)) {
            note = appendUnrestorable(note, restoreStock(order));
        }

        order.setStatus(to);
        order.setStatusUpdatedAt(LocalDateTime.now());

        statusHistoryRepository.save(OrderStatusHistory.builder()
                .orderId(order.getId())
                .fromStatus(from)
                .toStatus(to)
                .actorType(OrderStatusActorType.ADMIN)
                .actorId(adminId)
                .actorLabel(actorLabel(adminId))
                .note(note)
                .build());

        return OrderResponse.from(order,
                statusHistoryRepository.findByOrderIdOrderByCreatedAtAscIdAsc(order.getId()));
    }

    /**
     * Cộng trả tồn kho cho từng dòng của đơn bị hủy; trả về tên những mặt hàng KHÔNG
     * trả được.
     *
     * product == null là chuyện bình thường ở đây: FK của order_item là ON DELETE
     * SET NULL nên đơn cũ vẫn sống sau khi admin xóa vật lý sản phẩm. Không có mặt
     * hàng nào để cộng trả, và cũng không được phép bịa ra một sản phẩm mới chỉ cho
     * xong việc.
     *
     * 0 dòng bị ảnh hưởng cũng cùng một nghĩa (sản phẩm vừa bị xóa giữa hai lần đọc).
     * Cả hai trường hợp đều được ghi vào note của dòng lịch sử chứ không chặn việc
     * hủy: khách đã trả hàng rồi, đơn vẫn phải đóng được.
     */
    private List<String> restoreStock(Order order) {
        List<String> unrestorable = new ArrayList<>();

        for (OrderItem item : order.getItems()) {
            Product product = item.getProduct();

            // || ngắn mạch: product null thì không có id để gọi increaseStock.
            if (product == null
                    || productRepository.increaseStock(product.getId(), item.getQuantity()) == 0) {
                unrestorable.add(item.getProductName());
            }
        }

        return unrestorable;
    }

    private String appendUnrestorable(String note, List<String> unrestorable) {
        if (unrestorable.isEmpty()) {
            return note;
        }

        String extra = "Không trả được kho: " + String.join(", ", unrestorable)
                + " (sản phẩm đã bị xóa)";
        String merged = note == null ? extra : note + " — " + extra;

        // Cắt theo đúng độ dài cột: chuỗi dài hơn sẽ làm PostgreSQL báo "value too
        // long" lúc commit, tức là lúc người dùng chỉ còn thấy một lỗi không tên.
        return merged.length() <= MAX_NOTE_LENGTH
                ? merged
                : merged.substring(0, MAX_NOTE_LENGTH - 3) + "...";
    }

    private String actorLabel(Integer adminId) {
        return adminRepository.findById(adminId)
                .map(Admin::getUsername)
                .orElse("admin#" + adminId);
    }

    /** Cùng cách dựng pattern với ProductService để hai ô tìm kiếm không lệch nhau. */
    private String buildSearchPattern(String search) {
        String keyword = search == null ? "" : search.trim().toLowerCase(Locale.ROOT);
        return "%" + keyword + "%";
    }

    private String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
