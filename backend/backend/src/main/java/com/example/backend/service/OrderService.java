package com.example.backend.service;

import com.example.backend.dto.OrderCreateRequest;
import com.example.backend.dto.OrderItemRequest;
import com.example.backend.dto.response.OrderResponse;
import com.example.backend.entity.Order;
import com.example.backend.entity.OrderItem;
import com.example.backend.entity.OrderStatusHistory;
import com.example.backend.entity.Product;
import com.example.backend.enums.OrderStatus;
import com.example.backend.enums.OrderStatusActorType;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.OrderRepository;
import com.example.backend.repository.OrderStatusHistoryRepository;
import com.example.backend.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class OrderService {

    private final OrderRepository orderRepository;
    private final ProductRepository productRepository;
    private final OrderStatusHistoryRepository statusHistoryRepository;

    /**
     * Tạo đơn hàng.
     *
     * Toàn bộ chạy trong MỘT transaction: sản phẩm nào không đủ hàng thì ném
     * BusinessException và rollback, nên không có cảnh đã trừ 2 sản phẩm rồi
     * sản phẩm thứ 3 hết hàng mà đơn vẫn lưu một nửa.
     *
     * userId DO SERVER truyền vào (từ token, xem ShopOrderController), không phải
     * từ body: khách không còn tự khai được đơn này của tài khoản nào.
     */
    @Transactional
    public OrderResponse create(OrderCreateRequest request, Integer userId) {
        Map<Integer, Integer> quantities = mergeItems(request.getItems());

        List<OrderItem> items = new ArrayList<>(quantities.size());
        BigDecimal totalAmount = BigDecimal.ZERO;

        for (Map.Entry<Integer, Integer> entry : quantities.entrySet()) {
            Integer productId = entry.getKey();
            Integer quantity = entry.getValue();

            Product product = productRepository.findById(productId)
                    .orElseThrow(() -> new BusinessException(
                            "Sản phẩm #" + productId + " không còn tồn tại"));

            BigDecimal unitPrice = product.getPrice();
            int stockBefore = product.getQuantity() == null ? 0 : product.getQuantity();

            // "UPDATE ... WHERE quantity >= :qty" là phép trừ nguyên tử cấp DB:
            // 1 dòng bị ảnh hưởng = giữ được hàng, 0 dòng = không đủ hàng hoặc sản
            // phẩm vừa bị xóa giữa lúc đọc và lúc trừ. Không đọc rồi ghi lại, vì hai
            // request song song đều có thể cùng thấy "còn 1" và cùng bán một cái.
            if (productRepository.decreaseStock(productId, quantity) == 0) {
                throw new BusinessException("Sản phẩm \"" + product.getName()
                        + "\" chỉ còn " + stockBefore + " cái, đơn yêu cầu " + quantity);
            }

            items.add(OrderItem.builder()
                    .product(product)
                    .productName(product.getName())
                    .unitPrice(unitPrice)
                    .quantity(quantity)
                    .build());

            totalAmount = totalAmount.add(unitPrice.multiply(BigDecimal.valueOf(quantity)));
        }

        Order order = Order.builder()
                .orderCode(temporaryOrderCode())
                .userId(userId)
                .customerName(request.getCustomerName().trim())
                .phone(request.getPhone().trim())
                .address(request.getAddress().trim())
                .note(blankToNull(request.getNote()))
                .totalAmount(totalAmount)
                .status(OrderStatus.PENDING)
                .items(items)
                .build();

        Order saved = orderRepository.save(order);

        // Mã đơn lấy số thứ tự từ id nên chỉ biết được sau insert; column lại đang
        // NOT NULL + UNIQUE. Vì vậy lúc insert ghi một mã tạm random rồi sửa ngay
        // trong cùng transaction: không va chạm unique giữa hai request song song,
        // và khách hàng không bao giờ nhìn thấy mã tạm (chưa commit).
        saved.setOrderCode(formatOrderCode(saved.getId()));

        // Dòng lịch sử đầu tiên: đơn ĐÃ bước vào PENDING lúc này, do chính khách.
        // Ghi lại thành dữ liệu thay để UI suy ra từ created_at khi hiển thị:
        // created_at chỉ là thời điểm tạo row, hai thứ trùng nhau hôm nay không có
        // nghĩa là mãi mãi, mà lịch sử thì phải đọc được độc lập.
        OrderStatusHistory placed = statusHistoryRepository.save(OrderStatusHistory.builder()
                .orderId(saved.getId())
                .toStatus(OrderStatus.PENDING)
                .actorType(OrderStatusActorType.USER)
                .actorId(userId)
                .note("Đặt hàng")
                .build());

        return OrderResponse.from(saved, List.of(placed));
    }

    /**
     * "Đơn hàng của tôi": 50 đơn gần nhất của đúng tài khoản đang đăng nhập.
     *
     * Lọc bằng câu truy vấn theo user_id chứ không lấy hết về rồi lọc bằng code:
     * một tài khoản không được phép kéo toàn bộ lịch sử đặt hàng của hệ thống về
     * bộ nhớ tiến trình. Chặn ở 50 là đủ cho khách cuộn xem — chưa làm phân trang
     * vì chưa ai có nhiều đơn đến thế, và làm vội sẽ thành hai nguồn dữ liệu lệch
     * nhau giữa màn "Đơn của tôi" và trang chủ.
     *
     * Không dùng join fetch cho items: Spring Data không cho hạn chế số dòng (Top50)
     * đi cùng fetch join, nên mỗi đơn sẽ kéo thêm một lệnh đọc items. N+1 đó bị
     * chặn ở 50 lệnh và chạy trong một transaction đọc — chấp nhận được ở đây.
     */
    @Transactional(readOnly = true)
    public List<OrderResponse> myOrders(Integer userId) {
        List<Order> orders = orderRepository
                .findTop50ByUserIdOrderByCreatedAtDescIdDesc(userId);

        // Một câu truy vấn cho lịch sử của CẢ danh sách, không phải một câu cho từng
        // đơn: 50 đơn là 50 lệnh đọc chỉ để vẽ cái stepper.
        Map<Integer, List<OrderStatusHistory>> histories = statusHistoryRepository
                .mapByOrderId(orders.stream().map(Order::getId).toList());

        return orders.stream()
                .map(order -> OrderResponse.from(
                        order, histories.getOrDefault(order.getId(), List.of())))
                .toList();
    }

    /** Cộng gộp các dòng trùng sản phẩm trước khi kiểm tra tồn kho. */
    private Map<Integer, Integer> mergeItems(List<OrderItemRequest> requested) {
        Map<Integer, Integer> quantities = new LinkedHashMap<>();
        for (OrderItemRequest item : requested) {
            quantities.merge(item.getProductId(), item.getQuantity(), Integer::sum);
        }
        return quantities;
    }

    private String temporaryOrderCode() {
        return "TMP-" + UUID.randomUUID().toString().replace("-", "").substring(0, 20);
    }

    private String formatOrderCode(Integer id) {
        return String.format("DH-%06d", id);
    }

    private String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
