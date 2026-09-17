package com.example.backend.dto.response;

import com.example.backend.entity.Order;
import com.example.backend.entity.OrderStatusHistory;
import com.example.backend.enums.OrderStatus;
import com.example.backend.enums.OrderStatusActorType;
import com.example.backend.service.OrderStatusRules;
import lombok.Builder;
import lombok.Getter;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Một đơn hàng, dạng trả ra client (khách lẫn trang quản trị dùng chung).
 *
 * Từ vòng luồng đơn hàng, DTO này TỰ sinh từ entity bằng {@link #from}: trước đó
 * OrderService có một hàm toResponse riêng, trang quản trị sẽ phải sao chép thêm
 * bản thứ hai, và hai bản map cùng một entity là hai bản sẽ lệch nhau chỗ nào đó
 * vào lúc không ai nhớ.
 */
@Getter
@Builder
public class OrderResponse {

    private final Integer id;
    private final String orderCode;
    private final String customerName;
    private final String phone;
    private final String address;
    private final String note;
    private final BigDecimal totalAmount;
    private final OrderStatus status;

    /** Nhãn tiếng Việt của trạng thái, do server quyết để hai client không tự dịch lệch nhau. */
    private final String statusLabel;

    /**
     * Trạng thái hợp lệ kế tiếp theo OrderStatusRules. Trang quản trị vẽ nút bấm từ
     * danh sách này thay vì tự lặp lại máy trạng thái trong JavaScript: hai nguồn
     * luật thì một trong hai nguồn sẽ sai trước.
     */
    private final List<OrderStatus> nextStatuses;

    private final LocalDateTime createdAt;
    private final LocalDateTime statusUpdatedAt;
    private final String shippingUnit;
    private final String trackingCode;

    private final List<Item> items;

    /** Lịch sử đổi trạng thái, cũ nhất trước. Rỗng với đơn chưa từng được chuyển. */
    private final List<Change> history;

    @Getter
    @Builder
    public static class Item {
        private final Integer productId;
        private final String productName;
        private final BigDecimal unitPrice;
        private final Integer quantity;
        private final BigDecimal lineTotal;
    }

    /** Một bước trong lịch sử: từ đâu sang đâu, lúc nào, ai làm. */
    @Getter
    @Builder
    public static class Change {
        private final OrderStatus fromStatus;
        private final OrderStatus toStatus;
        private final String toStatusLabel;
        private final OrderStatusActorType actorType;
        private final String actorLabel;
        private final String note;
        private final LocalDateTime createdAt;
    }

    public static OrderResponse from(Order order, List<OrderStatusHistory> history) {
        List<Item> items = order.getItems().stream()
                .map(item -> Item.builder()
                        .productId(item.getProduct() == null ? null : item.getProduct().getId())
                        .productName(item.getProductName())
                        .unitPrice(item.getUnitPrice())
                        .quantity(item.getQuantity())
                        .lineTotal(item.getUnitPrice()
                                .multiply(BigDecimal.valueOf(item.getQuantity())))
                        .build())
                .toList();

        List<Change> changes = history.stream()
                .map(entry -> Change.builder()
                        .fromStatus(entry.getFromStatus())
                        .toStatus(entry.getToStatus())
                        .toStatusLabel(OrderStatusRules.label(entry.getToStatus()))
                        .actorType(entry.getActorType())
                        .actorLabel(entry.getActorLabel())
                        .note(entry.getNote())
                        .createdAt(entry.getCreatedAt())
                        .build())
                .toList();

        return OrderResponse.builder()
                .id(order.getId())
                .orderCode(order.getOrderCode())
                .customerName(order.getCustomerName())
                .phone(order.getPhone())
                .address(order.getAddress())
                .note(order.getNote())
                .totalAmount(order.getTotalAmount())
                .status(order.getStatus())
                .statusLabel(OrderStatusRules.label(order.getStatus()))
                .nextStatuses(List.copyOf(OrderStatusRules.nextStates(order.getStatus())))
                .createdAt(order.getCreatedAt())
                .statusUpdatedAt(order.getStatusUpdatedAt())
                .shippingUnit(order.getShippingUnit())
                .trackingCode(order.getTrackingCode())
                .items(items)
                .history(changes)
                .build();
    }
}
