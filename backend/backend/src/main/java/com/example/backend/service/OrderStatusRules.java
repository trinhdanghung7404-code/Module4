package com.example.backend.service;

import com.example.backend.enums.OrderStatus;
import com.example.backend.exception.BusinessException;

import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Máy trạng thái của đơn hàng: từ trạng thái này được bước sang trạng thái nào.
 *
 * Vì sao nằm ở server mà không phải do trang quản trị quyết: nếu chỉ cần một nút
 * "PATCH status" thì ai cũng gửi được {"status":"DELIVERED"} cho một đơn vừa đặt,
 * và lịch sử trở thành chuyện bịa. Muốn luồng khép kín thì luật phải nằm ở nơi
 * không ai qua được — cùng lý do giá được tính trong DB chứ không lấy từ request.
 *
 * Luật:
 *   PENDING   -> CONFIRMED | CANCELLED      chờ xác nhận
 *   CONFIRMED -> SHIPPING  | CANCELLED      đã xác nhận, chưa bàn giao
 *   SHIPPING  -> DELIVERED                  đã trao cho đơn vị vận chuyển
 *   DELIVERED -> (hết)                      bằng chứng giao hàng, không sửa được
 *   CANCELLED -> (hết)
 *
 * Hai quyết định đáng hỏi lại khi đọc:
 *
 * 1) Đang vận chuyển thì KHÔNG hủy được nữa. Muốn có nhánh "giao không thành công"
 *    thì phải thêm trạng thái riêng (RETURNED) chứ không được phép quay ngược
 *    SHIPPING -> CANCELLED, vì hàng đã rời kho và không ai xác nhận nó về lại chưa.
 *    Chưa làm: đơn giao hỏng hiện phải xử lý bằng cách... để nguyên SHIPPING.
 *
 * 2) DELIVERED là trạng thái cuối. Đã ghi "khách nhận rồi" mà còn sửa được thì cột
 *    status không còn là bằng chứng, chỉ là một ô text.
 */
public final class OrderStatusRules {

    /** Nhãn tiếng Việt dùng trong thông báo lỗi — một nguồn duy nhất, client cũng đọc qua API. */
    private static final Map<OrderStatus, String> LABELS = labelMap();

    private static final Map<OrderStatus, Set<OrderStatus>> NEXT = nextMap();

    private OrderStatusRules() {
    }

    /**
     * Trạng thái đích có được phép không? Không thì ném BusinessException kèm đúng
     * lý do, để người bấm nút biết mình định làm gì sai thay vì nhận "400".
     */
    public static void assertAllowed(OrderStatus from, OrderStatus to) {
        if (from == to) {
            throw new BusinessException(
                    "Đơn đang ở \"" + label(to) + "\", không cần chuyển lại.");
        }
        if (!NEXT.getOrDefault(from, Collections.emptySet()).contains(to)) {
            throw new BusinessException("Không thể chuyển từ \"" + label(from) + "\" sang \""
                    + label(to) + "\". Các bước tiếp theo hợp lệ: " + nextLabelList(from) + ".");
        }
    }

    public static boolean allows(OrderStatus from, OrderStatus to) {
        return NEXT.getOrDefault(from, Collections.emptySet()).contains(to);
    }

    /** Những gì được bấm tiếp từ một trạng thái — admin-frontend vẽ nút dựa trên đúng danh sách này. */
    public static Set<OrderStatus> nextStates(OrderStatus from) {
        return Collections.unmodifiableSet(NEXT.getOrDefault(from, Collections.emptySet()));
    }

    /**
     * Chỉ có một bước duy nhất phải cộng trả tồn kho: hủy đơn.
     *
     * Tồn kho đã trừ ngay lúc đặt (OrderService#create dùng decreaseStock), nên đơn
     * bị hủy mà không trả hàng thì số liệu âm thầm tụt vĩnh viễn — càng nhiều đơn
     * hủy, kho càng ít đi so với thực tế mà không có lỗi nào báo cả.
     */
    public static boolean returnsStock(OrderStatus to) {
        return to == OrderStatus.CANCELLED;
    }

    public static String label(OrderStatus status) {
        return LABELS.getOrDefault(status, status == null ? "(chưa xác định)" : status.name());
    }

    public static Map<OrderStatus, String> labels() {
        return Collections.unmodifiableMap(LABELS);
    }

    private static String nextLabelList(OrderStatus from) {
        Set<OrderStatus> next = nextStates(from);
        if (next.isEmpty()) {
            return "không còn (đơn đã chốt ở trạng thái cuối)";
        }
        return next.stream().map(OrderStatusRules::label).collect(Collectors.joining(", "));
    }

    private static Map<OrderStatus, Set<OrderStatus>> nextMap() {
        Map<OrderStatus, Set<OrderStatus>> map = new EnumMap<>(OrderStatus.class);
        map.put(OrderStatus.PENDING, EnumSet.of(OrderStatus.CONFIRMED, OrderStatus.CANCELLED));
        map.put(OrderStatus.CONFIRMED, EnumSet.of(OrderStatus.SHIPPING, OrderStatus.CANCELLED));
        map.put(OrderStatus.SHIPPING, EnumSet.of(OrderStatus.DELIVERED));
        map.put(OrderStatus.DELIVERED, EnumSet.noneOf(OrderStatus.class));
        map.put(OrderStatus.CANCELLED, EnumSet.noneOf(OrderStatus.class));
        return Collections.unmodifiableMap(map);
    }

    private static Map<OrderStatus, String> labelMap() {
        Map<OrderStatus, String> map = new LinkedHashMap<>();
        map.put(OrderStatus.PENDING, "Chờ xác nhận");
        map.put(OrderStatus.CONFIRMED, "Chờ lấy hàng");
        map.put(OrderStatus.SHIPPING, "Đang vận chuyển");
        map.put(OrderStatus.DELIVERED, "Giao hàng thành công");
        map.put(OrderStatus.CANCELLED, "Đã hủy");
        return Collections.unmodifiableMap(map);
    }
}
