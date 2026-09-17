package com.example.backend.controller;

import com.example.backend.dto.AdminOrderStatusRequest;
import com.example.backend.dto.response.AdminOrderPageResponse;
import com.example.backend.dto.response.OrderResponse;
import com.example.backend.dto.response.OrderSummaryResponse;
import com.example.backend.enums.OrderStatus;
import com.example.backend.exception.BusinessException;
import com.example.backend.security.AdminSessionInterceptor;
import com.example.backend.service.AdminOrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestAttribute;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Locale;

/**
 * Đơn hàng phía quản trị. Mọi đường ở đây đều đã qua AdminSessionInterceptor, nên
 * không có nhánh "không có token thì 401" nào trong controller — đúng như phía khách.
 *
 * Đường /summary đứng cạnh đường /{id}: Spring ưu tiên mẫu khớp nguyên văn hơn mẫu
 * có biến, nên "summary" không rơi vào {id}: Integer.
 */
@RestController
@RequestMapping("/api/admin/orders")
@RequiredArgsConstructor
public class AdminOrderController {

    private final AdminOrderService adminOrderService;

    /** status = bỏ trống hoặc "ALL" nghĩa là không lọc; page tính từ 0, size tối đa 50. */
    @GetMapping
    public AdminOrderPageResponse list(
            @RequestParam(required = false) String status,
            @RequestParam(required = false, defaultValue = "") String search,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return adminOrderService.list(parseStatus(status), search, page, size);
    }

    @GetMapping("/summary")
    public OrderSummaryResponse summary() {
        return adminOrderService.summary();
    }

    @GetMapping("/{id}")
    public OrderResponse detail(@PathVariable Integer id) {
        return adminOrderService.detail(id);
    }

    /**
     * Chuyển một đơn sang trạng thái kế tiếp.
     *
     * Dùng POST chứ không PUT/PATCH: đây là một hành động ("xác nhận đơn"), không
     * phải ghi đè tài nguyên. PUT /orders/{id} với nguyên body sẽ mời gọi client
     * gửi kèm cả totalAmount hoặc statusUpdatedAt — những thứ chỉ server được quyết.
     */
    @PostMapping("/{id}/status")
    public OrderResponse changeStatus(
            @PathVariable Integer id,
            @Valid @RequestBody AdminOrderStatusRequest request,
            @RequestAttribute(AdminSessionInterceptor.CURRENT_ADMIN_ID) Integer adminId) {
        return adminOrderService.changeStatus(id, request, adminId);
    }

    /**
     * status nhận dạng chuỗi rồi mới đổi ra enum: gửi "FOO" phải trả về đúng định
     * dạng lỗi {message} như mọi lỗi nghiệp vụ khác, thay vì cái body bind mặc định
     * của Spring mà client không biết moi chỗ nào ra thông báo.
     */
    private OrderStatus parseStatus(String raw) {
        if (raw == null || raw.isBlank() || "ALL".equalsIgnoreCase(raw.trim())) {
            return null;
        }

        try {
            return OrderStatus.valueOf(raw.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new BusinessException("Trạng thái không hợp lệ: " + raw);
        }
    }
}
