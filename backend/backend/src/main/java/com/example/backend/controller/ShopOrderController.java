package com.example.backend.controller;

import com.example.backend.dto.OrderCreateRequest;
import com.example.backend.dto.response.OrderResponse;
import com.example.backend.security.UserSessionInterceptor;
import com.example.backend.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestAttribute;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Đặt hàng phía khách.
 *
 * Ca hai duong deu dang nhap moi vao duoc: UserSessionInterceptor (dang ky trong
 * WebConfig) chan request thieu token hoac token sai truoc khi toi day. Yeu cau
 * "phai dang nhap moi duoc thanh toan" bay gio la luat cua server, khong phai
 * thuan giua hai ben trinh duyet nua — ghi chu cu o README client-frontend.
 */
@RestController
@RequestMapping("/api/orders")
@RequiredArgsConstructor
public class ShopOrderController {

    private final OrderService orderService;

    /**
     * Dat hang. userId lay tu token, khong tu body: don luu vao dung tai khoan
     * dang goi, va payload khong con cho nao de khach tu gan don cho nguoi khac.
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public OrderResponse create(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId,
            @Valid @RequestBody OrderCreateRequest request) {
        return orderService.create(request, userId);
    }

    /**
     * "Đơn hàng của tôi". Danh sach nay khong the loc bang cach gui
     * /api/orders?userId=X nhu mot so cua hang lam: du co kiem tra token thi van
     * thua mot loi the de doan id. Duong "mine" khong cho phep hoi ve nguoi khac
     * ngay từ hình dạng của nó.
     */
    @GetMapping("/mine")
    public List<OrderResponse> mine(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId) {
        return orderService.myOrders(userId);
    }
}
