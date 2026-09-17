package com.example.backend.config;

import com.example.backend.security.AdminSessionInterceptor;
import com.example.backend.security.UserSessionInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Đăng ký interceptor xác thực — hai lớp riêng cho hai loại người dùng.
 *
 * Khách (UserSessionInterceptor), viết theo hướng "đường nào thuộc về khách đã
 * đăng nhập":
 *   - /api/users/**   : hồ sơ và sổ người nhận của chính tài khoản đang gọi
 *   - /api/orders     : đặt hàng
 *   - /api/orders/mine: "Đơn hàng của tôi"
 *
 * Quản trị (AdminSessionInterceptor): toàn bộ /api/admin/**, gồm cả sản phẩm,
 * danh mục và đơn hàng. Trước vòng này trang quản trị chỉ kiểm tra localStorage
 * ở phía trình duyệt, server không chặn gì — ai cũng POST/PUT/DELETE được.
 *
 * Vì sao phải là hai interceptor mà không phải một: admin và khách là hai bảng
 * người dùng khác nhau (admin / user_account). Một interceptor dùng chung nghĩa là
 * token của khách cũng đăng được vào trang quản trị — đúng cái cảm giác an toàn giả
 * cần tránh. Hai lớp cũng không chồng nhau: client-frontend không gọi /api/admin/**
 * và ngược lại, nên không request nào phải mang hai loại token.
 *
 * Đường công khai còn lại: /api/products, /api/categories (trạng thái hiển thị của
 * cửa hàng, chặn là phá trang chủ) và /api/users/login|register, /api/admin/login.
 * Riêng /api/admin/register vẫn mở — xem ghi chú trong AdminController.
 *
 * Thêm endpoint mới: cứ nằm dưới /api/admin/** là tự động phải có token admin,
 * không phải sửa file này nữa.
 */
@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {

    private final UserSessionInterceptor userSessionInterceptor;
    private final AdminSessionInterceptor adminSessionInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(userSessionInterceptor)
                .addPathPatterns("/api/users/**", "/api/orders", "/api/orders/**")
                .excludePathPatterns("/api/users/login", "/api/users/register");

        registry.addInterceptor(adminSessionInterceptor)
                .addPathPatterns("/api/admin/**")
                .excludePathPatterns("/api/admin/login", "/api/admin/register");
    }
}
