package com.example.backend.security;

import com.example.backend.exception.UnauthorizedException;
import com.example.backend.service.SessionService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.cors.CorsUtils;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * Cổng chặn mọi đường dẫn cần đăng nhập của khách.
 *
 * Cách làm: đọc header "Authorization: Bearer &lt;token&gt;", đổi token ra id tài
 * khoản, và gắn id đó vào request attribute. Controller nhận id bằng
 * `@RequestAttribute` — không còn đường dẫn nào để client tự khai "tôi là user 5"
 * nữa, đó là toàn bộ lý do sinh ra file này.
 *
 * Không dùng filter: filter chạy TRƯỚC bộ máy CORS của Spring MVC, nên một cú 401
 * do filter trả ra sẽ không kèm header CORS và trình duyệt chỉ báo "không gọi được
 * máy chủ" thay vì cho biết phiên đã hết hạn. Interceptor nằm trong DispatcherServlet,
 * chạy sau khi header CORS đã được gắn.
 *
 * Cũng vì nằm trong DispatcherServlet nên interceptor thấy cả preflight OPTIONS —
 * xem ghi chú trong preHandle(). Đưa lớp chặn này sang filter thì bug đó quay lại,
 * và lần này còn khó thấy hơn vì filter không có cơ hội nào cho OPTIONS đi qua.
 */
@Component
@RequiredArgsConstructor
public class UserSessionInterceptor implements HandlerInterceptor {

    /** Key của request attribute; Controller đọc đúng key này, không tự đặt lại. */
    public static final String CURRENT_USER_ID = "currentUserId";

    /**
     * Token gốc của người đang gọi — logout cần đúng chuỗi nguyên vẹn để tính lại
     * SHA-256 và tìm ra dòng cần xoá.
     */
    public static final String CURRENT_TOKEN = "currentToken";

    private static final String BEARER_PREFIX = "Bearer ";

    private final SessionService sessionService;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response,
                             Object handler) {
        // Preflight của trình duyệt KHÔNG BAO GIỜ mang Authorization: nó chỉ liệt kê
        // header sẽ gửi trong Access-Control-Request-Headers. Spring giữ nguyên
        // interceptor chain khi xử lý preflight (AbstractHandlerMapping
        // #getCorsHandlerExecutionChain), nên chặn ở đây biến mọi endpoint đăng nhập
        // thành "lỗi CORS": browser trả về network error, client chỉ còn báo "không
        // gọi được máy chủ" trong khi /users/login (không bị chặn) vẫn chạy bình
        // thường. Cho preflight đi qua — nó không đọc/ghi gì cả, và chính
        // PreFlightHandler của Spring trả lời phần CORS.
        if (CorsUtils.isPreFlightRequest(request)) {
            return true;
        }

        String header = request.getHeader("Authorization");

        if (header == null || !header.startsWith(BEARER_PREFIX)) {
            throw new UnauthorizedException("Chưa đăng nhập. Vui lòng đăng nhập để tiếp tục.");
        }

        String token = header.substring(BEARER_PREFIX.length()).trim();
        Integer userId = sessionService.resolveUserId(token);
        if (userId == null) {
            throw new UnauthorizedException("Phiên đăng nhập không còn hiệu lực. Vui lòng đăng nhập lại.");
        }

        request.setAttribute(CURRENT_USER_ID, userId);
        request.setAttribute(CURRENT_TOKEN, token);
        return true;
    }
}
