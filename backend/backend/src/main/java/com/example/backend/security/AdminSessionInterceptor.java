package com.example.backend.security;

import com.example.backend.exception.UnauthorizedException;
import com.example.backend.service.AdminSessionService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.cors.CorsUtils;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * Cổng chặn mọi đường dẫn thuộc về trang quản trị.
 *
 * Cùng kiểu với {@link UserSessionInterceptor} và phải giữ đúng hai điểm itu:
 *
 * 1) LA INTERCEPTOR, KHONG PHAI FILTER. Filter chay truoc bo may CORS cua Spring
 *    MVC, nen mot cou 401 tra ra tu filter khong kem header CORS; trinh duyet
 *    bien no thanh "khong goi duoc may chu" va nguoi dung doi ma khong hieu
 *    viec gi xay ra.
 *
 * 2) CHO PREFLIGHT OPTIONS DI QUA. Trinh duyet truoc khi goi that luon gui mot
 *    cou OPTIONS khong mang Authorization. Spring giu nguyen interceptor chain
 *    khi xu ly preflight, nen chan o day = moi chuc nang cua admin-frontend
 *    chet o trinh duyet trong khi curl van "OK". Do chinh la loi da gay ra banner
 *    "Khong goi duoc may chu" phia khach hang — xem UserSessionInterceptor#preHandle.
 *
 * Preflight khong doc/ghi gi, no chi hoi "duoc phep goi khong"; chinh
 * PreFlightHandler cua Spring tra loi, va no van phai di qua danh sach origin
 * trong CorsConfig.
 */
@Component
@RequiredArgsConstructor
public class AdminSessionInterceptor implements HandlerInterceptor {

    /** Key cua request attribute; controller admin doc dung key nay. */
    public static final String CURRENT_ADMIN_ID = "currentAdminId";

    /** Token goc — logout can nguyen ven de tinh lai SHA-256 va tim dong can xoa. */
    public static final String CURRENT_ADMIN_TOKEN = "currentAdminToken";

    private static final String BEARER_PREFIX = "Bearer ";

    private final AdminSessionService adminSessionService;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response,
                             Object handler) {
        if (CorsUtils.isPreFlightRequest(request)) {
            return true;
        }

        String header = request.getHeader("Authorization");

        if (header == null || !header.startsWith(BEARER_PREFIX)) {
            throw new UnauthorizedException("Chưa đăng nhập quản trị. Vui lòng đăng nhập lại.");
        }

        String token = header.substring(BEARER_PREFIX.length()).trim();
        Integer adminId = adminSessionService.resolveAdminId(token);
        if (adminId == null) {
            throw new UnauthorizedException(
                    "Phiên quản trị không còn hiệu lực. Vui lòng đăng nhập lại.");
        }

        request.setAttribute(CURRENT_ADMIN_ID, adminId);
        request.setAttribute(CURRENT_ADMIN_TOKEN, token);
        return true;
    }
}
