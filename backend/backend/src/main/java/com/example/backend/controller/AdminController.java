package com.example.backend.controller;

import com.example.backend.dto.AdminLoginRequest;
import com.example.backend.dto.AdminRegisterRequest;
import com.example.backend.dto.response.AdminAuthResponse;
import com.example.backend.dto.response.AdminProfileResponse;
import com.example.backend.security.AdminSessionInterceptor;
import com.example.backend.service.AdminService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * Đăng nhập / đăng ký tài khoản quản trị.
 *
 * Chỉ hai đường công khai: /login và /register — lúc chưa có token thì không thể đòi
 * token. Mọi đường khác dưới /api/admin/** (sản phẩm, danh mục, đơn hàng) đã bị
 * AdminSessionInterceptor chặn, khai báo trong WebConfig.
 *
 * /register vẫn công khai là một lỗ hổng CÓ CHỦ ĐÍCH, tạm giữ để không phá cách tạo
 * tài khoản test đang dùng: bất kỳ ai cũng dựng được một admin mới. Sửa = bỏ
 * "/api/admin/register" khỏi danh sách loại trừ ở WebConfig, nhưng khi đó phải có
 * màn "thêm quản trị viên" bên trong trang đã đăng nhập. Ghi rõ ở README admin-frontend.
 */
@RestController
@RequestMapping("/api/admin")
public class AdminController {

    private final AdminService adminService;

    public AdminController(AdminService adminService) {
        this.adminService = adminService;
    }

    @PostMapping("/register")
    public ResponseEntity<String> register(
            @Valid @RequestBody AdminRegisterRequest request) {

        adminService.register(request);

        return ResponseEntity.ok("Đăng ký thành công");
    }

    /** Đăng nhập trả token; từ đây mọi call admin phải kèm "Authorization: Bearer <token>". */
    @PostMapping("/login")
    public ResponseEntity<AdminAuthResponse> login(
            @Valid @RequestBody AdminLoginRequest request
    ) {
        return ResponseEntity.ok(adminService.login(request));
    }

    /** Client gọi đây để biết phiên còn sống hay đã hết hạn; 401 là phải đăng nhập lại. */
    @GetMapping("/me")
    public AdminProfileResponse me(
            @RequestAttribute(AdminSessionInterceptor.CURRENT_ADMIN_ID) Integer adminId) {
        return adminService.profile(adminId);
    }

    /** Thu hồi đúng phiên đang dùng, không đá các tab khác của cùng tài khoản. */
    @PostMapping("/logout")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void logout(
            @RequestAttribute(AdminSessionInterceptor.CURRENT_ADMIN_TOKEN) String token) {
        adminService.logout(token);
    }
}