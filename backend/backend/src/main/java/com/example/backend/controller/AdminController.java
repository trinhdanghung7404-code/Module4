package com.example.backend.controller;

import com.example.backend.dto.AdminLoginRequest;
import com.example.backend.dto.AdminRegisterRequest;
import com.example.backend.service.AdminService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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

    @PostMapping("/login")
    public ResponseEntity<String> login(
            @Valid @RequestBody AdminLoginRequest request
    ) {
        adminService.login(request);
        return ResponseEntity.ok("Đăng nhập thành công");
    }
}