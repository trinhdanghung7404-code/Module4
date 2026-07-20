package com.example.backend.service;

import com.example.backend.dto.AdminLoginRequest;
import com.example.backend.dto.AdminRegisterRequest;
import com.example.backend.entity.Admin;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.AdminRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AdminService {

    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;

    public AdminService(AdminRepository adminRepository, PasswordEncoder passwordEncoder) {
        this.adminRepository = adminRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public void register(AdminRegisterRequest request) {

        if (adminRepository.existsByUsername(request.getUsername())) {
            throw new BusinessException("Username đã tồn tại");
        }

        if (adminRepository.existsByEmail(request.getEmail())) {
            throw new BusinessException("Email đã tồn tại");
        }

        if (!request.getPassword().equals(request.getConfirmPassword())) {
            throw new BusinessException("Mật khẩu xác nhận không khớp");
        }

        Admin admin = new Admin();
        admin.setUsername(request.getUsername());
        admin.setFullName(request.getFullName());
        admin.setEmail(request.getEmail());

        // Tạm thời lưu password thường
        // Sau sẽ đổi sang BCrypt
        admin.setPassword(passwordEncoder.encode(request.getPassword()));

        adminRepository.save(admin);
    }

    public Admin login(AdminLoginRequest request){
        String account = request.getAccount();
        Admin admin = adminRepository.findByUsernameOrEmail(account, account)
                .orElseThrow(()->
                        new BusinessException("Tài khoản không tồn tại"));

        if (!passwordEncoder.matches(
                request.getPassword(),
                admin.getPassword()
        )) {
            throw new BusinessException(
                    "Tài khoản hoặc mật khẩu không chính xác"
            );
        }

        return admin;
    }
}