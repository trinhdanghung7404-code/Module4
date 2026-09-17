package com.example.backend.service;

import com.example.backend.dto.AdminLoginRequest;
import com.example.backend.dto.AdminRegisterRequest;
import com.example.backend.dto.response.AdminAuthResponse;
import com.example.backend.dto.response.AdminProfileResponse;
import com.example.backend.entity.Admin;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.AdminRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AdminService {

    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;
    private final AdminSessionService adminSessionService;

    /**
     * Một hash bcrypt thật, không ai biết nguyên văn, chỉ để nhánh "không có tài
     * khoản" cũng phải chạy đúng một lần so sánh như nhánh kia. Nếu bỏ hẳn bước
     * so sánh khi tài khoản lạ thì phản hồi nhanh hơn hẳn, và kẻ gọi API đo thời
     * gian là suy ra được email nào đang có trong bảng admin.
     */
    private final String timingEqualizerHash;

    public AdminService(AdminRepository adminRepository,
                        PasswordEncoder passwordEncoder,
                        AdminSessionService adminSessionService) {
        this.adminRepository = adminRepository;
        this.passwordEncoder = passwordEncoder;
        this.adminSessionService = adminSessionService;
        this.timingEqualizerHash = passwordEncoder.encode("timing-equalizer-not-a-password");
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

        // Mật khẩu chỉ đi vào DB dưới dạng bcrypt; plain text không nằm lại ở đâu.
        admin.setPassword(passwordEncoder.encode(request.getPassword()));

        adminRepository.save(admin);
    }

    /**
     * Đăng nhập và mở một phiên mới.
     *
     * Hai thay đổi so với bản cũ:
     * - trả về token, vì từ vòng này mọi call /api/admin/** đều phải chứng minh
     *   được phiên hợp lệ (AdminSessionInterceptor). Trước đó "đăng nhập admin"
     *   chỉ là chuyện giữa trình duyệt và localStorage của nó;
     * - "Tài khoản không tồn tại" tách riêng một câu như cũ là tự vạch áo cho người
     *   ngoài xem: không cần thử mật khẩu, đọc thông báo là biết email nào có trong
     *   bảng admin. Nay một thông báo duy nhất cho cả hai trường hợp.
     */
    @Transactional
    public AdminAuthResponse login(AdminLoginRequest request) {
        String account = request.getAccount() == null ? "" : request.getAccount().trim();

        Admin admin = adminRepository.findByUsernameOrEmail(account, account).orElse(null);

        String storedHash = admin == null ? timingEqualizerHash : admin.getPassword();
        if (admin == null || !passwordEncoder.matches(request.getPassword(), storedHash)) {
            throw new BusinessException("Tài khoản hoặc mật khẩu không đúng");
        }

        String token = adminSessionService.issue(admin.getId());

        return AdminAuthResponse.builder()
                .token(token)
                .admin(AdminProfileResponse.from(admin))
                .build();
    }

    /** Cho admin-frontend kiểm tra phiên còn sống hay đã hết hạn trước khi hiển thị tên. */
    @Transactional(readOnly = true)
    public AdminProfileResponse profile(Integer adminId) {
        return AdminProfileResponse.from(adminRepository.findById(adminId)
                .orElseThrow(() -> new BusinessException("Tài khoản quản trị không tồn tại")));
    }

    /** Thu hồi đúng phiên đang gọi; tab khác của cùng admin vẫn còn hiệu lực tới hạn của nó. */
    public void logout(String token) {
        adminSessionService.revoke(token);
    }
}