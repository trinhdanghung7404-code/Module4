package com.example.backend.controller;

import com.example.backend.dto.RecipientRequest;
import com.example.backend.dto.UserLoginRequest;
import com.example.backend.dto.UserRegisterRequest;
import com.example.backend.dto.response.RecipientResponse;
import com.example.backend.dto.response.UserAuthResponse;
import com.example.backend.dto.response.UserProfileResponse;
import com.example.backend.security.UserSessionInterceptor;
import com.example.backend.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Đăng ký / đăng nhập khách và sổ người nhận hàng.
 *
 * Moi duong dan thuoc ve khach deu di qua "/me": khong con /users/{id}/... nao
 * nua. Duong dan cu cho bat ky ai doan duoc id doc duoc so nguoi nhan (ten + SDT
 * + dia chi) cua nguoi khac. Bay gio id den tu token da duoc
 * {@link UserSessionInterceptor} kiem tra, nen "toi" la nguoi server biet, khong
 * phai nguoi tu xung — khong co noi de gan id cua nguoi khac vao.
 *
 * Controller nay khong co nhanh "userId == null thi 401" nao ca: interceptor da
 * chan truoc. Nhom duoc goi o day chac chan dang co phien hop le.
 */
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    /** Đăng ký xong có token ngay: khách đang đứng trước giỏ hàng, không bắt nhập lại. */
    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public UserAuthResponse register(@Valid @RequestBody UserRegisterRequest request) {
        return userService.register(request);
    }

    /** Hai duong cong khai duy nhat cua ca nhom — cung la ly do chung nam dau file. */
    @PostMapping("/login")
    public UserAuthResponse login(@Valid @RequestBody UserLoginRequest request) {
        return userService.login(request);
    }

    /** Thu hồi đúng phiên đang dùng; tab khác đăng nhập bằng token khác thì không bị ảnh hưởng. */
    @PostMapping("/logout")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void logout(@RequestAttribute(UserSessionInterceptor.CURRENT_TOKEN) String token) {
        userService.logout(token);
    }

    /** Client gọi đây để biết phiên còn sống hay đã hết hạn trước khi hiển thị tên khách. */
    @GetMapping("/me")
    public UserProfileResponse me(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId) {
        return userService.profile(userId);
    }

    @GetMapping("/me/recipients")
    public List<RecipientResponse> listRecipients(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId) {
        return userService.listRecipients(userId);
    }

    @PostMapping("/me/recipients")
    @ResponseStatus(HttpStatus.CREATED)
    public RecipientResponse addRecipient(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId,
            @Valid @RequestBody RecipientRequest request) {
        return userService.addRecipient(userId, request);
    }

    @PutMapping("/me/recipients/{recipientId}")
    public RecipientResponse updateRecipient(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId,
            @PathVariable Integer recipientId,
            @Valid @RequestBody RecipientRequest request) {
        return userService.updateRecipient(userId, recipientId, request);
    }

    /** Service đã kiểm tra người nhận thuộc về userId, nên không có đường xoá nhầm của người khác. */
    @DeleteMapping("/me/recipients/{recipientId}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteRecipient(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId,
            @PathVariable Integer recipientId) {
        userService.deleteRecipient(userId, recipientId);
    }

    /** Trả về toàn bộ danh sách mới vì đổi mặc định làm thay đổi cờ của nhiều dòng. */
    @PutMapping("/me/recipients/{recipientId}/default")
    public List<RecipientResponse> setDefaultRecipient(
            @RequestAttribute(UserSessionInterceptor.CURRENT_USER_ID) Integer userId,
            @PathVariable Integer recipientId) {
        return userService.setDefaultRecipient(userId, recipientId);
    }
}
