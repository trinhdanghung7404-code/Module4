package com.example.backend.dto.response;

import lombok.Builder;
import lombok.Getter;

/**
 * Những gì một lần đăng ký / đăng nhập trả về: token + hồ sơ.
 *
 * Token la gia tri duy nhat khong the lay lai duoc sau khi tra ve — DB chi giu
 * SHA-256 cua no. Client mat no thi phai dang nhap lai, va do la gia tri chap
 * nhan duoc cho mot luong ma khong phat sinh them bang ghi.
 *
 * `user` di kem de client khoi phai goi them GET /users/me ngay sau khi dang nhap
 * (mot lan tra loi cho hai viec), nhưng luu y: quyen luu cua no nam o token, khong
 * phai o cai id ma client tu ghi vao localStorage.
 */
@Getter
@Builder
public class UserAuthResponse {

    private final String token;
    private final UserProfileResponse user;
}
