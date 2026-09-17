package com.example.backend.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class UserLoginRequest {

    /** Chấp nhận cả username lẫn email, giống form đăng nhập của admin. */
    @NotBlank(message = "Username hoặc email không được để trống")
    private String account;

    @NotBlank(message = "Mật khẩu không được để trống")
    private String password;
}
