package com.example.backend.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AdminLoginRequest {

    @NotBlank(message = "Username hoặc email không được để trống")
    private String account;

    @NotBlank(message = "Mật khẩu không được để trống")
    private String password;
}