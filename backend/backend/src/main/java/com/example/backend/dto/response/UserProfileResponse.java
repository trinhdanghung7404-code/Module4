package com.example.backend.dto.response;

import com.example.backend.entity.User;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * Những gì client được biết về một tài khoản. Không có mật khẩu, và cũng không
 * có số điện thoại: SĐT là thuộc tính của người nhận, không của tài khoản.
 */
@Getter
@Setter
public class UserProfileResponse {

    private Integer id;
    private String username;
    private String fullName;
    private String email;
    private LocalDateTime createdAt;

    public static UserProfileResponse from(User user) {
        UserProfileResponse response = new UserProfileResponse();

        response.setId(user.getId());
        response.setUsername(user.getUsername());
        response.setFullName(user.getFullName());
        response.setEmail(user.getEmail());
        response.setCreatedAt(user.getCreatedAt());

        return response;
    }
}
