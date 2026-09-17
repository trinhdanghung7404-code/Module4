package com.example.backend.dto.response;

import com.example.backend.entity.Admin;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class AdminProfileResponse {

    private Integer id;
    private String username;
    private String fullName;
    private String email;
    private LocalDateTime createdAt;

    public static AdminProfileResponse from(Admin admin) {
        AdminProfileResponse response = new AdminProfileResponse();

        response.setId(admin.getId());
        response.setUsername(admin.getUsername());
        response.setFullName(admin.getFullName());
        response.setEmail(admin.getEmail());
        response.setCreatedAt(admin.getCreatedAt());

        return response;
    }
}
