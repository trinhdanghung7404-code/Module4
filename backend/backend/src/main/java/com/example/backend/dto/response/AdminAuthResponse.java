package com.example.backend.dto.response;

import lombok.Builder;
import lombok.Getter;

/**
 * Một lần đăng nhập quản trị: token + hồ sơ.
 *
 * Cùng kiểu UserAuthResponse. Token là giá trị duy nhất không lấy lại được sau khi
 * trả về — DB chỉ giữ SHA-256 của nó — nên admin-frontend phải cất ngay từ lúc
 * này, mất nó là phải đăng nhập lại.
 */
@Getter
@Builder
public class AdminAuthResponse {

    private final String token;
    private final AdminProfileResponse admin;
}
