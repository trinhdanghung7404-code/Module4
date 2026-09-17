package com.example.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    private final String[] allowedOrigins;

    /**
     * Origin đọc từ cấu hình thay vì hardcode trong mã.
     *
     * Vite mặc định chiếm cổng 5173. Chạy admin và client cùng lúc thì app thứ
     * hai tự nhảy sang 5174, và nếu 5174 không có trong danh sách này thì
     * browser chặn toàn bộ response dù backend đã trả về 200.
     */
    public CorsConfig(
            @Value("${app.cors.allowed-origins:http://localhost:5173,http://localhost:5174}")
            String[] allowedOrigins
    ) {
        this.allowedOrigins = allowedOrigins;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins(allowedOrigins)
                .allowedMethods(
                        "GET",
                        "POST",
                        "PUT",
                        "DELETE",
                        "OPTIONS"
                )
                .allowedHeaders("*");
    }
}