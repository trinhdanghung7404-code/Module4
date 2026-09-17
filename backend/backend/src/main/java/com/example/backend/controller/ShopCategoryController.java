package com.example.backend.controller;

import com.example.backend.dto.response.CategoryResponse;
import com.example.backend.service.CategoryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Danh mục cho thanh lọc của client.
 *
 * categoryCount nằm sẵn trong CategoryResponse -> client hiển thị được
 * "12 sản phẩm" cạnh mỗi danh mục mà không cần gọi thêm API.
 */
@RestController
@RequestMapping("/api/categories")
public class ShopCategoryController {

    private final CategoryService categoryService;

    public ShopCategoryController(CategoryService categoryService) {
        this.categoryService = categoryService;
    }

    @GetMapping
    public ResponseEntity<List<CategoryResponse>> list() {
        return ResponseEntity.ok(categoryService.getAll());
    }
}
