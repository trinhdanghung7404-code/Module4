package com.example.backend.controller;

import com.example.backend.dto.CategoryCreateRequest;
import com.example.backend.entity.Category;
import com.example.backend.service.CategoryService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/admin/categories")
public class CategoryController {

    private final CategoryService categoryService;

    public CategoryController(CategoryService categoryService) {
        this.categoryService = categoryService;
    }

    @PostMapping
    public ResponseEntity<Category> create(
            @Valid @RequestBody CategoryCreateRequest request
    ) {
        Category category = categoryService.create(request);

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(category);
    }

    @GetMapping
    public ResponseEntity<List<Category>> getAll() {
        return ResponseEntity.ok(categoryService.getAll());
    }
}