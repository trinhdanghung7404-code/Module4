package com.example.backend.service;

import com.example.backend.dto.CategoryCreateRequest;
import com.example.backend.entity.Category;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.CategoryRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CategoryService {

    private final CategoryRepository categoryRepository;

    public CategoryService(CategoryRepository categoryRepository) {
        this.categoryRepository = categoryRepository;
    }

    public Category create(CategoryCreateRequest request) {
        String categoryName = request.getName().trim();

        if (categoryRepository.existsByName(categoryName)) {
            throw new BusinessException("Tên danh mục đã tồn tại");
        }

        Category category = new Category();
        category.setName(categoryName);

        return categoryRepository.save(category);
    }

    public List<Category> getAll() {
        return categoryRepository.findAll();
    }
}