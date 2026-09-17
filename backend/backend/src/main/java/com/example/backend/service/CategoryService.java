package com.example.backend.service;

import com.example.backend.dto.CategoryCreateRequest;
import com.example.backend.dto.response.CategoryResponse;
import com.example.backend.entity.Category;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.CategoryRepository;
import com.example.backend.repository.ProductRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CategoryService {

    private final CategoryRepository categoryRepository;
    private final ProductRepository productRepository;

    public CategoryService(
            CategoryRepository categoryRepository,
            ProductRepository productRepository
    ) {
        this.categoryRepository = categoryRepository;
        this.productRepository = productRepository;
    }

    public CategoryResponse create(CategoryCreateRequest request) {
        String categoryName = request.getName().trim();

        if (categoryRepository.existsByName(categoryName)) {
            throw new BusinessException("Tên danh mục đã tồn tại");
        }

        Category category = new Category();
        category.setName(categoryName);

        return toResponse(categoryRepository.save(category));
    }

    public List<CategoryResponse> getAll() {
        return categoryRepository.findAll().stream()
                .map(this::toResponse)
                .toList();
    }

    public CategoryResponse update(
            Integer id,
            CategoryCreateRequest request
    ) {
        Category category = getOrThrow(id);
        String categoryName = request.getName().trim();

        if (categoryRepository.existsByNameAndIdNot(categoryName, id)) {
            throw new BusinessException("Tên danh mục đã tồn tại");
        }

        category.setName(categoryName);

        return toResponse(categoryRepository.save(category));
    }

    public void delete(Integer id) {
        Category category = getOrThrow(id);
        long productCount = productRepository.countByCategoryId(id);

        if (productCount > 0) {
            throw new BusinessException(
                    "Danh mục \"" + category.getName() + "\" đang có "
                            + productCount + " sản phẩm, không thể xóa"
            );
        }

        categoryRepository.delete(category);
    }

    private Category getOrThrow(Integer id) {
        return categoryRepository.findById(id)
                .orElseThrow(() -> new BusinessException(
                        "Không tìm thấy danh mục với id: " + id
                ));
    }

    private CategoryResponse toResponse(Category category) {
        CategoryResponse response = new CategoryResponse();

        response.setId(category.getId());
        response.setName(category.getName());
        response.setProductCount(
                productRepository.countByCategoryId(category.getId())
        );

        return response;
    }
}