package com.example.backend.service;

import com.example.backend.dto.ProductCreateRequest;
import com.example.backend.dto.response.CloudinaryUploadResponse;
import com.example.backend.dto.response.ProductImageResponse;
import com.example.backend.dto.response.ProductResponse;
import com.example.backend.entity.Category;
import com.example.backend.entity.Product;
import com.example.backend.entity.ProductImage;
import com.example.backend.enums.ProductImageType;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.CategoryRepository;
import com.example.backend.repository.ProductRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

@Service
public class ProductService {

    private final ProductRepository productRepository;
    private final CategoryRepository categoryRepository;
    private final CloudinaryService cloudinaryService;

    public ProductService(
            ProductRepository productRepository,
            CategoryRepository categoryRepository,
            CloudinaryService cloudinaryService
    ) {
        this.productRepository = productRepository;
        this.categoryRepository = categoryRepository;
        this.cloudinaryService = cloudinaryService;
    }

    @Transactional
    public ProductResponse create(ProductCreateRequest request) {
        String sku = request.getSku().trim();

        if (productRepository.existsBySku(sku)) {
            throw new BusinessException("SKU đã tồn tại");
        }

        Category category = categoryRepository
                .findById(request.getCategoryId())
                .orElseThrow(() ->
                        new BusinessException("Danh mục không tồn tại")
                );

        validateRequiredImages(request);

        List<String> uploadedPublicIds = new ArrayList<>();

        try {
            Product product = new Product();
            product.setName(request.getName().trim());
            product.setSku(sku);
            product.setPrice(request.getPrice());
            product.setQuantity(request.getQuantity());
            product.setCategory(category);
            product.setDescription(trimToNull(request.getDescription()));

            uploadAndAttach(
                    product,
                    request.getFrontImage(),
                    ProductImageType.FRONT,
                    uploadedPublicIds
            );
            uploadAndAttach(
                    product,
                    request.getBackImage(),
                    ProductImageType.BACK,
                    uploadedPublicIds
            );
            uploadAndAttach(
                    product,
                    request.getLeftImage(),
                    ProductImageType.LEFT,
                    uploadedPublicIds
            );
            uploadAndAttach(
                    product,
                    request.getRightImage(),
                    ProductImageType.RIGHT,
                    uploadedPublicIds
            );

            Product savedProduct = productRepository.save(product);
            return toResponse(savedProduct);

        } catch (Exception exception) {
            uploadedPublicIds.forEach(cloudinaryService::deleteQuietly);

            if (exception instanceof BusinessException businessException) {
                throw businessException;
            }

            throw new BusinessException("Không thể lưu sản phẩm");
        }
    }

    @Transactional(readOnly = true)
    public List<ProductResponse> getAll() {
        return productRepository.findAll()
                .stream()
                .map(this::toResponse)
                .toList();
    }

    private void validateRequiredImages(ProductCreateRequest request) {
        validateImagePresent(request.getFrontImage(), "Ảnh trước");
        validateImagePresent(request.getBackImage(), "Ảnh sau");
        validateImagePresent(request.getLeftImage(), "Ảnh trái");
        validateImagePresent(request.getRightImage(), "Ảnh phải");
    }

    private void validateImagePresent(MultipartFile file, String label) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(label + " không được để trống");
        }
    }

    private void uploadAndAttach(
            Product product,
            MultipartFile file,
            ProductImageType imageType,
            List<String> uploadedPublicIds
    ) {
        CloudinaryUploadResponse uploaded =
                cloudinaryService.upload(file);

        uploadedPublicIds.add(uploaded.getPublicId());

        ProductImage image = new ProductImage();
        image.setImageType(imageType);
        image.setImageUrl(uploaded.getImageUrl());
        image.setImagePublicId(uploaded.getPublicId());

        product.addImage(image);
    }

    private ProductResponse toResponse(Product product) {
        ProductResponse response = new ProductResponse();

        response.setId(product.getId());
        response.setName(product.getName());
        response.setSku(product.getSku());
        response.setPrice(product.getPrice());
        response.setQuantity(product.getQuantity());
        response.setDescription(product.getDescription());
        response.setCreatedAt(product.getCreatedAt());
        response.setCategoryId(product.getCategory().getId());
        response.setCategoryName(product.getCategory().getName());

        response.setImages(
                product.getImages()
                        .stream()
                        .map(this::toImageResponse)
                        .toList()
        );

        return response;
    }

    private ProductImageResponse toImageResponse(
            ProductImage productImage
    ) {
        ProductImageResponse response =
                new ProductImageResponse();

        response.setId(productImage.getId());
        response.setImageType(productImage.getImageType());
        response.setImageUrl(productImage.getImageUrl());

        return response;
    }

    private String trimToNull(String value) {
        if (value == null) {
            return null;
        }

        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }
}
