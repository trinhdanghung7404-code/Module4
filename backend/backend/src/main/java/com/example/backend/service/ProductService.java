package com.example.backend.service;

import com.example.backend.dto.ProductCreateRequest;
import com.example.backend.dto.response.CloudinaryUploadResponse;
import com.example.backend.dto.response.ProductImageResponse;
import com.example.backend.dto.response.ProductResponse;
import com.example.backend.dto.response.ShopProductCardResponse;
import com.example.backend.entity.Category;
import com.example.backend.entity.Product;
import com.example.backend.entity.ProductImage;
import com.example.backend.enums.ProductImageType;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.CategoryRepository;
import com.example.backend.repository.ProductRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

@Service
public class ProductService {

    /** categoryId = 0 trong câu query nghĩa là "không lọc theo danh mục". */
    private static final Integer NO_CATEGORY_FILTER = 0;

    /** Chặn client xin size=100000 để kéo cả bảng về một lần. */
    private static final int MAX_SHOP_PAGE_SIZE = 48;

    private static final int DEFAULT_SHOP_PAGE_SIZE = 12;

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

        Category category = getCategory(request.getCategoryId());
        validateRequiredImages(request);

        List<String> uploadedPublicIds = new ArrayList<>();

        try {
            Product product = new Product();
            applyProductData(product, request, sku, category);

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
            uploadedPublicIds.forEach(this::deleteCloudinaryQuietly);
            throwProductException(exception, "Không thể lưu sản phẩm");
            return null;
        }
    }

    @Transactional(readOnly = true)
    public List<ProductResponse> getAll() {
        return productRepository.findAll()
                .stream()
                .map(this::toResponse)
                .toList();
    }

    @Transactional(readOnly = true)
    public ProductResponse getById(Integer id) {
        return toResponse(getProduct(id));
    }

    @Transactional
    public ProductResponse update(
            Integer id,
            ProductCreateRequest request
    ) {
        Product product = getProduct(id);
        String sku = request.getSku().trim();

        if (productRepository.existsBySkuAndIdNot(sku, id)) {
            throw new BusinessException("SKU đã tồn tại");
        }

        Category category = getCategory(request.getCategoryId());
        List<String> uploadedPublicIds = new ArrayList<>();
        List<String> replacedPublicIds = new ArrayList<>();

        try {
            applyProductData(product, request, sku, category);

            replaceImageIfProvided(
                    product,
                    request.getFrontImage(),
                    ProductImageType.FRONT,
                    uploadedPublicIds,
                    replacedPublicIds
            );
            replaceImageIfProvided(
                    product,
                    request.getBackImage(),
                    ProductImageType.BACK,
                    uploadedPublicIds,
                    replacedPublicIds
            );
            replaceImageIfProvided(
                    product,
                    request.getLeftImage(),
                    ProductImageType.LEFT,
                    uploadedPublicIds,
                    replacedPublicIds
            );
            replaceImageIfProvided(
                    product,
                    request.getRightImage(),
                    ProductImageType.RIGHT,
                    uploadedPublicIds,
                    replacedPublicIds
            );

            Product savedProduct = productRepository.saveAndFlush(product);
            replacedPublicIds.forEach(this::deleteCloudinaryQuietly);

            return toResponse(savedProduct);

        } catch (Exception exception) {
            uploadedPublicIds.forEach(this::deleteCloudinaryQuietly);
            throwProductException(exception, "Không thể cập nhật sản phẩm");
            return null;
        }
    }

    @Transactional
    public void delete(Integer id) {
        Product product = getProduct(id);

        List<String> imagePublicIds = product.getImages()
                .stream()
                .map(ProductImage::getImagePublicId)
                .filter(publicId -> publicId != null && !publicId.isBlank())
                .toList();

        try {
            productRepository.delete(product);
            productRepository.flush();
            imagePublicIds.forEach(this::deleteCloudinaryQuietly);
        } catch (Exception exception) {
            throwProductException(exception, "Không thể xóa sản phẩm");
        }
    }

    /**
     * Một trang card sản phẩm cho lưới hiển thị phía client.
     *
     * sort được whitelist thay vì lấy thẳng thuộc tính từ URL: nếu cho client
     * truyền tự do thì mọi cột đều ORDER BY theo được, vừa dễ sai vừa lộ thêm
     * thông tin về cấu trúc bảng.
     */
    @Transactional(readOnly = true)
    public Page<ShopProductCardResponse> getShopCards(
            Integer categoryId,
            String search,
            String sort,
            int page,
            int size
    ) {
        Pageable pageable = PageRequest.of(
                Math.max(page, 0),
                clampShopPageSize(size),
                resolveShopSort(sort)
        );

        return productRepository.findShopCards(
                categoryId == null ? NO_CATEGORY_FILTER : categoryId,
                buildSearchPattern(search),
                pageable
        );
    }

    private int clampShopPageSize(int size) {
        if (size <= 0) {
            return DEFAULT_SHOP_PAGE_SIZE;
        }

        return Math.min(size, MAX_SHOP_PAGE_SIZE);
    }

    private Sort resolveShopSort(String sort) {
        if (sort == null) {
            return Sort.by(Sort.Order.desc("createdAt"));
        }

        return switch (sort) {
            case "price-asc" -> Sort.by(Sort.Order.asc("price"));
            case "price-desc" -> Sort.by(Sort.Order.desc("price"));
            case "name-asc" -> Sort.by(Sort.Order.asc("name"));
            case "name-desc" -> Sort.by(Sort.Order.desc("name"));
            case "quantity-asc" -> Sort.by(Sort.Order.asc("quantity"));
            case "quantity-desc" -> Sort.by(Sort.Order.desc("quantity"));
            case "oldest" -> Sort.by(Sort.Order.asc("createdAt"));
            default -> Sort.by(Sort.Order.desc("createdAt"));
        };
    }

    /**
     * Tìm kiếm bằng LIKE nên chỉ khớp được dấu bình thường: gõ "dien thoai"
     * sẽ không ra "Điện thoại". Muốn bỏ dấu thì cần hàm unaccent của Postgres
     * và cột chuẩn hoá, chưa đáng làm ở bước này.
     */
    private String buildSearchPattern(String search) {
        String keyword = search == null
                ? ""
                : search.trim().toLowerCase();

        return "%" + keyword + "%";
    }

    private Product getProduct(Integer id) {
        return productRepository.findById(id)
                .orElseThrow(() ->
                        new BusinessException("Sản phẩm không tồn tại")
                );
    }

    private Category getCategory(Integer categoryId) {
        return categoryRepository.findById(categoryId)
                .orElseThrow(() ->
                        new BusinessException("Danh mục không tồn tại")
                );
    }

    private void applyProductData(
            Product product,
            ProductCreateRequest request,
            String sku,
            Category category
    ) {
        product.setName(request.getName().trim());
        product.setSku(sku);
        product.setPrice(request.getPrice());
        product.setQuantity(request.getQuantity());
        product.setCategory(category);
        product.setDescription(trimToNull(request.getDescription()));
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

    private void replaceImageIfProvided(
            Product product,
            MultipartFile file,
            ProductImageType imageType,
            List<String> uploadedPublicIds,
            List<String> replacedPublicIds
    ) {
        if (file == null || file.isEmpty()) {
            return;
        }

        CloudinaryUploadResponse uploaded = cloudinaryService.upload(file);
        uploadedPublicIds.add(uploaded.getPublicId());

        ProductImage oldImage = product.getImages()
                .stream()
                .filter(image -> image.getImageType() == imageType)
                .findFirst()
                .orElse(null);

        if (oldImage != null) {
            String oldPublicId = oldImage.getImagePublicId();

            if (oldPublicId != null && !oldPublicId.isBlank()) {
                replacedPublicIds.add(oldPublicId);
            }

            oldImage.setImageUrl(uploaded.getImageUrl());
            oldImage.setImagePublicId(uploaded.getPublicId());
            return;
        }

        ProductImage newImage = new ProductImage();
        newImage.setImageType(imageType);
        newImage.setImageUrl(uploaded.getImageUrl());
        newImage.setImagePublicId(uploaded.getPublicId());
        product.addImage(newImage);
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

    private void deleteCloudinaryQuietly(String publicId) {
        try {
            cloudinaryService.delete(publicId);
        } catch (Exception ignored) {
            // Không làm hỏng thao tác DB khi Cloudinary xóa ảnh thất bại.
        }
    }

    private void throwProductException(
            Exception exception,
            String fallbackMessage
    ) {
        if (exception instanceof BusinessException businessException) {
            throw businessException;
        }

        throw new BusinessException(fallbackMessage);
    }

    private String trimToNull(String value) {
        if (value == null) {
            return null;
        }

        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }
}
