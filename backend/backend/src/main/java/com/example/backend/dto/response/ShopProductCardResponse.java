package com.example.backend.dto.response;

import lombok.Getter;

import java.math.BigDecimal;

/**
 * Bản rút gọn của sản phẩm, dùng cho lưới card phía client.
 *
 * Chỉ giữ những thứ cần để vẽ một card: tên, giá, còn hàng, danh mục, ảnh đại diện.
 * Không kéo theo description (TEXT) và 3 ảnh còn lại của mỗi sản phẩm.
 *
 * Class này được Hibernate khởi tạo trực tiếp bằng JPQL constructor expression,
 * nên thứ tự và kiểu của tham số constructor phải khớp tuyệt đối với câu SELECT
 * trong ProductRepository#findShopCards.
 */
@Getter
public class ShopProductCardResponse {

    private final Integer id;
    private final String name;
    private final BigDecimal price;
    private final Integer quantity;
    private final String categoryName;
    private final String imageUrl;

    public ShopProductCardResponse(
            Integer id,
            String name,
            BigDecimal price,
            Integer quantity,
            String categoryName,
            String imageUrl
    ) {
        this.id = id;
        this.name = name;
        this.price = price;
        this.quantity = quantity;
        this.categoryName = categoryName;
        this.imageUrl = imageUrl;
    }
}
