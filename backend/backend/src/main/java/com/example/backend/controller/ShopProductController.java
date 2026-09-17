package com.example.backend.controller;

import com.example.backend.dto.response.ProductResponse;
import com.example.backend.dto.response.ShopProductCardResponse;
import com.example.backend.dto.response.ShopProductPageResponse;
import com.example.backend.service.ProductService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * API đọc sản phẩm cho phía client, nằm ngoài /api/admin/**.
 *
 * Tách đường dẫn ngay từ đầu để ngày bật Spring Security chỉ cần khai báo:
 * /api/products/** -> permitAll, /api/admin/** -> authenticated.
 * Nếu client đi nhờ path admin thì lúc đó hoặc client chết, hoặc phải mở khoá
 * path admin và mất luôn ý nghĩa của việc đăng nhập.
 */
@RestController
@RequestMapping("/api/products")
public class ShopProductController {

    private final ProductService productService;

    public ShopProductController(ProductService productService) {
        this.productService = productService;
    }

    @GetMapping
    public ResponseEntity<ShopProductPageResponse> list(
            @RequestParam(defaultValue = "0") Integer categoryId,
            @RequestParam(required = false) String search,
            @RequestParam(defaultValue = "newest") String sort,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "12") int size
    ) {
        return ResponseEntity.ok(
                ShopProductPageResponse.from(
                        productService.getShopCards(
                                categoryId,
                                search,
                                sort,
                                page,
                                size
                        )
                )
        );
    }

    /**
     * Trang chi tiết cần đủ mô tả và cả 4 ảnh FRONT/BACK/LEFT/RIGHT,
     * nên dùng lại ProductResponse như bên admin.
     */
    @GetMapping("/{id}")
    public ResponseEntity<ProductResponse> getById(
            @PathVariable Integer id
    ) {
        return ResponseEntity.ok(productService.getById(id));
    }
}
