package com.example.backend.dto.response;

import lombok.Getter;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Getter
@Setter
public class ProductResponse {

    private Integer id;
    private String name;
    private String sku;
    private BigDecimal price;
    private Integer quantity;

    private Integer categoryId;
    private String categoryName;

    private String description;
    private LocalDateTime createdAt;

    private List<ProductImageResponse> images;
}