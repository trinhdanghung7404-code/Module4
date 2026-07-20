package com.example.backend.dto.response;

import com.example.backend.enums.ProductImageType;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ProductImageResponse {

    private Integer id;
    private ProductImageType imageType;
    private String imageUrl;
}