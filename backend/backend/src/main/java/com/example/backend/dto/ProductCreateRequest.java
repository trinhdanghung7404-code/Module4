package com.example.backend.dto;

import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

import java.math.BigDecimal;

@Getter
@Setter
public class ProductCreateRequest {

    @NotBlank(message = "Tên sản phẩm không được để trống")
    @Size(max = 150, message = "Tên sản phẩm tối đa 150 ký tự")
    private String name;

    @NotBlank(message = "SKU không được để trống")
    @Size(max = 50, message = "SKU tối đa 50 ký tự")
    private String sku;

    @NotNull(message = "Giá không được để trống")
    @DecimalMin(value = "0.01", message = "Giá phải lớn hơn 0")
    private BigDecimal price;

    @NotNull(message = "Số lượng không được để trống")
    @Min(value = 0, message = "Số lượng không được âm")
    private Integer quantity;

    @NotNull(message = "Danh mục không được để trống")
    private Integer categoryId;

    @Size(max = 5000, message = "Mô tả tối đa 5000 ký tự")
    private String description;

    private MultipartFile frontImage;
    private MultipartFile backImage;
    private MultipartFile leftImage;
    private MultipartFile rightImage;
}
