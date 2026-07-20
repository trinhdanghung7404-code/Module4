package com.example.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class CategoryCreateRequest {

    @NotBlank(message = "Tên danh mục không được để trống")
    @Size(max = 20, message = "Tên danh mục tối đa 20 ký tự")
    private String name;
}