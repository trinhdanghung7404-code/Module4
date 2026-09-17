package com.example.backend.dto.response;

import lombok.Getter;
import org.springframework.data.domain.Page;

import java.util.List;

/**
 * Vỏ phân trang ổn định cho client.
 *
 * Không trả thẳng Page<...> của Spring Data vì JSON của nó không được cam kết
 * ổn định giữa các phiên bản, và client chỉ cần đúng mấy con số này để vẽ nút
 * trang trước/sau.
 */
@Getter
public class ShopProductPageResponse {

    private final List<ShopProductCardResponse> content;
    private final int page;
    private final int size;
    private final long totalElements;
    private final int totalPages;
    private final boolean last;

    public ShopProductPageResponse(
            List<ShopProductCardResponse> content,
            int page,
            int size,
            long totalElements,
            int totalPages,
            boolean last
    ) {
        this.content = content;
        this.page = page;
        this.size = size;
        this.totalElements = totalElements;
        this.totalPages = totalPages;
        this.last = last;
    }

    public static ShopProductPageResponse from(
            Page<ShopProductCardResponse> products
    ) {
        return new ShopProductPageResponse(
                products.getContent(),
                products.getNumber(),
                products.getSize(),
                products.getTotalElements(),
                products.getTotalPages(),
                products.isLast()
        );
    }
}
