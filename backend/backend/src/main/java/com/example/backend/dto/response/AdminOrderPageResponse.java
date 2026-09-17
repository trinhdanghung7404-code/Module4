package com.example.backend.dto.response;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * Một trang danh sách đơn cho trang quản trị.
 *
 * Không trả thẳng {@code org.springframework.data.domain.Page}: Spring Boot 3.5 đã
 * đánh dấu việc serialize PageImpl "as-is" là không được hỗ trợ ổn định giữa các
 * bản, còn client thì chỉ cần đúng bốn thông tin: danh sách, trang hiện tại, còn
 * bao nhiêu trang, hết chưa. Khai báo tường minh cũng khiến client-frontend và
 * admin-frontend không phải đoán shape của đối tượng giữa các lần nâng cấp.
 */
@Getter
@Builder
public class AdminOrderPageResponse {

    private final List<OrderResponse> content;
    private final int page;
    private final int size;
    private final long totalElements;
    private final int totalPages;
    private final boolean last;
}
