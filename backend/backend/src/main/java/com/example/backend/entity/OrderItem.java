package com.example.backend.entity;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;

/**
 * Một dòng trong đơn hàng.
 *
 * productName và unitPrice là snapshot tại thời điểm đặt: đơn đã tạo rồi thì không
 * được đổi số tiền chỉ vì admin sửa giá, và đơn vẫn đọc được sau khi sản phẩm bị
 * xóa cứng (product_id khi đó đã thành null).
 */
@Entity
@Table(name = "order_item")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrderItem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /** Nullable trong DB, nên để ManyToOne tùy chọn chứ không phải bắt buộc. */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "product_id")
    private Product product;

    @Column(name = "product_name", nullable = false, length = 160)
    private String productName;

    @Column(name = "unit_price", nullable = false, precision = 12, scale = 2)
    private BigDecimal unitPrice;

    @Column(nullable = false)
    private Integer quantity;
}
