package com.example.backend.repository;

import com.example.backend.dto.response.ShopProductCardResponse;
import com.example.backend.entity.Product;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface ProductRepository
        extends JpaRepository<Product, Integer> {

    boolean existsBySku(String sku);

    boolean existsBySkuAndIdNot(String sku, Integer id);

    long countByCategoryId(Integer categoryId);

    /**
     * Trừ tồn kho một cách nguyên tử, trả về số dòng bị ảnh hưởng.
     *
     * Điều kiện "còn đủ hàng" nằm ngay trong WHERE của câu UPDATE chứ không kiểm
     * tra ở Java rồi mới ghi: đọc-rồi-ghi giữa hai request song song có thể khiến
     * cả hai cùng thấy "còn 1 cái" và cùng bán một sản phẩm. Ở đây PostgreSQL đảm
     * bảo chỉ một UPDATE được 1 dòng, câu còn lại được 0 dòng và service sẽ ném
     * BusinessException + rollback.
     *
     * Bulk update đi thẳng xuống DB nên bỏ qua persistence context; callers chỉ
     * được dùng nó khi không cần đọc lại số lượng vừa trừ trong cùng transaction.
     */
    @Modifying
    @Query(
            value = """
                    UPDATE Product p
                       SET p.quantity = p.quantity - :quantity
                     WHERE p.id = :productId
                       AND p.quantity >= :quantity
                    """
    )
    int decreaseStock(
            @Param("productId") Integer productId,
            @Param("quantity") Integer quantity
    );

    /**
     * Cộng trả tồn kho — chiều ngược lại của decreaseStock, dùng khi một đơn bị hủy.
     *
     * Cũng là bulk update vì đúng một lý do: phép tính phải nằm trong câu UPDATE
     * ("quantity = quantity + :quantity") để hai lần trả hàng của hai đơn khác nhau
     * trên cùng một sản phẩm không giẫm lên nhau. Đọc số lượng về Java cộng rồi ghi
     * lại sẽ đánh mất một trong hai lần cộng.
     *
     * COALESCE: cột quantity được phép null ở entity, và NULL + n vẫn là NULL, nên
     * phải quy về 0 trước khi cộng.
     *
     * Trả về số dòng bị ảnh hưởng. 0 nghĩa là sản phẩm đã bị xóa khỏi catalog từ lúc
     * đặt hàng — nhận trả lại hàng cho một mặt hàng không còn tồn tại là vô nghĩa,
     * caller phải BỎ QUA CÓ CHỦ ĐÍCH chứ không được coi là lỗi (xem AdminOrderService).
     */
    @Modifying
    @Query(
            value = """
                    UPDATE Product p
                       SET p.quantity = COALESCE(p.quantity, 0) + :quantity
                     WHERE p.id = :productId
                    """
    )
    int increaseStock(
            @Param("productId") Integer productId,
            @Param("quantity") Integer quantity
    );

    /**
     * Một trang card sản phẩm cho client, chỉ dùng đúng một câu SQL.
     *
     * Ảnh đại diện lấy bằng scalar subquery trong SELECT thay vì JOIN collection
     * ProductImage: join vào collection khi đang phân trang sẽ nhân bản số dòng
     * và buộc Hibernate phải cắt trang trong bộ nhớ, còn subquery thì giữ nguyên
     * một dòng cho một sản phẩm.
     *
     * MIN() chỉ để đảm bảo subquery trả về một giá trị duy nhất; ràng buộc
     * uq_product_image_type đã khiến mỗi sản phẩm tối đa một ảnh FRONT.
     *
     * categoryId = 0 là giá trị gửi, nghĩa là "không lọc theo danh mục".
     * searchPattern đã được service bọc %...% và hạ về chữ thường.
     */
    @Query(
            value = """
                    SELECT new com.example.backend.dto.response.ShopProductCardResponse(
                        p.id,
                        p.name,
                        p.price,
                        p.quantity,
                        c.name,
                        (SELECT MIN(img.imageUrl)
                           FROM ProductImage img
                          WHERE img.product = p
                            AND img.imageType = com.example.backend.enums.ProductImageType.FRONT)
                    )
                    FROM Product p
                    JOIN p.category c
                    WHERE (:categoryId = 0 OR p.category.id = :categoryId)
                      AND LOWER(p.name) LIKE :searchPattern
                    """,
            countQuery = """
                    SELECT COUNT(p)
                    FROM Product p
                    WHERE (:categoryId = 0 OR p.category.id = :categoryId)
                      AND LOWER(p.name) LIKE :searchPattern
                    """
    )
    Page<ShopProductCardResponse> findShopCards(
            @Param("categoryId") Integer categoryId,
            @Param("searchPattern") String searchPattern,
            Pageable pageable
    );
}
