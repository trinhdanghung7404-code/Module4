CHÉP FILE VÀO PROJECT

controller/
- ProductController.java

dto/
- ProductCreateRequest.java

dto/response/
- CloudinaryUploadResponse.java
- ProductImageResponse.java
- ProductResponse.java

entity/
- Product.java
- ProductImage.java

enums/
- ProductImageType.java

repository/
- ProductImageRepository.java

service/
- ProductService.java
- CloudinaryService.java


TEST POSTMAN

POST http://localhost:8080/api/admin/products

Body -> form-data:
name          Text
sku           Text
price         Text
quantity      Text
categoryId    Text
description   Text
frontImage    File
backImage     File
leftImage     File
rightImage    File

Không tự thêm Content-Type trong Headers; Postman tự tạo multipart boundary.

Sau khi chép file:
mvnw.cmd clean compile

Lưu ý:
- Bảng product không còn cột image_url.
- Bảng product_image phải tồn tại.
- Cả 4 ảnh đang được cấu hình là bắt buộc.
