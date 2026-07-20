import { Link } from "react-router-dom";

function AddProductPage() {
  return (
    <div className="admin-layout">
      <header className="admin-header">
        <div>
          <h2>Thêm sản phẩm</h2>
          <p>Quản lý sản phẩm</p>
        </div>

        <Link className="back-button" to="/admin">
          ← Quay lại
        </Link>
      </header>

      <main className="admin-content">
        <section className="product-form-card">
          <h1>Form thêm sản phẩm</h1>

          <p>
            Bước tiếp theo sẽ tạo form gồm tên sản phẩm, SKU,
            giá, danh mục, mô tả, URL ảnh và trạng thái.
          </p>
        </section>
      </main>
    </div>
  );
}

export default AddProductPage;