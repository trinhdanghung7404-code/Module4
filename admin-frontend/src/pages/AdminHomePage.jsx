import { Link } from "react-router-dom";

function AdminHomePage() {
  return (
    <div className="admin-container">
      <section className="welcome-section">
        <h1>Xin chào, Admin</h1>
        <p>Chọn chức năng bạn muốn quản lý.</p>
      </section>

      <section className="management-grid">
        <Link
          to="/admin/products"
          className="management-card"
        >
          <div className="card-icon">☰</div>

          <div className="card-content">
            <h3>Quản lý sản phẩm</h3>
            <p>
              Xem danh sách, thêm mới và quản lý sản phẩm.
            </p>
          </div>

          <span className="card-arrow">→</span>
        </Link>
      </section>
    </div>
  );
}

export default AdminHomePage;