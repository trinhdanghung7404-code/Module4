import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchOrderSummary } from "../api/orderApi";
import { getAdminSession } from "../utils/adminSession";
import "../styles/orders.css";

function AdminHomePage() {
  const [profile] = useState(() => getAdminSession());
  const [pendingCount, setPendingCount] = useState(null);
  const fullName = profile?.fullName || profile?.username || "Admin";

  /**
   * Con số duy nhất đáng hiện ở trang này là "bao nhiêu đơn đang chờ xác nhận":
   * đó là việc admin quay lại để làm.
   *
   * null khi server không trả lời -> ẨN hẳn cái badge, không hiện số 0. "0 đơn chờ"
   * là một khẳng định sai sự thật khi sự thật là "không lấy được dữ liệu".
   */
  useEffect(() => {
    fetchOrderSummary()
      .then((data) => setPendingCount(data?.byStatus?.PENDING ?? 0))
      .catch(() => setPendingCount(null));
  }, []);

  return (
    <div className="admin-container">
      <section className="welcome-section">
        <h1>Xin chào, {fullName}</h1>
        <p>Chọn chức năng bạn muốn quản lý.</p>
      </section>

      <section className="management-grid">
        <Link
          to="/admin/orders"
          className="management-card"
        >
          <div className="card-icon">▦</div>

          <div className="card-content">
            <h3>
              Đơn hàng
              {pendingCount !== null && pendingCount > 0 ? (
                <span className="card-count">{pendingCount} chờ xác nhận</span>
              ) : null}
            </h3>
            <p>
              Xác nhận đơn, bàn giao cho đơn vị vận chuyển và ghi nhận đã giao.
            </p>
          </div>

          <span className="card-arrow">→</span>
        </Link>

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

        <Link
          to="/admin/categories"
          className="management-card"
        >
          <div className="card-icon">▤</div>

          <div className="card-content">
            <h3>Quản lý danh mục</h3>
            <p>Thêm, đổi tên và xóa danh mục sản phẩm.</p>
          </div>

          <span className="card-arrow">→</span>
        </Link>
      </section>
    </div>
  );
}

export default AdminHomePage;