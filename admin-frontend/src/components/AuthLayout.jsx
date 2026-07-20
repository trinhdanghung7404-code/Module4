import { NavLink } from "react-router-dom";

function AuthLayout({ children }) {
  return (
    <main className="auth-page">
      <section className="brand-panel">
        <div className="brand-content">
          <div className="brand-logo">A</div>
          <h1>Admin Management</h1>
          <p>
            Hệ thống quản trị giúp bạn quản lý dữ liệu nhanh chóng,
            tập trung và an toàn.
          </p>

          <div className="feature">
            <span>✓</span>
            Quản lý tập trung
          </div>
          <div className="feature">
            <span>✓</span>
            Bảo mật mật khẩu
          </div>
          <div className="feature">
            <span>✓</span>
            Giao diện dễ sử dụng
          </div>
        </div>
      </section>

      <section className="form-panel">
        <div className="auth-card">
          <div className="mobile-logo">A</div>

          <nav className="tabs" aria-label="Điều hướng xác thực">
            <NavLink
              to="/login"
              className={({ isActive }) =>
                isActive ? "tab active" : "tab"
              }
            >
              Đăng nhập
            </NavLink>

            <NavLink
              to="/register"
              className={({ isActive }) =>
                isActive ? "tab active" : "tab"
              }
            >
              Đăng ký
            </NavLink>
          </nav>

          {children}
        </div>
      </section>
    </main>
  );
}

export default AuthLayout;
