import { Link, useNavigate } from "react-router-dom";

function AdminHeader() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("adminLoggedIn");
    navigate("/login", { replace: true });
  };

  return (
    <header className="admin-header">
      <Link to="/admin" className="admin-brand">
        Admin Management
      </Link>

      <nav className="admin-navigation">
        <Link to="/admin">Trang chủ</Link>
        <Link to="/admin/products">Sản phẩm</Link>

        <button type="button" onClick={handleLogout}>
          Đăng xuất
        </button>
      </nav>
    </header>
  );
}

export default AdminHeader;