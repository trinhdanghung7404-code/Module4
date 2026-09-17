import { Link, NavLink } from "react-router-dom";
import AdminAccountMenu from "./AdminAccountMenu";

function AdminHeader() {
  return (
    <header className="admin-header">
      <Link to="/admin" className="admin-brand">
        Admin Management
      </Link>

      <nav className="admin-navigation">
        <NavLink
          to="/admin/products"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          Sản phẩm
        </NavLink>

        <NavLink
          to="/admin/categories"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          Danh mục
        </NavLink>

        <NavLink
          to="/admin/orders"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          Đơn hàng
        </NavLink>

        <AdminAccountMenu />
      </nav>
    </header>
  );
}

export default AdminHeader;