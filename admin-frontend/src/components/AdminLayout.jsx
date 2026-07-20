import { Outlet } from "react-router-dom";
import AdminHeader from "./AdminHeader";
import "../styles/admin.css";

function AdminLayout() {
  return (
    <div className="admin-layout">
      <AdminHeader />

      <main className="admin-main">
        <Outlet />
      </main>
    </div>
  );
}

export default AdminLayout;