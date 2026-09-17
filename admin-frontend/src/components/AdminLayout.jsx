import { useEffect } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import AdminHeader from "./AdminHeader";
import { UNAUTHORIZED_EVENT } from "../api/client";
import "../styles/admin.css";

function AdminLayout() {
  const navigate = useNavigate();

  /**
   * client.js phát UNAUTHORIZED_EVENT mỗi lần server trả 401. Nghe ở đây MỘT lần
   * cho cả cụm trang thay vì bắt từng trang tự xử lý: trang nào quên là người dùng
   * đứng giữa màn hình báo lỗi không rõ nguyên nhân, trong khi phiên đã chết.
   */
  useEffect(() => {
    const handleUnauthorized = () => {
      navigate("/login", {
        replace: true,
        state: {
          message: "Phiên quản trị không còn hiệu lực. Vui lòng đăng nhập lại.",
        },
      });
    };

    window.addEventListener(UNAUTHORIZED_EVENT, handleUnauthorized);

    return () => window.removeEventListener(UNAUTHORIZED_EVENT, handleUnauthorized);
  }, [navigate]);

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