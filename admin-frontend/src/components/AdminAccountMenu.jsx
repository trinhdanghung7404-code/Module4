import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { logoutAdmin } from "../api/adminApi";
import {
  clearAdminSession,
  getAdminSession,
  getInitials,
} from "../utils/adminSession";

function formatJoinDate(value) {
  if (!value) {
    return "Chưa xác định";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Chưa xác định";
  }

  return date.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function AdminAccountMenu() {
  const navigate = useNavigate();
  const [profile] = useState(() => getAdminSession());
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const handlePointerDown = (event) => {
      if (!wrapperRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const [loggingOut, setLoggingOut] = useState(false);

  /**
   * Báo server thu hồi token TRƯỚC rồi mới xóa localStorage. Làm ngược thứ tự thì
   * dòng trong admin_session vẫn nằm im tới hết hạn, tức là token lấy được từ một
   * trình duyệt cũ vẫn dùng được — đúng cái mà nút "Đăng xuất" hứa là không.
   *
   * Gọi không thành công (mất mạng) vẫn phải cho người dùng thoát khỏi màn hình.
   */
  const handleLogout = async () => {
    setLoggingOut(true);

    try {
      await logoutAdmin();
    } catch {
      // Server không thu hồi được thì vẫn xóa ở phía client; token sẽ tự chết khi hết hạn.
    } finally {
      clearAdminSession();
      navigate("/login", {
        replace: true,
        state: { message: "Đã đăng xuất khỏi trang quản trị." },
      });
    }
  };

  const fullName = profile?.fullName || "Quản trị viên";

  return (
    <div className="account-menu" ref={wrapperRef}>
      <button
        type="button"
        className="account-avatar"
        onClick={() => setOpen((previous) => !previous)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Tài khoản của ${fullName}`}
        title={fullName}
      >
        {getInitials(profile)}
      </button>

      {open && (
        <div className="account-dropdown" role="menu">
          <div className="account-summary">
            <span className="account-avatar account-avatar-large">
              {getInitials(profile)}
            </span>

            <div className="account-summary-text">
              <span className="account-name">{fullName}</span>
              <span className="account-email">
                {profile?.email || "Chưa có email"}
              </span>
            </div>
          </div>

          <dl className="account-details">
            <div className="account-detail">
              <dt>Username</dt>
              <dd>{profile?.username || "—"}</dd>
            </div>

            <div className="account-detail">
              <dt>Tham gia từ</dt>
              <dd>{formatJoinDate(profile?.createdAt)}</dd>
            </div>
          </dl>

          <button
            type="button"
            className="account-logout"
            role="menuitem"
            disabled={loggingOut}
            onClick={handleLogout}
          >
            {loggingOut ? "Đang đăng xuất..." : "Đăng xuất"}
          </button>
        </div>
      )}
    </div>
  );
}

export default AdminAccountMenu;
