import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useUser } from "../hooks/useUser";
import { formatDate } from "../utils/format";
import { getInitials } from "../utils/userSession";

/**
 * Khối tài khoản của khách trên header.
 *
 * Trước đây header trải hết "avatar + tên + Đăng xuất" ra một hàng: chiếm chỗ trên
 * màn hẹp, tên dài bị cắt "B22DCVT230_Trịnh …", và khách nhìn vào vẫn không biết
 * tài khoản đang dùng là email nào. Gom lại thành một dropdown — nút bấm chỉ còn
 * avatar + tên, bên trong là thông tin tài khoản, các đường dẫn và nút Đăng xuất.
 *
 * Giỏ hàng KHÔNG nằm trong menu này. Nó đã là một nút riêng của header, ngay cạnh
 * khối tài khoản, và khách chưa đăng nhập cũng phải bấm được — lặp lại nó trong menu
 * chỉ thêm một chỗ phải nhớ cập nhật mà chẳng thêm được đường đi nào.
 *
 * Component này KHÔNG tự xoá phiên: logout() của useUser mới là nơi quyết định thứ
 * tự (thu hồi token trên server trước, xoá localStorage sau). Ở đây chỉ gọi nó rồi
 * lo phần điều hướng.
 */
export default function ShopAccountMenu() {
  const { user, logout } = useUser();
  const location = useLocation();
  const navigate = useNavigate();

  // Menu ghi lại "mở ở đường dẫn nào", còn mở thật hay không là giá trị suy ra:
  // đổi trang làm pathname khác dấu đã ghi nên menu tự đóng, khỏi cần effect gọi
  // setState (react-hooks/set-state-in-effect cấm đúng kiểu đó).
  const [openForPath, setOpenForPath] = useState(null);
  const open = openForPath !== null && openForPath === location.pathname;
  const wrapperRef = useRef(null);

  function toggleMenu() {
    setOpenForPath((previous) =>
      previous === location.pathname ? null : location.pathname
    );
  }

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handlePointerDown(event) {
      if (!wrapperRef.current?.contains(event.target)) {
        setOpenForPath(null);
      }
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setOpenForPath(null);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  function closeMenu() {
    setOpenForPath(null);
  }

  /**
   * Đăng xuất ngay trên trang cần tài khoản: đưa khách về trang chủ để khỏi rơi
   * vào màn hình trống sau khi useUser xoá phiên.
   */
  function handleLogout() {
    closeMenu();
    logout();
    if (["/checkout", "/account"].includes(location.pathname)) navigate("/");
  }

  if (!user) {
    return (
      <div className="shop-account">
        <Link to="/login" className="shop-account__link">
          Đăng nhập
        </Link>
        <Link
          to="/register"
          className="shop-account__link shop-account__link--ghost"
        >
          Đăng ký
        </Link>
      </div>
    );
  }

  const fullName = user.fullName || user.username;

  return (
    <div className="shop-account" ref={wrapperRef}>
      <button
        type="button"
        className={`shop-account__trigger${open ? " is-open" : ""}`}
        onClick={toggleMenu}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Tài khoản của ${fullName}`}
        title={user.email || fullName}
      >
        <span className="shop-account__avatar" aria-hidden="true">
          {getInitials(user)}
        </span>
        <span className="shop-account__name">{fullName}</span>
        <span className="shop-account__caret" aria-hidden="true">
          ▾
        </span>
      </button>

      {open ? (
        <div className="shop-account__panel" role="menu">
          <div className="shop-account__summary">
            <span
              className="shop-account__avatar shop-account__avatar--lg"
              aria-hidden="true"
            >
              {getInitials(user)}
            </span>
            <span className="shop-account__who">
              <span className="shop-account__who-name">{fullName}</span>
              <span className="shop-account__who-email">
                {user.email || "Chưa có email"}
              </span>
            </span>
          </div>

          <dl className="shop-account__details">
            <div className="shop-account__detail">
              <dt>Username</dt>
              <dd title={user.username}>{user.username}</dd>
            </div>
            <div className="shop-account__detail">
              <dt>Tham gia từ</dt>
              <dd>{formatDate(user.createdAt)}</dd>
            </div>
          </dl>

          {/* Bấm link cũng phải đóng menu, không đợi suy ra từ pathname: đang ở
              /account mà bấm "Tài khoản của tôi" thì đường dẫn giữ nguyên và menu
              sẽ treo lại nếu không đóng tay ở đây. */}
          <Link
            to="/account"
            className="shop-account__item"
            role="menuitem"
            onClick={closeMenu}
          >
            Tài khoản của tôi
          </Link>

          {/* "Đơn hàng của tôi" là một mục riêng của menu, không phải khối nằm chờ ở
              cuối trang tài khoản: bấm là vào thẳng ?tab=orders, khỏi cuộn qua sổ người
              nhận và form thêm người nhận. Link này cũng gửi cho khách tự mở được. */}
          <Link
            to="/account?tab=orders"
            className="shop-account__item"
            role="menuitem"
            onClick={closeMenu}
          >
            Đơn hàng của tôi
          </Link>

          <div className="shop-account__divider" />

          <button
            type="button"
            className="shop-account__logout"
            role="menuitem"
            onClick={handleLogout}
          >
            Đăng xuất
          </button>
        </div>
      ) : null}
    </div>
  );
}