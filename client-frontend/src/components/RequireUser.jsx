import { Navigate, Outlet, useLocation } from "react-router-dom";
import { isLoggedIn } from "../utils/userSession";

/**
 * Chặn đường thanh toán khi chưa đăng nhập, nhớ đúng trang khách đang định vào
 * để đăng nhập xong quay lại đó thay vì bị ném về trang chủ mất giỏ hàng.
 *
 * Chan nay chi de khach khoi nhap thong tin roi moi bi bao phai dang nhap. Rang
 * buoc that nam o server: POST /api/orders doi header Authorization hop le
 * (UserSessionInterceptor), nen goi thang vao API cung khong tao duoc don.
 */
function RequireUser() {
  const location = useLocation();

  return isLoggedIn() ? (
    <Outlet />
  ) : (
    <Navigate
      to="/login"
      replace
      state={{
        from: `${location.pathname}${location.search}`,
        message: "Đăng nhập để thanh toán và không phải nhập lại thông tin mỗi lần.",
      }}
    />
  );
}

export default RequireUser;
