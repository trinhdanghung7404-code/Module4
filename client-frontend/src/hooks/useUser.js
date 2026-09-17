import { useCallback, useEffect, useState } from "react";
import { logoutUser } from "../api/userApi";
import {
  SESSION_EVENT,
  clearUserSession,
  getUserSession,
  saveUserSession,
} from "../utils/userSession.js";

/**
 * Phiên đăng nhập của khách, đọc được từ mọi component.
 *
 * Hai listener là cố ý và không thừa nhau: SESSION_EVENT bắn khi chính tab này
 * đăng nhập/đăng xuất, còn "storage" chỉ bắn ở TAB KHÁC. Thiếu một trong hai thì
 * header đứng ở trạng thái cũ cho tới lúc reload.
 *
 * Moi component goi hook nay mot lan: no khong giu state cua rieng no ma chi
 * phat lai nhung gi userSession.js doc duoc, nen nguon su that van la mot cho.
 */
export function useUser() {
  const [session, setSession] = useState(() => getUserSession());

  useEffect(() => {
    const sync = () => setSession(getUserSession());

    window.addEventListener(SESSION_EVENT, sync);
    window.addEventListener("storage", sync);

    return () => {
      window.removeEventListener(SESSION_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  /** Nhan dung { token, user } /users/login tra ve. */
  const login = useCallback((next) => {
    saveUserSession(next);
    setSession(next);
  }, []);

  /**
   * Đăng xuất thật: thu hồi token trên server rồi mới xoá trong máy.
   *
   * Thu tu la bat buoc — request() doc luc nay token tai luc phat dong goi, neu
   * xoa phien truoc thi lenh logout gui di khong kem Authorization va server thu
   * hoi khong dung cai nao.
   *
   * That bai khi thu hoi (mat mang, token da bi xoa o tab khac) thi van phai xoá
   * trong máy: khach bam "Dang xuat" la khong con dang nhap nua, va token ay cung
   * khong the dung lai duoc sau khi het han 401 lan dau.
   */
  const logout = useCallback(() => {
    logoutUser().catch(() => {});
    clearUserSession();
    setSession(null);
  }, []);

  return {
    user: session?.user ?? null,
    token: session?.token ?? null,
    /**
     * Chi de UI biet "co dang nhap khong" va de dem nhan tai khi doi tai khoan.
     * KHONG duoc dung de ghep vao duong dan API nua — danh tinh do token quyet.
     */
    userId: session?.user?.id ?? null,
    isLoggedIn: !!session,
    login,
    logout,
  };
}
