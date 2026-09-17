import { useCallback, useEffect, useState } from "react";
import { fetchRecipients } from "../api/userApi";
import { useUser } from "./useUser";

/**
 * Danh sách người nhận đã lưu của CHÍNH tài khoản đang đăng nhập.
 *
 * Khong co tham so userId: goi API nam duoc danh tinh tu token, nen truyen id
 * vao day chi them mot cach de goi sai nguoi. `userId` cua useUser o day chi
 * dung lam nhan "dang tai cho ai" de dung danh sach cua tai khoan cu khi khach
 * doan xuong va mot tai khoan khac trong cung mot lan mo trang.
 */
export function useRecipients() {
  const { userId } = useUser();
  const [items, setItems] = useState([]);
  const [loadedFor, setLoadedFor] = useState(null);
  const [error, setError] = useState("");

  /**
   * "loading" KHONG duoc luu thanh state: no chi la "chua tai xong cho dung tai
   * khoan nay". Luu no thi moi lan tai phai setState ba cai, va mot lan trong số
   * do roi vao luc body cua effect — React 19 bao cast render.
   */
  const refresh = useCallback(() => {
    if (!userId) return Promise.resolve([]);

    return fetchRecipients().then(
      (list) => {
        setItems(Array.isArray(list) ? list : []);
        setError("");
        setLoadedFor(userId);
        return list;
      },
      (caught) => {
        // Khong xoá danh sách đang có khi tải lại thất bại: khách vẫn còn người
        // nhận để chọn, mất mạng không được biến thành mất dữ liệu trên màn hình.
        // (401 thi shopApi da xoa phien; loi se theo `error` ra man hinh.)
        setError(caught.message || "Không tải được sổ người nhận.");
        setLoadedFor(userId);
        throw caught;
      }
    );
  }, [userId]);

  useEffect(() => {
    // .catch() nay khong an loi: loi da nam trong state `error` de trang hien ra;
    // o day no chi ngan mot unhandled rejection tu phia effect.
    refresh().catch(() => {});
  }, [refresh]);

  const status = !userId
    ? "idle"
    : loadedFor !== userId
      ? "loading"
      : error
        ? "error"
        : "ready";

  return {
    items: userId ? items : [],
    status,
    error,
    reload: refresh,
    setItems,
  };
}

export function defaultRecipient(items = []) {
  return items.find((item) => item.isDefault) ?? null;
}
