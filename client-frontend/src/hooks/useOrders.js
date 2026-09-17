import { useCallback, useEffect, useState } from "react";
import { fetchMyOrders } from "../api/userApi";
import { useUser } from "./useUser";

/**
 * "Đơn hàng của tôi" — 50 đơn gần nhất của tài khoản đang đăng nhập.
 *
 * Sao chep cach lam cua useRecipients (cung mot nhan "dang tai xong cho ai", cung
 * khong luu `loading` thanh state): endpoint khong cho phep hoi don cua nguoi
 * khac, nen o day khong co tham so nao de truyen sai duoc nua.
 */
export function useOrders() {
  const { userId } = useUser();
  const [items, setItems] = useState([]);
  const [loadedFor, setLoadedFor] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    if (!userId) return Promise.resolve([]);

    return fetchMyOrders().then(
      (list) => {
        setItems(Array.isArray(list) ? list : []);
        setError("");
        setLoadedFor(userId);
        return list;
      },
      (caught) => {
        // Giữ danh sách cũ khi tải lại thất bại: đây là dữ liệu đã xem rồi, mất
        // mạng không được biến thành "bạn chưa đặt đơn nào".
        setError(caught.message || "Không tải được danh sách đơn hàng.");
        setLoadedFor(userId);
        throw caught;
      }
    );
  }, [userId]);

  useEffect(() => {
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
  };
}

/** Tom tat don cho danh sach: khach nho ma DH-xxxxxx chu khong nho ten ca don. */


export function orderSummary(order) {
  const lines = order?.items ?? [];
  const units = lines.reduce((sum, item) => sum + (Number(item.quantity) || 0), 0);
  if (lines.length === 0) return "Không có sản phẩm nào";
  if (lines.length === 1) return `${lines[0].productName} × ${units}`;
  return `${lines.length} sản phẩm (${units} cái)`;
}
