import { buildQuery, request } from "./client";

/**
 * Đơn hàng phía quản trị.
 *
 * Trang này KHÔNG tự định nghĩa máy trạng thái. Mỗi đơn trả về kèm
 * `nextStatuses` — danh sách bước hợp lệ do OrderStatusRules ở server tính — và UI
 * chỉ vẽ nút từ danh sách đó. Hai bản sao của một luật ở hai nơi nghĩa là một trong
 * hai bản sẽ sai trước, thường là bản ở client vì nó không bị test nghiệp vụ chạm tới.
 */
export function fetchOrders({ status, search, page, size } = {}) {
  return request(`/orders${buildQuery({ status, search, page, size })}`);
}

export function fetchOrderSummary() {
  return request("/orders/summary");
}

export function fetchOrder(id) {
  return request(`/orders/${id}`);
}

/** payload: { toStatus, note?, shippingUnit?, trackingCode? } */
export function changeOrderStatus(id, payload) {
  return request(`/orders/${id}/status`, { method: "POST", body: payload });
}
