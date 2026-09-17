/**
 * Máy trạng thái ĐỂ HIỂN THỊ, không phải để quyết định.
 *
 * Luật "đơn được chuyển sang đâu" nằm ở backend (OrderStatusRules) và client không
 * có quyền đó — ở đây chỉ cần biết bốn mốc thuận để vẽ cái timeline. Đổi lại, mọi
 * nhãn đều ưu tiên chuỗi server gửi sang (statusLabel / history[].toStatusLabel);
 * FALLBACK_LABELS chỉ dùng khi dữ liệu cũ không còn nhãn kèm theo. Hai bản dịch
 * cùng tồn tại mà không có bản nào thắng là cách chắc để khách thấy "Đã xác nhận"
 * ở trang này còn trang kia báo "Chờ lấy hàng".
 */
export const FLOW_STEPS = ["PENDING", "CONFIRMED", "SHIPPING", "DELIVERED"];

// Người dùng chỉ thấy chữ này khi server không gửi kèm chữ đó. Nằm dưới đây
// trùng khít OrderStatusRules.labels() trong backend (nó là nơi duy nhất có
// thẩm quyền đặt tên cho trạng thái), vì không có gì tệ hơn việc cùng một trạng
// thái mà hai chỗ gọi tên khác nhau.
const FALLBACK_LABELS = {
  PENDING: "Chờ xác nhận",
  CONFIRMED: "Chờ lấy hàng",
  SHIPPING: "Đang vận chuyển",
  DELIVERED: "Giao hàng thành công",
  CANCELLED: "Đã hủy",
};

/**
 * status -> mốc thời gian đầu tiên đơn bước vào status đó.
 *
 * Đọc từ history (server ghi lại từng bước). Trạng thái hiện tại mà history không
 * có — ví dụ đơn cũ tạo trước khi có bảng lịch sử — thì lấy statusUpdatedAt, rồi
 * cuối cùng mới đến createdAt: createdAt chỉ là thời điểm tạo row, không phải bằng
 * chứng đơn đã ở trạng thái đó từ lúc đó.
 */
export function reachedMap(order) {
  const reached = new Map();

  for (const line of order?.history ?? []) {
    if (line?.toStatus && !reached.has(line.toStatus)) {
      reached.set(line.toStatus, line.createdAt ?? null);
    }
  }

  if (order?.status && !reached.has(order.status)) {
    reached.set(order.status, order.statusUpdatedAt ?? order.createdAt ?? null);
  }

  // Mỗi đơn sinh ra đã ở PENDING nên mốc này luôn đã qua, kể cả khi lịch sử
  // không ghi lại — tạo đơn không phải một lần chuyển trạng thái. Không có
  // trạng thái thì không có gì để khăng: đơn rỗng (`{}`) vẫn là đơn rỗng.
  if (order?.status && !reached.has("PENDING")) {
    reached.set("PENDING", order.createdAt ?? null);
  }

  return reached;
}

/**
 * Bốn mốc thuận, mỗi mốc kèm nhãn + đã đạt hay chưa + mốc thời gian.
 *
 * Bước CHƯA đạt vẫn được trả về có chủ đích: timeline mà ẩn các bước tương lai thì
 * khách không biết đơn của mình còn phải đi qua những gì.
 *
 * `done` không chỉ là "có mặt trong history". Đơn đi một chiều (SHIPPING không quay
 * lại CONFIRMED, DELIVERED và CANCELLED là hai trạng thái cuối — xem
 * OrderStatusRules.nextMap), nên mọi mốc đứng TRƯỚC mốc hiện tại đều đã qua, kể cả
 * khi dòng đó bị mất trong lịch sử. `at` thì không đoán được: đúng là đã qua, nhưng
 * không có bằng chứng về thời gian thì để trống — dán "Chưa tới" cho một mốc thực tế
 * đã đi qua còn tệ hơn dán một giờ cụ thể chưa từng được ghi lại ở đâu.
 */
export function buildSteps(order) {
  const reached = reachedMap(order);
  const currentIndex = FLOW_STEPS.indexOf(order?.status);

  return FLOW_STEPS.map((status, index) => ({
    status,
    label: labelFor(order, status),
    done: reached.has(status) || (currentIndex > -1 && index <= currentIndex),
    current: order?.status === status,
    at: reached.get(status) ?? null,
  }));
}

export function statusLabel(order) {
  if (!order?.status) {
    return "";
  }

  return labelFor(order, order.status);
}

export function isCancelled(order) {
  return order?.status === "CANCELLED";
}

/**
 * Những gì trang khách được nói về một đơn đã hủy: lúc hủy + ghi chú của bước hủy.
 *
 * Ghi chú lấy từ dòng lịch sử, tức là chữ admin gõ lúc bấm hủy — và khi server không
 * trả được hàng về kho thì chính nó ghi thêm "Không trả được kho: ...". Vì vậy client
 * không được tự khẳng định "hàng đã quay lại kho": có những đơn sản phẩm đã bị xoá vật
 * lý nên chẳng có gì để trả, nói bừa là khách tưởng số tồn đã đúng rồi.
 *
 * Trả về `null` với đơn chưa hủy, để chỗ gọi không phải tự kiểm tra trạng thái.
 */
export function cancelInfo(order) {
  if (!isCancelled(order)) {
    return null;
  }

  const lines = order?.history ?? [];

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i]?.toStatus === "CANCELLED") {
      return { at: lines[i].createdAt ?? null, note: lines[i].note ?? null };
    }
  }

  return { at: order.statusUpdatedAt ?? null, note: null };
}

function labelFor(order, status) {
  const line = (order?.history ?? []).find((entry) => entry?.toStatus === status);

  if (line?.toStatusLabel) {
    return line.toStatusLabel;
  }

  if (order?.status === status && order.statusLabel) {
    return order.statusLabel;
  }

  return FALLBACK_LABELS[status] ?? status;
}
