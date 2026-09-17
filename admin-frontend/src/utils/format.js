const priceFormatter = new Intl.NumberFormat("vi-VN", {
  style: "currency",
  currency: "VND",
  maximumFractionDigits: 0,
});

/**
 * Đưa định dạng dùng chung ra một chỗ.
 *
 * Trang sản phẩm trước đây tự khai formatPrice trong file của nó; thêm trang đơn
 * hàng mà copy lại dòng đó là hai chỗ chỉnh kiểu tiền tệ, và chúng sẽ lệch nhau
 * vào lúc không ai để ý.
 */
export function formatPrice(value) {
  return priceFormatter.format(Number(value) || 0);
}

/**
 * LocalDateTime của server gửi về dạng "2026-09-17T11:37:24.123" KHÔNG kèm múi giờ.
 * JS bản hiện đại coi chuỗi date-time không offset là giờ địa phương của máy — đúng
 * ý nghĩa dữ liệu này (server và người dùng cùng ở +07), nên chỉ cần new Date().
 */
function parse(value) {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatDateTime(value) {
  const date = parse(value);

  if (!date) {
    return "—";
  }

  return date.toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDate(value) {
  const date = parse(value);

  if (!date) {
    return "—";
  }

  return date.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}
