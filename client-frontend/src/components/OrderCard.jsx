import { orderSummary } from "../hooks/useOrders";
import { buildSteps, cancelInfo, statusLabel } from "../utils/orderFlow";
import { formatDate } from "../utils/format";
import Price from "./Price";

/**
 * Một đơn hàng trong "Đơn hàng của tôi".
 *
 * Tách khỏi OrderHistory vì thân thẻ dài gấp mấy lần khung danh sách: danh sách chỉ
 * cần biết "có những đơn nào, đang tải hay lỗi", còn kể một đơn đã đi tới đâu là
 * chuyện riêng của cái thẻ này — và là chỗ duy nhất phải sửa khi server thêm mốc.
 *
 * Timeline bốn mốc vẽ từ `history` mà server gửi sang — cùng nguồn dữ liệu với trang
 * quản trị, không phải một bản đoán từ chữ "Đã hủy" hay "Đã giao". Hai nơi cùng kể về
 * một đơn mà ra hai câu chuyện khác nhau là kiểu lỗi khó chuộc nhất.
 *
 * Nhận đúng một prop `order` và không tự gọi API nào: một danh sách 50 đơn vẫn là một
 * request duy nhất.
 */
export default function OrderCard({ order }) {
  const steps = buildSteps(order);
  // cancelInfo() null cũng chính là "đơn chưa hủy", nên không cần thêm
  // một biến `cancelled` thứ hai có thể lệch pha nó.
  const cancel = cancelInfo(order);

  return (
    <li className="order">
      <div className="order__head">
        <span className="order__code">{order.orderCode}</span>
        <span
          className={`order__status order__status--${String(order.status ?? "").toLowerCase()}`}
        >
          {statusLabel(order) || order.status}
        </span>
      </div>

      {cancel ? (
        <p className="order__cancel">
          Đơn đã hủy
          {cancel.at ? ` ngày ${formatDate(cancel.at)}` : ""}.
          {cancel.note ? ` Ghi chú khi hủy: ${cancel.note}` : ""}
        </p>
      ) : (
        <ol className="order__steps" aria-label="Diễn biến đơn hàng">
          {steps.map((step) => (
            <li
              key={step.status}
              className={
                "order__step" +
                (step.done ? " order__step--done" : "") +
                (step.current ? " order__step--current" : "")
              }
            >
              <span className="order__step-dot" aria-hidden="true" />
              <span className="order__step-label">{step.label}</span>
              <span className="order__step-at">
                {step.at
                  ? formatDate(step.at)
                  : step.done
                    ? "Đã qua"
                    : "Chưa tới"}
              </span>
            </li>
          ))}
        </ol>
      )}

      <p className="order__meta">
        {formatDate(order.createdAt)} · {orderSummary(order)}
      </p>
      <p className="order__to">
        Giao cho <strong>{order.customerName}</strong> · {order.phone}
      </p>
      <p className="order__address">{order.address}</p>
      {order.shippingUnit ? (
        <p className="order__ship">
          Đơn vị vận chuyển: <strong>{order.shippingUnit}</strong>
          {order.trackingCode ? ` · Mã vận đơn: ${order.trackingCode}` : ""}
        </p>
      ) : null}
      {order.note ? <p className="order__note">Ghi chú: {order.note}</p> : null}

      <Price value={order.totalAmount} className="order__total" />
    </li>
  );
}
