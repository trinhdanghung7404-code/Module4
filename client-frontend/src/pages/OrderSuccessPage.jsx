import { Link } from "react-router-dom";
import { readOrderReceipt } from "../utils/orderPayload";
import { formatPrice } from "../utils/format";

const STATUS_LABELS = {
  PENDING: "Chờ xác nhận",
  CONFIRMED: "Đã xác nhận",
  SHIPPING: "Đang giao hàng",
  DELIVERED: "Giao thành công",
  CANCELLED: "Đã hủy",
};

/**
 * Trang xác nhận sau khi đặt hàng.
 *
 * Đọc từ sessionStorage nên F5 vẫn còn đơn; nhưng đây chỉ là BIÊN LAI ở phía
 * người dùng — nguồn thật của đơn hàng nằm trong bảng orders, backend chưa có
 * endpoint tra cứu theo mã đơn (mã DH-xxxxxx tăng dần nên mở endpoint tra cứu là
 * cho phép dò đơn của người khác).
 */
export default function OrderSuccessPage() {
  const order = readOrderReceipt();

  if (!order) {
    return (
      <section className="order-success order-success--empty">
        <h1>Không tìm thấy đơn hàng</h1>
        <p>
          Trình duyệt không còn lưu biên lai của đơn vừa đặt — có thể bạn đã mở tab
          mới hoặc tắt tab cũ.
        </p>
        <Link to="/" className="btn btn--primary">
          Tiếp tục mua sắm
        </Link>
      </section>
    );
  }

  const items = Array.isArray(order.items) ? order.items : [];

  return (
    <section className="order-success">
      <p className="order-success__mark" aria-hidden="true">
        ✓
      </p>
      <h1>Đặt hàng thành công</h1>
      <p className="order-success__lead">
        Mã đơn của bạn: <strong>{order.orderCode}</strong>
      </p>
      <p className="order-success__status">
        Trạng thái: {STATUS_LABELS[order.status] ?? order.status}
      </p>

      <div className="receipt">
        <dl className="receipt__facts">
          <div>
            <dt>Người nhận</dt>
            <dd>{order.customerName}</dd>
          </div>
          <div>
            <dt>Điện thoại</dt>
            <dd>{order.phone}</dd>
          </div>
          <div>
            <dt>Địa chỉ</dt>
            <dd>{order.address}</dd>
          </div>
          {order.note ? (
            <div>
              <dt>Ghi chú</dt>
              <dd>{order.note}</dd>
            </div>
          ) : null}
        </dl>

        <ul className="receipt__items">
          {items.map((item, index) => (
            <li key={`${item.productId ?? "product"}-${index}`} className="receipt__item">
              <span className="receipt__item-name">{item.productName}</span>
              <span className="receipt__item-qty">
                {item.quantity} × {formatPrice(item.unitPrice)}
              </span>
              <span className="receipt__item-total">
                {formatPrice(item.lineTotal)}
              </span>
            </li>
          ))}
        </ul>

        <p className="summary__row">
          <span>Tổng thanh toán</span>
          <strong>{formatPrice(order.totalAmount)}</strong>
        </p>
      </div>

      <Link to="/" className="btn btn--primary">
        Tiếp tục mua sắm
      </Link>
    </section>
  );
}
