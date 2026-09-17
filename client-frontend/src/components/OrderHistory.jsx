import { useOrders } from "../hooks/useOrders";
import OrderCard from "./OrderCard";

/**
 * "Đơn hàng của tôi".
 *
 * Danh sách này không thể hỏi về người khác được: đường dẫn là /orders/mine và
 * token quyết định nội dung trả về, không có tham số nào để đoán.
 *
 * Chưa có trang chi tiết đơn theo mã DH-xxxxxx và chưa nên có. Mã đơn được đánh
 * tăng dần theo id (sau DH-000009 là DH-000010), nên một endpoint công khai
 * /orders/by-code/{code} sẽ cho phép đoán lần từng đơn của người khác và đọc tên,
 * SĐT, địa chỉ của họ. Làm trang chi tiết phải đi cùng việc xác nhận quyền sở hữu
 * đơn — không phải chỉ thêm một đường dẫn.
 *
 * Ở đây chỉ còn khung danh sách + ba trạng thái tải. Một đơn được kể thế nào là
 * việc của OrderCard.
 */

export default function OrderHistory() {
  const { items, status, error, reload } = useOrders();

  return (
    <section className="orders" aria-labelledby="orders-title">
      <h2 id="orders-title">Đơn hàng của tôi</h2>
      <p className="field__hint">
        50 đơn gần nhất, mới nhất lên đầu. Địa chỉ ở đây là bản chép lúc đặt hàng,
        nên sửa hay xoá sổ người nhận không làm đổi đơn đã chốt.
      </p>

      {status === "loading" ? <p className="banner">Đang tải đơn hàng…</p> : null}

      {status === "error" ? (
        <div className="banner banner--danger" role="alert">
          {error}{" "}
          <button type="button" className="link-btn" onClick={reload}>
            Thử lại
          </button>
        </div>
      ) : null}

      {status === "ready" && items.length === 0 ? (
        <p className="banner">Bạn chưa đặt đơn nào ở tài khoản này.</p>
      ) : null}

      <ul className="order-list">
        {items.map((order) => (
          <OrderCard key={order.orderCode} order={order} />
        ))}
      </ul>
    </section>
  );
}
