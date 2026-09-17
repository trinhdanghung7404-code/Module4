import { useCallback, useEffect, useState } from "react";
import { changeOrderStatus, fetchOrder } from "../api/orderApi";
import { formatDateTime, formatPrice } from "../utils/format";

const SHIPPING_STATUS = "SHIPPING";

/**
 * Modal chi tiết một đơn hàng + các bước chuyển trạng thái.
 *
 * Modal TỰ tải đơn theo id thay vì nhận nguyên object đang hiển thị ở hàng bảng,
 * và TẢI LẠI sau mỗi lần chuyển. Nhờ vậy nếu một admin khác vừa bấm trước thì đây
 * là chỗ thấy ngay trạng thái mới — chứ không phải một màn hình đóng đinh dữ liệu
 * của ba phút trước rồi ghi đè lên.
 *
 * Nút bấm nào được hiện lấy từ `order.nextStatuses` do server tính. UI không tự
 * đoán "đơn PENDING thì đương nhiên xác nhận được": luật đó nằm ở
 * OrderStatusRules, nhân bản sang JavaScript là cách chắc chắn để hai bản lệch nhau.
 */
function OrderDetailModal({ orderId, labels = {}, onClose, onChanged }) {
  const [order, setOrder] = useState(null);
  const [loadedId, setLoadedId] = useState(null);
  const [error, setError] = useState("");
  const [target, setTarget] = useState(null);
  const [shippingUnit, setShippingUnit] = useState("");
  const [trackingCode, setTrackingCode] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);

  /** Đang tải là suy ra từ "đã tải xong đúng đơn này chưa", không cần cái công tắc state. */
  const loading = order === null || loadedId !== orderId;

  const load = useCallback(() => {
    fetchOrder(orderId)
      .then((data) => {
        setOrder(data);
        setError("");
      })
      .catch((caught) => setError(caught.message))
      .finally(() => setLoadedId(orderId));
  }, [orderId]);

  useEffect(load, [load]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape" && !saving) {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, saving]);

  const chooseTarget = (status) => {
    setTarget(status);
    setError("");
  };

  const cancelTarget = () => {
    setTarget(null);
    setError("");
  };

  const submit = async () => {
    setSaving(true);
    setError("");

    try {
      const updated = await changeOrderStatus(orderId, {
        toStatus: target,
        note: note.trim() || null,
        shippingUnit: target === SHIPPING_STATUS ? shippingUnit.trim() || null : null,
        trackingCode:
          target === SHIPPING_STATUS ? trackingCode.trim() || null : null,
      });

      setOrder(updated);
      setTarget(null);
      setNote("");
      setShippingUnit("");
      setTrackingCode("");

      // Cho trang danh sách tải lại bảng + bộ đếm: đơn vừa đổi trạng thái có thể
      // phải rời khỏi bộ lọc đang xem (đang ở "Chờ xác nhận" thì nay đã xác nhận).
      onChanged?.();
    } catch (caught) {
      setError(caught.message);
    } finally {
      setSaving(false);
    }
  };

  const labelOf = (status) => labels[status] ?? status ?? "—";
  const needsShipping = target === SHIPPING_STATUS;
  const canSubmit = !saving && (!needsShipping || shippingUnit.trim().length > 0);

  return (
    <div className="modal-overlay" onMouseDown={saving ? undefined : onClose}>
      <div
        className="product-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="order-detail-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="order-detail-title">
              {order ? order.orderCode : `Đơn #${orderId}`}
            </h2>
            <p>
              {order
                ? `Đặt lúc ${formatDateTime(order.createdAt)}`
                : "Đang tải chi tiết đơn…"}
            </p>
          </div>

          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            disabled={saving}
            aria-label="Đóng"
          >
            ×
          </button>
        </div>

        {error ? (
          <div className="order-error" role="alert">
            {error}
          </div>
        ) : null}

        {order ? (
          <>
            <dl className="order-detail-grid">
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
              <div>
                <dt>Trạng thái</dt>
                <dd>
                  <span
                    className={`status-badge status-badge--${String(
                      order.status
                    ).toLowerCase()}`}
                  >
                    {order.statusLabel || labelOf(order.status)}
                  </span>
                  <span className="order-timeline__meta">
                    {" "}· {formatDateTime(order.statusUpdatedAt)}
                  </span>
                </dd>
              </div>
              <div>
                <dt>Đơn vị vận chuyển</dt>
                <dd>{order.shippingUnit || "—"}</dd>
              </div>
              <div>
                <dt>Mã vận đơn</dt>
                <dd>{order.trackingCode || "—"}</dd>
              </div>
              {order.note ? (
                <div className="full-column">
                  <dt>Ghi chú của khách</dt>
                  <dd>{order.note}</dd>
                </div>
              ) : null}
            </dl>

            <table className="order-detail-lines">
              <thead>
                <tr>
                  <th>Sản phẩm</th>
                  <th className="num">Đơn giá</th>
                  <th className="num">SL</th>
                  <th className="num">Thành tiền</th>
                </tr>
              </thead>
              <tbody>
                {order.items.map((item, index) => (
                  <tr key={`${item.productId ?? "product"}-${index}`}>
                    <td>{item.productName}</td>
                    <td className="num">{formatPrice(item.unitPrice)}</td>
                    <td className="num">{item.quantity}</td>
                    <td className="num">{formatPrice(item.lineTotal)}</td>
                  </tr>
                ))}
                <tr>
                  <td colSpan={3}>Tổng cộng</td>
                  <td className="num">{formatPrice(order.totalAmount)}</td>
                </tr>
              </tbody>
            </table>

            <h3 className="order-detail-heading">Diễn biến đơn</h3>
            <ol className="order-timeline">
              {order.history.map((entry, index) => (
                <li
                  className="order-timeline__item"
                  key={`${entry.toStatus}-${index}`}
                >
                  <div className="order-timeline__title">
                    {entry.toStatusLabel || labelOf(entry.toStatus)}
                  </div>
                  <div className="order-timeline__meta">
                    {formatDateTime(entry.createdAt)} · {describeActor(entry)}
                  </div>
                  {entry.note ? (
                    <div className="order-timeline__note">{entry.note}</div>
                  ) : null}
                </li>
              ))}
            </ol>

            {target ? (
              <div className="form-group">
                <label>Chuyển đơn này sang “{labelOf(target)}”</label>

                {needsShipping ? (
                  <>
                    <input
                      type="text"
                      value={shippingUnit}
                      maxLength={100}
                      placeholder="Đơn vị vận chuyển (bắt buộc)"
                      aria-label="Đơn vị vận chuyển"
                      disabled={saving}
                      onChange={(event) => setShippingUnit(event.target.value)}
                    />
                    <input
                      type="text"
                      value={trackingCode}
                      maxLength={60}
                      placeholder="Mã vận đơn (không bắt buộc)"
                      aria-label="Mã vận đơn"
                      disabled={saving}
                      onChange={(event) => setTrackingCode(event.target.value)}
                    />
                  </>
                ) : null}

                <textarea
                  rows={2}
                  value={note}
                  maxLength={500}
                  placeholder="Ghi chú cho bước này — hiển thị trong diễn biến đơn"
                  aria-label="Ghi chú chuyển trạng thái"
                  disabled={saving}
                  onChange={(event) => setNote(event.target.value)}
                />
              </div>
            ) : null}

            <div className="modal-actions">
              {target ? (
                <>
                  <button
                    type="button"
                    className="cancel-button"
                    onClick={cancelTarget}
                    disabled={saving}
                  >
                    Quay lại
                  </button>
                  <button
                    type="button"
                    className="save-button"
                    onClick={submit}
                    disabled={!canSubmit}
                  >
                    {saving ? "Đang lưu..." : `Xác nhận: ${labelOf(target)}`}
                  </button>
                </>
              ) : order.nextStatuses.length > 0 ? (
                order.nextStatuses.map((status) => (
                  <button
                    key={status}
                    type="button"
                    className={
                      status === "CANCELLED"
                        ? "status-action status-action--danger"
                        : "status-action"
                    }
                    disabled={saving || loading}
                    onClick={() => chooseTarget(status)}
                  >
                    {status === "CANCELLED" ? "Hủy đơn" : labelOf(status)}
                  </button>
                ))
              ) : (
                <span className="order-timeline__meta">
                  Đơn đã chốt ở trạng thái cuối, không còn bước nào để chuyển.
                </span>
              )}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Ai đã làm bước này. Phải đọc actorType TRƯỚC rồi mới đến tên: bản ghi do khách
 * hoặc do hệ thống ghi không có tên admin nào để hiển thị, và actorLabel là tên
 * chép tại thời điểm đó — tài khoản bị xóa về sau vẫn còn đọc được ai làm.
 */
function describeActor(entry) {
  if (entry.actorType === "ADMIN") {
    return entry.actorLabel ? `Quản trị ${entry.actorLabel}` : "Quản trị viên";
  }

  if (entry.actorType === "USER") {
    return "Khách hàng";
  }

  return "Hệ thống";
}

export default OrderDetailModal;
