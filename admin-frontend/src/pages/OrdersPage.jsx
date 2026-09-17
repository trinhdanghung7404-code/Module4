import { useCallback, useEffect, useState } from "react";
import { fetchOrderSummary, fetchOrders } from "../api/orderApi";
import OrderDetailModal from "../components/OrderDetailModal";
import { ErrorIcon, SuccessIcon, ViewIcon } from "../components/TableIcons";
import { formatDateTime, formatPrice } from "../utils/format";
import "../styles/product.css";
import "../styles/orders.css";

const PAGE_SIZE = 20;

/** Đúng thứ tự khai báo của OrderStatus trong backend — luồng đi theo chiều này. */
const STATUS_ORDER = ["PENDING", "CONFIRMED", "SHIPPING", "DELIVERED", "CANCELLED"];

/**
 * Chỉ là ô đỡ khi /orders/summary chưa về kịp (hoặc lỗi). Bản chính thức đến từ
 * OrderStatusRules ở server, qua summary.labels — để hai bên không dịch lệch nhau.
 */
const FALLBACK_LABELS = {
  PENDING: "Chờ xác nhận",
  CONFIRMED: "Chờ lấy hàng",
  SHIPPING: "Đang vận chuyển",
  DELIVERED: "Giao hàng thành công",
  CANCELLED: "Đã hủy",
};

const initialToast = { message: "", type: "success" };

/** Tóm tắt dòng hàng cho một dòng bảng: admin nhớ mã đơn, không nhớ hết tên sản phẩm. */
function describeLines(order) {
  const lines = order.items ?? [];
  const units = lines.reduce((sum, item) => sum + (Number(item.quantity) || 0), 0);

  if (lines.length === 0) {
    return "Không có sản phẩm nào";
  }

  if (lines.length === 1) {
    return `${lines[0].productName} × ${units}`;
  }

  return `${lines.length} sản phẩm (${units} cái)`;
}

function OrdersPage() {
  const [summary, setSummary] = useState(null);
  const [status, setStatus] = useState("ALL");
  const [searchDraft, setSearchDraft] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [result, setResult] = useState({
    content: [],
    page: 0,
    size: PAGE_SIZE,
    totalElements: 0,
    totalPages: 0,
    last: true,
  });
  const [error, setError] = useState("");
  const [detailId, setDetailId] = useState(null);
  const [toast, setToast] = useState(initialToast);

  /**
   * Đang tải là SUY RA, không phải state: "đã tải xong bộ lọc này chưa?" so với
   * "bộ lọc nào đang xem?". Gọi setLoading(true) rồi setState(false) trong effect
   * bị bản react-hooks mới chặn (react-hooks/set-state-in-effect) và thật ra cũng
   * thừa: mỗi lần đổi bộ lọc là một lần chạy effect, suy ra không cần ai bật công
   * tắc. Cách này client-frontend đang dùng ở useOrders/useRecipients.
   */
  const filterKey = `${status}|${search}|${page}`;
  const [loadedKey, setLoadedKey] = useState(null);
  const loading = loadedKey !== filterKey;

  const labels = summary?.labels ?? FALLBACK_LABELS;
  const labelOf = (value) => labels[value] ?? FALLBACK_LABELS[value] ?? value;

  const loadOrders = useCallback(() => {
    const requestedKey = filterKey;

    fetchOrders({ status, search, page, size: PAGE_SIZE })
      .then((data) => {
        setResult(data);
        setError("");
      })
      .catch((caught) => setError(caught.message))
      // Kể cả khi lỗi cũng đánh dấu "đã tải xong bộ lọc này", nếu không màn hình
      // sẽ treo chữ "Đang tải…" mãi mãi thay vì hiện thông báo lỗi.
      .finally(() => setLoadedKey(requestedKey));
  }, [status, search, page, filterKey]);

  const loadSummary = useCallback(() => {
    // Đếm chỉ để trang trí mấy cái chip; lỗi ở đây không được chặn danh sách đơn.
    fetchOrderSummary()
      .then(setSummary)
      .catch(() => setSummary(null));
  }, []);

  useEffect(loadOrders, [loadOrders]);
  useEffect(loadSummary, [loadSummary]);

  useEffect(() => {
    if (!toast.message) {
      return undefined;
    }

    const timer = window.setTimeout(() => setToast(initialToast), 3500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const handleSearch = (event) => {
    event.preventDefault();
    setPage(0);
    setSearch(searchDraft.trim());
  };

  const chooseStatus = (next) => {
    setStatus(next);
    setPage(0);
  };

  /**
   * Một đơn vừa đổi trạng thái có thể rời khỏi bộ lọc đang xem (đang lọc "Chờ xác
   * nhận" thì đơn vừa xác nhận phải biến mất) và chắc chắn làm thay đổi bộ đếm —
   * nên sau mỗi lần thành công phải tải lại cả hai.
   */
  const handleChanged = () => {
    loadOrders();
    loadSummary();
    setToast({ message: "Đã cập nhật trạng thái đơn hàng", type: "success" });
  };

  const counts = summary?.byStatus ?? {};
  const chips = [
    { key: "ALL", label: "Tất cả", count: summary?.total },
    ...STATUS_ORDER.map((value) => ({
      key: value,
      label: labelOf(value),
      count: counts[value] ?? 0,
    })),
  ];

  return (
    <div className="product-page">
      <div className="product-container orders-container">
        <header className="product-page-header">
          <div>
            <h1>Đơn hàng</h1>
            <p>
              Xác nhận, bàn giao cho đơn vị vận chuyển và ghi nhận đã giao. Mỗi
              bước đều lưu lại ai làm và làm lúc nào.
            </p>
          </div>

          <form className="order-search" onSubmit={handleSearch}>
            <input
              type="search"
              value={searchDraft}
              placeholder="Tìm theo mã đơn, tên hoặc số điện thoại"
              aria-label="Tìm đơn hàng"
              onChange={(event) => setSearchDraft(event.target.value)}
            />
            <button type="submit" className="small-button" disabled={loading}>
              Tìm
            </button>
          </form>
        </header>

        <div className="status-filter-bar">
          {chips.map((chip) => (
            <button
              key={chip.key}
              type="button"
              className={
                status === chip.key ? "status-chip active" : "status-chip"
              }
              aria-pressed={status === chip.key}
              onClick={() => chooseStatus(chip.key)}
            >
              {chip.label}
              <span className="status-chip__count">{chip.count ?? 0}</span>
            </button>
          ))}
        </div>

        <section className="product-table-card">
          {loading ? (
            <div className="empty-products">Đang tải đơn hàng…</div>
          ) : error ? (
            <div className="empty-products">{error}</div>
          ) : result.content.length === 0 ? (
            <div className="empty-products">
              Chưa có đơn hàng nào ở bộ lọc này.
            </div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Mã đơn</th>
                    <th>Người nhận</th>
                    <th>Sản phẩm</th>
                    <th>Tổng tiền</th>
                    <th>Trạng thái</th>
                    <th>Cập nhật</th>
                    <th className="actions-header">Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {result.content.map((order, index) => (
                    <tr key={order.id}>
                      <td>{result.page * result.size + index + 1}</td>
                      <td>{order.orderCode}</td>
                      <td className="order-customer">
                        <strong>{order.customerName}</strong>
                        <span>{order.phone}</span>
                      </td>
                      <td className="order-lines">{describeLines(order)}</td>
                      <td>{formatPrice(order.totalAmount)}</td>
                      <td>
                        <span
                          className={`status-badge status-badge--${String(
                            order.status
                          ).toLowerCase()}`}
                        >
                          {order.statusLabel || labelOf(order.status)}
                        </span>
                      </td>
                      <td className="order-lines">
                        {formatDateTime(
                          order.statusUpdatedAt ?? order.createdAt
                        )}
                      </td>
                      <td>
                        <div className="order-actions">
                          <button
                            type="button"
                            className="action-button view-action"
                            title="Xem chi tiết và diễn biến đơn"
                            aria-label={`Xem đơn ${order.orderCode}`}
                            onClick={() => setDetailId(order.id)}
                          >
                            <ViewIcon />
                          </button>

                          {/* Nút nào được bấm là do server liệt kê trong
                              nextStatuses — UI không tự đoán luật. */}
                          {(order.nextStatuses ?? []).map((next) => (
                            <button
                              key={next}
                              type="button"
                              className={
                                next === "CANCELLED"
                                  ? "status-action status-action--danger"
                                  : "status-action"
                              }
                              title={`Chuyển sang ${labelOf(next)}`}
                              onClick={() => setDetailId(order.id)}
                            >
                              {next === "CANCELLED" ? "Hủy" : labelOf(next)}
                            </button>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result.totalPages > 1 ? (
            <div className="pagination-bar">
              <span>
                Trang {result.page + 1} / {result.totalPages} ·{" "}
                {result.totalElements} đơn
              </span>

              <div className="order-actions">
                <button
                  type="button"
                  className="small-button"
                  disabled={loading || result.page === 0}
                  onClick={() => setPage((previous) => previous - 1)}
                >
                  ← Trước
                </button>
                <button
                  type="button"
                  className="small-button"
                  disabled={loading || result.last}
                  onClick={() => setPage((previous) => previous + 1)}
                >
                  Sau →
                </button>
              </div>
            </div>
          ) : null}


        </section>
      </div>

      {detailId !== null ? (
        <OrderDetailModal
          orderId={detailId}
          labels={labels}
          onClose={() => setDetailId(null)}
          onChanged={handleChanged}
        />
      ) : null}

      {toast.message ? (
        <div
          className={`product-toast ${toast.type}`}
          role="status"
          aria-live="polite"
        >
          <span className="product-toast-icon">
            {toast.type === "error" ? <ErrorIcon /> : <SuccessIcon />}
          </span>

          <span className="product-toast-message">{toast.message}</span>

          <button
            type="button"
            className="product-toast-close"
            onClick={() => setToast(initialToast)}
            aria-label="Đóng thông báo"
          >
            ×
          </button>
        </div>
      ) : null}
    </div>
  );
}

export default OrdersPage;

