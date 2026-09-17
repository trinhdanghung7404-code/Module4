/** Shared loading / error / empty states so every list page behaves the same way. */
export default function StatusPanel({ variant, message, onRetry }) {
  if (variant === "error") {
    return (
      <div className="status status--error" role="alert">
        <p className="status__title">Có lỗi xảy ra</p>
        <p className="status__text">{message}</p>
        {onRetry ? (
          <button type="button" className="btn btn--ghost" onClick={onRetry}>
            Thử lại
          </button>
        ) : null}
      </div>
    );
  }

  if (variant === "empty") {
    return (
      <div className="status status--empty">
        <p className="status__title">{message || "Chưa có dữ liệu"}</p>
        {onRetry ? (
          <button type="button" className="btn btn--ghost" onClick={onRetry}>
            Tải lại
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div className="status status--loading">
      <span className="status__spinner" aria-hidden="true" />
      <p className="status__text">{message || "Đang tải…"}</p>
    </div>
  );
}
