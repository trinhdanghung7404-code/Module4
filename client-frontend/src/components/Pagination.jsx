function buildPageList(current, total, span = 1) {
  const pages = new Set([0, total - 1]);
  for (let p = current - span; p <= current + span; p += 1) {
    if (p >= 0 && p < total) pages.add(p);
  }

  const sorted = [...pages].sort((a, b) => a - b);
  const withGaps = [];
  let previous = null;

  sorted.forEach((p) => {
    if (previous !== null && p - previous > 1) withGaps.push("gap");
    withGaps.push(p);
    previous = p;
  });

  return withGaps;
}

/** `page` is 0-based, matching the backend contract; labels are 1-based for humans. */
export default function Pagination({ page, totalPages, onChange }) {
  if (!totalPages || totalPages <= 1) return null;

  return (
    <nav className="pagination" aria-label="Phân trang sản phẩm">
      <button
        type="button"
        className="pagination__btn"
        disabled={page <= 0}
        onClick={() => onChange(page - 1)}
      >
        ← Trước
      </button>

      <ul className="pagination__pages">
        {buildPageList(page, totalPages).map((item, index) =>
          item === "gap" ? (
            <li key={`gap-${index}`} className="pagination__gap">
              …
            </li>
          ) : (
            <li key={item}>
              <button
                type="button"
                className={`pagination__page${item === page ? " is-active" : ""}`}
                aria-current={item === page ? "page" : undefined}
                onClick={() => onChange(item)}
              >
                {item + 1}
              </button>
            </li>
          )
        )}
      </ul>

      <button
        type="button"
        className="pagination__btn"
        disabled={page >= totalPages - 1}
        onClick={() => onChange(page + 1)}
      >
        Sau →
      </button>
    </nav>
  );
}
