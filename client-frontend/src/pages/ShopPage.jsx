import { useEffect, useState } from "react";
import { useShopUi } from "../hooks/useShopUi";
import { DEFAULT_PAGE_SIZE, SORT_OPTIONS, fetchProducts } from "../api/shopApi";
import ProductCard from "../components/ProductCard";
import Pagination from "../components/Pagination";
import StatusPanel from "../components/StatusPanel";

export default function ShopPage() {
  const { filter, setFilter, resetFilters, page, setPage } = useShopUi();
  const [settled, setSettled] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const requestKey = [
    filter.categoryId,
    filter.search,
    filter.sort,
    page,
    attempt,
  ].join("#");

  useEffect(() => {
    let cancelled = false;

    fetchProducts({ ...filter, page, size: DEFAULT_PAGE_SIZE }).then(
      (data) => {
        if (!cancelled) setSettled({ key: requestKey, data, error: null });
      },
      (err) => {
        if (!cancelled) {
          setSettled({ key: requestKey, data: null, error: err.message });
        }
      }
    );

    // Requests overlap on purpose while the user clicks through filters: the old
    // response is dropped by its key, and this cleanup only fires for the real unmount.
    return () => {
      cancelled = true;
    };
  }, [filter, page, requestKey]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [page]);

  const isCurrent = settled?.key === requestKey;
  const data = settled?.data;
  const products = data?.content ?? [];
  const totalElements = data?.totalElements ?? 0;
  const totalPages = data?.totalPages ?? 0;
  const error = isCurrent ? settled.error : null;
  const firstLoad = !isCurrent && products.length === 0;
  const hasActiveFilter =
    filter.categoryId > 0 || filter.search !== "" || filter.sort !== "newest";

  return (
    <section className="shop-page">
      <div className="toolbar">
        <p className="toolbar__count" aria-live="polite">
          {error
            ? "Không tải được danh sách"
            : isCurrent
              ? `Có ${totalElements} sản phẩm`
              : "Đang tải…"}
        </p>

        <label className="toolbar__sort">
          <span>Sắp xếp</span>
          <select
            value={filter.sort}
            onChange={(event) => setFilter({ sort: event.target.value })}
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error ? (
        <StatusPanel
          variant="error"
          message={error}
          onRetry={() => setAttempt((n) => n + 1)}
        />
      ) : firstLoad ? (
        <StatusPanel variant="loading" />
      ) : products.length === 0 ? (
        <StatusPanel
          variant="empty"
          message="Không tìm thấy sản phẩm nào khớp bộ lọc."
          onRetry={hasActiveFilter ? resetFilters : undefined}
        />
      ) : (
        <>
          {isCurrent ? null : (
            <p className="shop-page__refreshing">Đang cập nhật…</p>
          )}

          <ul className="grid">
            {products.map((product) => (
              <li key={product.id}>
                <ProductCard product={product} />
              </li>
            ))}
          </ul>

          <Pagination page={page} totalPages={totalPages} onChange={setPage} />
        </>
      )}
    </section>
  );
}
