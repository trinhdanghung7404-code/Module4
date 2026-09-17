import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchProduct } from "../api/shopApi";
import ProductImage from "../components/ProductImage";
import Price from "../components/Price";
import StatusPanel from "../components/StatusPanel";
import { useCart } from "../hooks/useCart";
import { formatDate } from "../utils/format";

const ANGLES = [
  { type: "FRONT", label: "Mặt trước" },
  { type: "BACK", label: "Mặt sau" },
  { type: "LEFT", label: "Cạnh trái" },
  { type: "RIGHT", label: "Cạnh phải" },
];

export default function ProductDetailPage() {
  const { id } = useParams();
  const { add, lines } = useCart();
  const [settled, setSettled] = useState(null);
  const [active, setActive] = useState(0);
  const [quantity, setQuantity] = useState(1);
  const [note, setNote] = useState(null);
  const [attempt, setAttempt] = useState(0);

  // attempt is folded into the key so "Thử lại" re-runs the effect.
  const requestKey = `${id}#${attempt}`;

  useEffect(() => {
    let cancelled = false;

    // No reset here: the route remounts this page when :id changes (see App.jsx),
    // and "loading" is derived from whether `settled` matches the current key.
    fetchProduct(id).then(
      (data) => {
        if (!cancelled) setSettled({ key: requestKey, data, error: null });
      },
      (err) => {
        if (!cancelled) {
          setSettled({ key: requestKey, data: null, error: err.message });
        }
      }
    );

    return () => {
      cancelled = true;
    };
  }, [id, requestKey]);

  const product = settled?.data;

  /** Four fixed angles first, then anything else the admin uploaded. */
  const shots = useMemo(() => {
    const images = Array.isArray(product?.images) ? product.images : [];
    const byType = new Map(
      images
        .filter((img) => img?.imageType && img?.imageUrl)
        .map((img) => [img.imageType, img.imageUrl])
    );

    const known = ANGLES
      .map((angle) => ({ ...angle, url: byType.get(angle.type) ?? null }))
      .filter((shot) => shot.url);

    const extras = images
      .filter(
        (img) =>
          img?.imageUrl && !ANGLES.some((angle) => angle.type === img.imageType)
      )
      .map((img, index) => ({
        type: `OTHER-${index}`,
        label: "Ảnh khác",
        url: img.imageUrl,
      }));

    return [...known, ...extras];
  }, [product]);

  const isCurrent = settled?.key === requestKey;

  if (isCurrent && settled.error) {
    return (
      <StatusPanel
        variant="error"
        message={settled.error}
        onRetry={() => setAttempt((n) => n + 1)}
      />
    );
  }

  if (!product) {
    return <StatusPanel variant="loading" message="Đang tải sản phẩm…" />;
  }

  const stock = Number(product.quantity);
  const hasStockInfo = Number.isFinite(stock);
  const outOfStock = hasStockInfo && stock <= 0;
  const inCart =
    lines.find((line) => line.productId === product.id)?.quantity ?? 0;
  const canAddMore = !hasStockInfo || inCart + quantity <= stock;
  const currentShot = shots[active];

  function changeQuantity(next) {
    const target = Number(next);
    if (!Number.isFinite(target) || target < 1) {
      setQuantity(1);
      return;
    }
    if (hasStockInfo && target > stock) {
      setQuantity(stock);
      setNote(`Kho chỉ còn ${stock}.`);
      return;
    }
    setQuantity(target);
    setNote(null);
  }

  function handleAdd() {
    const result = add(product, quantity);
    setNote(
      result.ok
        ? result.clampedTo
          ? `Đã chỉnh lại theo kho: ${result.clampedTo}.`
          : null
        : result.reason
    );
  }

  return (
    <section className="detail">
      <nav className="crumbs" aria-label="Đường dẫn">
        <Link to="/">Cửa hàng</Link>
        {product.categoryName ? (
          <>
            <span className="crumbs__sep">/</span>
            <span>{product.categoryName}</span>
          </>
        ) : null}
        <span className="crumbs__sep">/</span>
        <span className="crumbs__current">{product.name}</span>
      </nav>

      <div className="detail__layout">
        <div className="detail__gallery">
          <div className="detail__stage">
            <ProductImage
              src={currentShot?.url ?? null}
              alt={
                currentShot
                  ? `${product.name} – ${currentShot.label}`
                  : product.name
              }
              className="detail__stage-img"
              ratio="4 / 3"
            />
            <span className="detail__stage-caption">
              {currentShot ? currentShot.label : "Chưa có ảnh cho sản phẩm này"}
            </span>
          </div>

          {shots.length > 1 ? (
            <ul className="thumbs">
              {shots.map((shot, index) => (
                <li key={`${shot.type}-${index}`}>
                  <button
                    type="button"
                    className={`thumb${index === active ? " is-active" : ""}`}
                    onClick={() => setActive(index)}
                    aria-label={`Xem ${shot.label}`}
                  >
                    <ProductImage src={shot.url} alt={shot.label} ratio="1 / 1" />
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <div className="detail__info">
          {product.categoryName ? (
            <span className="detail__cat">{product.categoryName}</span>
          ) : null}
          <h1 className="detail__title">{product.name}</h1>
          {product.sku ? <p className="detail__sku">Mã SP: {product.sku}</p> : null}

          <Price value={product.price} className="detail__price" />

          <p className={`detail__stock${outOfStock ? " is-out" : ""}`}>
            {outOfStock
              ? "Tạm hết hàng"
              : hasStockInfo
                ? `Còn ${stock} trong kho`
                : "Chưa cập nhật số tồn"}
          </p>

          {product.description ? (
            <div className="detail__desc">
              <h2 className="detail__sub">Mô tả</h2>
              {/* pre-wrap keeps the admin's plain-text line breaks; the description is
                  never rendered as HTML. */}
              <p className="detail__desc-text">{product.description}</p>
            </div>
          ) : null}

          <div className="buybox">
            <div className="stepper" role="group" aria-label="Số lượng">
              <button
                type="button"
                onClick={() => changeQuantity(quantity - 1)}
                disabled={quantity <= 1}
                aria-label="Giảm số lượng"
              >
                −
              </button>
              <input
                type="number"
                min="1"
                max={hasStockInfo ? stock : undefined}
                value={quantity}
                onChange={(event) => changeQuantity(event.target.value)}
                aria-label="Số lượng"
              />
              <button
                type="button"
                onClick={() => changeQuantity(quantity + 1)}
                disabled={hasStockInfo && quantity >= stock}
                aria-label="Tăng số lượng"
              >
                +
              </button>
            </div>

            <button
              type="button"
              className="btn btn--primary"
              onClick={handleAdd}
              disabled={outOfStock || !canAddMore}
            >
              {outOfStock ? "Hết hàng" : "Thêm vào giỏ"}
            </button>
          </div>

          {note ? <p className="detail__note">{note}</p> : null}

          {inCart > 0 ? (
            <p className="detail__incart">
              Đang có <strong>{inCart}</strong> trong giỏ ·{" "}
              <Link to="/cart">Xem giỏ hàng</Link>
            </p>
          ) : null}

          <p className="detail__date">Thêm ngày {formatDate(product.createdAt)}</p>
        </div>
      </div>
    </section>
  );
}

