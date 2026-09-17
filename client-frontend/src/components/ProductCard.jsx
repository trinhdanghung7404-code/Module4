import { useState } from "react";
import { Link } from "react-router-dom";
import ProductImage from "./ProductImage";
import Price from "./Price";
import { useCart } from "../hooks/useCart";

export default function ProductCard({ product }) {
  const { lines, add } = useCart();
  const [note, setNote] = useState(null);

  const stock = Number(product.quantity);
  const hasStockInfo = Number.isFinite(stock);
  const outOfStock = hasStockInfo && stock <= 0;
  const inCart = lines.find((line) => line.productId === product.id);

  function handleAdd() {
    const result = add(product, 1);
    if (!result.ok) {
      setNote(result.reason);
      return;
    }
    setNote(result.clampedTo ? `Kho chỉ còn ${result.clampedTo}, đã chỉnh lại.` : null);
  }

  return (
    <article className="card">
      <Link
        to={`/product/${product.id}`}
        className="card__media"
        aria-label={product.name}
      >
        <ProductImage
          src={product.imageUrl}
          alt={product.name}
          className="card__img"
        />
        {outOfStock ? <span className="card__flag">Hết hàng</span> : null}
      </Link>

      <div className="card__body">
        {product.categoryName ? (
          <span className="card__cat">{product.categoryName}</span>
        ) : null}

        <h3 className="card__title">
          <Link to={`/product/${product.id}`}>{product.name}</Link>
        </h3>

        <div className="card__meta">
          <Price value={product.price} className="card__price" />
          {hasStockInfo && !outOfStock ? (
            <span className="card__stock">Còn {stock}</span>
          ) : null}
        </div>

        <button
          type="button"
          className="btn btn--add"
          onClick={handleAdd}
          disabled={outOfStock}
        >
          {outOfStock
            ? "Hết hàng"
            : inCart
              ? `Đã thêm · ${inCart.quantity} trong giỏ`
              : "Thêm vào giỏ"}
        </button>

        {note ? <p className="card__note">{note}</p> : null}
      </div>
    </article>
  );
}
