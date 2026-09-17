import { Link, useNavigate } from "react-router-dom";
import { useCart } from "../hooks/useCart";
import { useCartVerification } from "../hooks/useCartVerification";
import { isLoggedIn } from "../utils/userSession";
import ProductImage from "../components/ProductImage";
import Price from "../components/Price";
import { formatPrice } from "../utils/format";

export default function CartPage() {
  const { lines, subtotal, count, setQuantity, remove, clear, syncWithProducts } =
    useCart();
  const navigate = useNavigate();

  // Việc đối chiếu giá/tồn kho nằm trong hook dùng chung, vì trang thanh toán phải
  // gate đúng cùng một điều kiện: chưa biết số liệu thật của server thì chưa cho
  // gửi đơn.
  const { validated, isVerified } = useCartVerification(lines, syncWithProducts);

  if (lines.length === 0) {
    return (
      <section className="cart cart--empty">
        <h1>Giỏ hàng</h1>
        <p>Chưa có sản phẩm nào trong giỏ.</p>
        <Link to="/" className="btn btn--primary">
          Xem sản phẩm
        </Link>
      </section>
    );
  }

  return (
    <section className="cart">
      <div className="cart__head">
        <h1>Giỏ hàng</h1>
        <button type="button" className="link-btn" onClick={clear}>
          Xoá tất cả
        </button>
      </div>

      {isVerified && validated.removed > 0 ? (
        <p className="banner banner--warn">
          {validated.removed} sản phẩm không còn trong danh mục và đã được gỡ khỏi giỏ.
        </p>
      ) : null}
      {isVerified && validated.unreachable > 0 ? (
        <p className="banner banner--warn">
          Không kiểm tra được {validated.unreachable} sản phẩm với máy chủ — giá và số
          lượng có thể là dữ liệu cũ.
        </p>
      ) : null}
      {!isVerified ? (
        <p className="banner">Đang đối chiếu giá và tồn kho…</p>
      ) : null}

      <ul className="cart__lines">
        {lines.map((line) => {
          const stock = Number(line.stock);
          const max = Number.isFinite(stock) && stock > 0 ? stock : null;
          const atMax = max !== null && line.quantity >= max;

          return (
            <li key={line.productId} className="cart-line">
              <Link to={`/product/${line.productId}`} className="cart-line__media">
                <ProductImage src={line.imageUrl} alt={line.name} ratio="1 / 1" />
              </Link>

              <div className="cart-line__info">
                <Link to={`/product/${line.productId}`} className="cart-line__name">
                  {line.name}
                </Link>
                <span className="cart-line__unit">
                  Đơn giá: {formatPrice(line.price)}
                </span>
                {max === null ? (
                  <span className="cart-line__hint">
                    Chưa xác định được tồn kho với máy chủ
                  </span>
                ) : atMax ? (
                  <span className="cart-line__hint">
                    Đã đạt tối đa {max} trong kho
                  </span>
                ) : null}
              </div>

              <div
                className="stepper stepper--sm"
                role="group"
                aria-label={`Số lượng ${line.name}`}
              >
                <button
                  type="button"
                  onClick={() => setQuantity(line.productId, line.quantity - 1)}
                  disabled={line.quantity <= 1}
                  aria-label="Giảm số lượng"
                >
                  −
                </button>
                <input
                  type="number"
                  min="1"
                  max={max ?? undefined}
                  value={line.quantity}
                  onChange={(event) => {
                    // "" and NaN are transient edit states, never "remove this line".
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) return;
                    setQuantity(line.productId, Math.max(1, Math.floor(parsed)));
                  }}
                  aria-label="Số lượng"
                />
                <button
                  type="button"
                  onClick={() => setQuantity(line.productId, line.quantity + 1)}
                  disabled={atMax}
                  aria-label="Tăng số lượng"
                >
                  +
                </button>
              </div>

              <Price value={line.price * line.quantity} className="cart-line__total" />

              <button
                type="button"
                className="cart-line__remove"
                onClick={() => remove(line.productId)}
                aria-label={`Gỡ ${line.name} khỏi giỏ`}
              >
                ×
              </button>
            </li>
          );
        })}
      </ul>

      <aside className="summary">
        <p className="summary__row">
          <span>Tổng phụ ({count} sản phẩm)</span>
          <strong>{formatPrice(subtotal)}</strong>
        </p>
        <p className="summary__note">
          Phí vận chuyển và thuế chưa được tính. Server sẽ chốt lại giá và tồn kho
          một lần nữa khi nhận đơn.
        </p>
        <button
          type="button"
          className="btn btn--primary"
          disabled={!isVerified}
          onClick={() => navigate("/checkout")}
        >
          Thanh toán
        </button>
        {!isVerified ? (
          <p className="summary__pending">
            Chưa đối chiếu được với máy chủ nên chưa mở được thanh toán.
          </p>
        ) : null}
        {isVerified && !isLoggedIn() ? (
          <p className="summary__pending">
            Cần{" "}
            <Link to="/login" state={{ from: "/checkout" }}>
              đăng nhập
            </Link>{" "}
            để đặt hàng — bạn sẽ không phải gõ lại thông tin người nhận nữa.
          </p>
        ) : null}
        <Link to="/" className="btn btn--ghost">
          Tiếp tục mua
        </Link>
      </aside>

    </section>
  );
}
