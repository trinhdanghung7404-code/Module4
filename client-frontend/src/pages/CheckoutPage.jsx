import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useCart } from "../hooks/useCart";
import { useCartVerification } from "../hooks/useCartVerification";
import { useUser } from "../hooks/useUser";
import { defaultRecipient, useRecipients } from "../hooks/useRecipients";
import { createOrder } from "../api/shopApi";
import { addRecipient } from "../api/userApi";
import {
  RECIPIENT_LIMITS,
  buildOrderPayload,
  saveOrderReceipt,
  validateRecipient,
} from "../utils/orderPayload";
import Field, { Hint } from "../components/Field";
import Price from "../components/Price";
import ProductImage from "../components/ProductImage";
import { formatPrice } from "../utils/format";

/**
 * Client dung hinh thai { name, phone, address }; buildOrderPayload doi `name`
 * thanh `customerName` khi gui. Khong phai hai hinh thai lech nhau im lom.
 */
function samePerson(form, recipient) {
  if (!recipient) return false;
  return (
    form.name.trim() === recipient.name &&
    form.phone.trim() === recipient.phone &&
    form.address.trim() === recipient.address
  );
}

export default function CheckoutPage() {
  const { lines, subtotal, count, clear, syncWithProducts } = useCart();
  const { validated, isVerified } = useCartVerification(lines, syncWithProducts);
  const { user, isLoggedIn } = useUser();
  const { items: saved } = useRecipients();
  const navigate = useNavigate();

  // null = chura chon gi ca -> tu dong dung nguoi nhan mac dinh. "manual" = khach
  // muon nhap tay. Con lai la id trong so nguoi nhan.
  const [pickedId, setPickedId] = useState(null);
  // Chi luu nhung o khach thuc su go. Khi chuyen sang nguoi nhan khac thi xoa ve
  // null, de not le cua lan go truoc khong din vao nguoi moi chon.
  const [typed, setTyped] = useState(null);
  // Ghi chú là của RIÊNG đơn này, không phải thuộc tính người nhận, nên đứng ngoài.
  const [note, setNote] = useState("");
  const [saveToBook, setSaveToBook] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({});
  const [formError, setFormError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const manual = pickedId === "manual";
  const chosen = manual
    ? null
    : pickedId === null
      ? defaultRecipient(saved)
      : (saved.find((recipient) => recipient.id === pickedId) ?? defaultRecipient(saved));
  const activeId = manual ? "manual" : (chosen?.id ?? null);

  /**
   * Form duoc TINH ra tu nguoi nhan dang chon + nhung o khach go, khong phai mot
   * state phai dong bo bang useEffect. Cach cu phai set state trong effect moi
   * lan so nguoi nhan ve kip — React 19 bao cascading render, va khach co the
   * thay thong tin cua nguoi cu lech sang nguoi moi.
   */
  const form = {
    name: typed?.name ?? chosen?.name ?? "",
    phone: typed?.phone ?? chosen?.phone ?? "",
    address: typed?.address ?? chosen?.address ?? "",
    note,
  };

  const modified = !manual && typed !== null && !samePerson(form, chosen);
  const worthSaving =
    isLoggedIn &&
    (manual || modified) &&
    Boolean(form.name.trim() && form.phone.trim() && form.address.trim());

  function updateField(field) {
    return (event) => {
      const value = event.target.value;
      if (field === "note") {
        setNote(value);
      } else {
        setTyped((previous) => ({
          ...(previous ?? { name: form.name, phone: form.phone, address: form.address }),
          [field]: value,
        }));
      }
      // Xoá lỗi của chính ô vừa bắt đầu sửa, không đợi bấm đặt hàng lại lần nữa.
      setFieldErrors((previous) => {
        if (!previous[field]) return previous;
        const next = { ...previous };
        delete next[field];
        return next;
      });
    };
  }

  function chooseRecipient(recipient) {
    setPickedId(recipient.id);
    setTyped(null);
    setSaveToBook(false);
    setFieldErrors({});
  }

  function chooseManual() {
    setPickedId("manual");
    setTyped({ name: "", phone: "", address: "" });
    setSaveToBook(false);
    setFieldErrors({});
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitting) return;

    const errors = validateRecipient(form);
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setFormError("Vui lòng kiểm tra lại thông tin người nhận.");
      return;
    }

    setFieldErrors({});
    setFormError("");
    setSubmitting(true);

    try {
      const order = await createOrder(buildOrderPayload(lines, form));

      // Luu nguoi nhan SAU khi don da thanh cong va loi o buoc nay bi an: so
      // dia chi tien loi thi don van phai duoc tao, khong duoc de khach tuong
      // don that bai ma di dat lai lan nua.
      if (saveToBook && worthSaving) {
        try {
          await addRecipient({
            name: form.name.trim(),
            phone: form.phone.trim(),
            address: form.address.trim(),
            label: "",
            isDefault: false,
          });
        } catch {
          // Bo qua co chu dich: xem "Đặt hàng" la thanh cong.
        }
      }

      saveOrderReceipt(order);
      clear();
      navigate("/order/success");
    } catch (error) {
      // Giỏ hàng được GIỮ NGUYÊN có chủ đích: đơn chưa được tạo (server rollback cả
      // đơn nếu một sản phẩm không đủ kho), nên khách còn dữ liệu để sửa rồi đặt lại.
      // Tách theo dòng vì lỗi validation được trả về theo từng field.
      setFormError(error.message || "Không tạo được đơn hàng. Vui lòng thử lại.");
      setSubmitting(false);
    }
  }

  if (lines.length === 0) {
    return (
      <section className="checkout checkout--empty">
        <h1>Thanh toán</h1>
        <p>Giỏ hàng đang trống nên không có gì để đặt.</p>
        <Link to="/" className="btn btn--primary">
          Xem sản phẩm
        </Link>
      </section>
    );
  }

  return (
    <section className="checkout">
      <div className="cart__head">
        <h1>Thanh toán</h1>
        <Link to="/cart" className="link-btn">
          ← Sửa giỏ hàng
        </Link>
      </div>

      {!isVerified ? (
        <p className="banner">Đang đối chiếu giá và tồn kho với máy chủ…</p>
      ) : null}

      {isVerified && validated.unreachable > 0 ? (
        <p className="banner banner--warn">
          Không kiểm tra được {validated.unreachable} sản phẩm — server sẽ chốt lại
          số liệu khi nhận đơn.
        </p>
      ) : null}

      <form className="checkout__grid" onSubmit={handleSubmit} noValidate>
        <div className="checkout__fields">
          <section className="checkout__account">
            <h2>Tài khoản đặt hàng</h2>
            <p className="checkout__who">
              <strong>{user?.fullName || user?.username}</strong>
              <span>{user?.email}</span>
            </p>
            <p className="field__hint">
              Đơn được ghi dưới tên tài khoản này, còn người nhận ở bên dưới có thể là
              một người khác. <Link to="/account">Quản lý sổ người nhận</Link>
            </p>
          </section>

          <section className="checkout__recipient">
            <h2>Người nhận hàng</h2>

            {saved.length > 0 ? (
              <div
                className="recipient-picker"
                role="radiogroup"
                aria-label="Chọn người nhận đã lưu"
              >
                {saved.map((recipient) => (
                  <button
                    key={recipient.id}
                    type="button"
                    role="radio"
                    aria-checked={activeId === recipient.id}
                    className={`recipient-choice${
                      activeId === recipient.id ? " is-active" : ""
                    }`}
                    onClick={() => chooseRecipient(recipient)}
                  >
                    <span className="recipient-choice__name">
                      {recipient.name}
                      {recipient.label ? ` · ${recipient.label}` : ""}
                      {recipient.isDefault ? " · mặc định" : ""}
                    </span>
                    <span className="recipient-choice__meta">
                      {recipient.phone} — {recipient.address}
                    </span>
                  </button>
                ))}
                <button
                  type="button"
                  role="radio"
                  aria-checked={activeId === "manual"}
                  className={`recipient-choice recipient-choice--new${
                    activeId === "manual" ? " is-active" : ""
                  }`}
                  onClick={chooseManual}
                >
                  <span className="recipient-choice__name">＋ Người nhận khác</span>
                  <span className="recipient-choice__meta">Nhập tay một người mới</span>
                </button>
              </div>
            ) : (
              <p className="field__hint">
                Chưa có người nhận nào được lưu. Điền một lần rồi tích “Lưu vào sổ”,
                lần sau chỉ việc chọn.
              </p>
            )}

            <Field
              id="name"
              label="Tên người nhận"
              value={form.name}
              onChange={updateField("name")}
              error={fieldErrors.name}
              placeholder="Nguyễn Văn A"
              maxLength={RECIPIENT_LIMITS.name}
              autoComplete="name"
              required
            />
            <Field
              id="phone"
              label="Số điện thoại"
              value={form.phone}
              onChange={updateField("phone")}
              error={fieldErrors.phone}
              hint="Ví dụ 0901234567 hoặc +84901234567"
              placeholder="0901234567"
              maxLength={RECIPIENT_LIMITS.phone}
              inputMode="tel"
              autoComplete="tel"
              required
            />

            <label className="field" htmlFor="address">
              <span className="field__label">Địa chỉ nhận hàng</span>
              <textarea
                id="address"
                className="field__input"
                rows="3"
                value={form.address}
                onChange={updateField("address")}
                maxLength={RECIPIENT_LIMITS.address}
                placeholder="Số nhà, đường, phường/xã, quận/huyện, tỉnh/thành"
                aria-invalid={fieldErrors.address ? "true" : undefined}
                aria-describedby={fieldErrors.address ? "address-error" : undefined}
                required
              />
              {fieldErrors.address ? (
                <span className="field__error" id="address-error">
                  {fieldErrors.address}
                </span>
              ) : null}
            </label>

            {modified ? (
              <p className="field__hint">
                Bạn đang sửa thông tin khác với người nhận đã chọn — đơn sẽ ghi đúng
                nội dung đang hiển thị ở đây.
              </p>
            ) : null}

            {worthSaving ? (
              <label className="check">
                <input
                  type="checkbox"
                  checked={saveToBook}
                  onChange={(event) => setSaveToBook(event.target.checked)}
                />
                <span>Lưu người nhận này vào sổ để lần sau khỏi nhập lại</span>
              </label>
            ) : null}
          </section>

          <section className="checkout__note">
            <label className="field" htmlFor="note">
              <span className="field__label">Ghi chú cho đơn này</span>
              <textarea
                id="note"
                className="field__input"
                rows="3"
                value={form.note}
                onChange={updateField("note")}
                maxLength={RECIPIENT_LIMITS.note}
                placeholder="Ví dụ: gọi trước khi giao"
              />
              <Hint text="Không bắt buộc, và không lưu vào sổ người nhận" />
            </label>
          </section>
        </div>

        <aside className="checkout__aside">
          <h2>Sản phẩm ({count})</h2>

          <ul className="checkout__items">
            {lines.map((line) => (
              <li key={line.productId} className="checkout__item">
                <ProductImage src={line.imageUrl} alt={line.name} ratio="1 / 1" />
                <span className="checkout__item-info">
                  <span className="checkout__item-name">{line.name}</span>
                  <span className="cart-line__unit">
                    {line.quantity} × {formatPrice(line.price)}
                  </span>
                </span>
                <Price
                  value={line.price * line.quantity}
                  className="checkout__item-total"
                />
              </li>
            ))}
          </ul>

          <p className="summary__row">
            <span>Tạm tính</span>
            <strong>{formatPrice(subtotal)}</strong>
          </p>
          <p className="summary__note">
            Giá cuối cùng do server tính lại từ cơ sở dữ liệu, không lấy từ máy bạn.
          </p>

          {formError ? (
            <div className="banner banner--danger" role="alert">
              {formError.split("\n").map((row, index) => (
                <p key={`${index}-${row}`}>{row}</p>
              ))}
            </div>
          ) : null}

          <button
            type="submit"
            className="btn btn--primary"
            disabled={!isVerified || submitting}
          >
            {submitting ? "Đang gửi đơn…" : "Đặt hàng"}
          </button>
        </aside>
      </form>
    </section>
  );
}
