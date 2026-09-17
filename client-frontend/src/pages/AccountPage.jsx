import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useUser } from "../hooks/useUser";
import { useRecipients } from "../hooks/useRecipients";
import {
  addRecipient,
  deleteRecipient,
  setDefaultRecipient,
  updateRecipient,
} from "../api/userApi";
import { RECIPIENT_LIMITS, validateRecipient } from "../utils/orderPayload";
import Field from "../components/Field";
import OrderHistory from "../components/OrderHistory";

const EMPTY_FORM = { name: "", phone: "", address: "", label: "", isDefault: false };

/**
 * Trang tài khoản: một menu, mỗi mục một màn.
 *
 * Trước đây "Sổ người nhận" và "Đơn hàng của tôi" cùng nằm một trang nên muốn xem
 * đơn phải cuộn qua hết danh sách người nhận và form thêm người nhận. Giờ mỗi mục
 * là một mục riêng trong .account__menu, chọn qua ?tab=... trên URL.
 *
 * TABS là nguồn duy nhất: thêm mục mới chỉ cần thêm một dòng ở đây và một nhánh
 * render bên dưới. Đường dẫn ?tab=<lạ> rơi về mục đầu tiên thay vì màn trắng.
 */
const TABS = [
  { key: "recipients", label: "Sổ người nhận" },
  { key: "orders", label: "Đơn hàng của tôi" },
];

const DEFAULT_TAB = TABS[0].key;

function readTab(value) {
  return TABS.some((item) => item.key === value) ? value : DEFAULT_TAB;
}

export default function AccountPage() {
  const { user } = useUser();
  const [searchParams] = useSearchParams();
  const tab = readTab(searchParams.get("tab"));
  const { items, status, error, reload, setItems } = useRecipients();

  const [form, setForm] = useState(EMPTY_FORM);
  const [editingId, setEditingId] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  function update(event) {
    const { name, value, type, checked } = event.target;
    const next = type === "checkbox" ? checked : value;
    setForm((previous) => ({ ...previous, [name]: next }));
    setFieldErrors((previous) => {
      if (!previous[name]) return previous;
      const copy = { ...previous };
      delete copy[name];
      return copy;
    });
  }

  function startEdit(recipient) {
    setEditingId(recipient.id);
    setMessage("");
    setFieldErrors({});
    setForm({
      name: recipient.name,
      phone: recipient.phone,
      address: recipient.address,
      label: recipient.label ?? "",
      isDefault: !!recipient.isDefault,
    });
  }

  function stopEdit() {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setFieldErrors({});
    setMessage("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (saving) return;

    const errors = validateRecipient(form);
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setMessage("");
      return;
    }

    setFieldErrors({});
    setMessage("");
    setSaving(true);

    const body = {
      name: form.name.trim(),
      phone: form.phone.trim(),
      address: form.address.trim(),
      label: form.label.trim(),
      isDefault: form.isDefault,
    };

    try {
      const saved = editingId
        ? await updateRecipient(editingId, body)
        : await addRecipient(body);
      const wasEditing = editingId !== null;
      stopEdit();
      setMessage(
        wasEditing ? "Đã cập nhật người nhận." : `Đã lưu người nhận “${saved.name}”.`
      );
      // Tai lai toan bo danh sach (co mac dinh doi o nhieu dong), nhưng loi tai
      // lai KHONG duoc phep bien mot lan luu thanh cong thanh tin that bai.
      try {
        await reload();
      } catch {
        setMessage("Đã lưu, nhưng chưa tải lại được danh sách — thử tải lại trang.");
      }
    } catch (caught) {
      setMessage(caught.message || "Không lưu được người nhận.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(recipient) {
    if (!window.confirm(`Xoá người nhận “${recipient.name}” khỏi sổ?`)) return;

    setMessage("");
    try {
      await deleteRecipient(recipient.id);
      if (editingId === recipient.id) stopEdit();
    } catch (caught) {
      setMessage(caught.message || "Không xoá được người nhận.");
      return;
    }

    try {
      await reload();
    } catch {
      setMessage("Đã xoá, nhưng chưa tải lại được danh sách — thử tải lại trang.");
    }
  }

  async function handleMakeDefault(recipient) {
    setMessage("");
    try {
      const next = await setDefaultRecipient(recipient.id);
      if (Array.isArray(next)) setItems(next);
    } catch (caught) {
      setMessage(caught.message || "Không đổi được người nhận mặc định.");
    }
  }

  if (!user) {
    return (
      <section className="account account--signed-out">
        <h1>Tài khoản</h1>
        <p>Bạn chưa đăng nhập.</p>
        <Link to="/login" className="btn btn--primary">
          Đăng nhập
        </Link>
      </section>
    );
  }

  return (
    <section className="account">
      <div className="account__head">
        {/* Không đặt nút Đăng xuất ở đây nữa: header đã có menu tài khoản, và hai
            chỗ cùng một hành động ngay cạnh nhau chỉ khiến khách đoán nhầm là hai
            việc khác nhau. */}
        <div>
          <h1>Xin chào {user.fullName || user.username}</h1>
          <p className="account__meta">
            <span>{user.username}</span>
            <span>·</span>
            <span>{user.email}</span>
          </p>
        </div>
      </div>

      <nav className="account__menu" aria-label="Các mục của tài khoản">
        {TABS.map((item) => (
          <Link
            key={item.key}
            to={`/account?tab=${item.key}`}
            className={"account__menu-item" + (tab === item.key ? " is-active" : "")}
            aria-current={tab === item.key ? "page" : undefined}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      {/* Mục không được chọn bị bỏ hẳn khỏi DOM thay vì chỉ display:none: form đang
          gõ dở mà bị ẩn thì khách quay lại thấy chữ còn nguyên nhưng con trỏ đã mất,
          và các ô nhập vẫn nằm trong thứ tự tab. Đổi mục sẽ mất bản nháp — chấp nhận,
          vì đây là hai việc khác nhau chứ không phải hai khối của cùng một biểu mẫu. */}
      {tab === "recipients" ? (
      <div className="account__grid account__panel">
        <div className="account__book">
          <h2>Sổ người nhận hàng</h2>
          <p className="field__hint">
            Chỉ dùng để điền sẵn. Đơn hàng luôn lưu bản sao riêng, nên sửa hoặc xoá ở
            đây không làm đổi đơn đã đặt.
          </p>

          {status === "loading" ? <p className="banner">Đang tải sổ người nhận…</p> : null}
          {status === "error" ? (
            <div className="banner banner--danger" role="alert">
              {error}{" "}
              <button type="button" className="link-btn" onClick={reload}>
                Thử lại
              </button>
            </div>
          ) : null}
          {status === "ready" && items.length === 0 ? (
            <p className="banner">
              Chưa có người nhận nào. Thêm một người ở khung bên cạnh để lần sau khỏi
              gõ lại.
            </p>
          ) : null}

          <ul className="recipient-list">
            {items.map((recipient) => (
              <li key={recipient.id} className="recipient">
                <div className="recipient__body">
                  <p className="recipient__title">
                    <strong>{recipient.name}</strong>
                    {recipient.label ? <span className="tag">{recipient.label}</span> : null}
                    {recipient.isDefault ? (
                      <span className="tag tag--ok">Mặc định</span>
                    ) : null}
                  </p>
                  <p className="recipient__line">{recipient.phone}</p>
                  <p className="recipient__line">{recipient.address}</p>
                </div>
                <div className="recipient__actions">
                  <button
                    type="button"
                    className="link-btn"
                    onClick={() => startEdit(recipient)}
                  >
                    Sửa
                  </button>
                  <button
                    type="button"
                    className="link-btn"
                    onClick={() => handleMakeDefault(recipient)}
                    disabled={recipient.isDefault}
                  >
                    Đặt làm mặc định
                  </button>
                  <button
                    type="button"
                    className="link-btn link-btn--danger"
                    onClick={() => handleDelete(recipient)}
                  >
                    Xoá
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <form className="account__form" onSubmit={handleSubmit} noValidate>
          <h2>{editingId ? "Sửa người nhận" : "Thêm người nhận"}</h2>

          <Field
            id="name"
            name="name"
            label="Tên người nhận"
            value={form.name}
            onChange={update}
            error={fieldErrors.name}
            maxLength={RECIPIENT_LIMITS.name}
            autoComplete="name"
            required
          />
          <Field
            id="recipient-phone"
            name="phone"
            label="Số điện thoại"
            value={form.phone}
            onChange={update}
            error={fieldErrors.phone}
            hint="Ví dụ 0901234567 hoặc +84901234567"
            maxLength={RECIPIENT_LIMITS.phone}
            inputMode="tel"
            autoComplete="tel"
            required
          />
          <label className="field" htmlFor="recipient-address">
            <span className="field__label">Địa chỉ nhận hàng</span>
            <textarea
              id="recipient-address"
              name="address"
              className="field__input"
              rows="3"
              value={form.address}
              onChange={update}
              maxLength={RECIPIENT_LIMITS.address}
              placeholder="Số nhà, đường, phường/xã, quận/huyện, tỉnh/thành"
              aria-invalid={fieldErrors.address ? "true" : undefined}
              required
            />
            {fieldErrors.address ? (
              <span className="field__error">{fieldErrors.address}</span>
            ) : null}
          </label>
          <Field
            id="label"
            name="label"
            label="Nhãn (không bắt buộc)"
            value={form.label}
            onChange={update}
            error={fieldErrors.label}
            hint="Ví dụ: Nhà, Cơ quan, Quà tặng"
            maxLength={30}
          />

          <label className="check">
            <input
              type="checkbox"
              name="isDefault"
              checked={form.isDefault}
              onChange={update}
            />
            <span>Dùng làm người nhận mặc định</span>
          </label>

          {message ? (
            <div className="banner" role="status">
              {message}
            </div>
          ) : null}

          <div className="account__form-actions">
            <button type="submit" className="btn btn--primary" disabled={saving}>
              {saving ? "Đang lưu…" : editingId ? "Cập nhật" : "Thêm vào sổ"}
            </button>
            {editingId ? (
              <button type="button" className="btn btn--ghost" onClick={stopEdit}>
                Huỷ sửa
              </button>
            ) : null}
          </div>
        </form>
      </div>
      ) : null}

      {tab === "orders" ? <OrderHistory /> : null}
    </section>
  );
}
