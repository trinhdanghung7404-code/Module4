import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { loginUser } from "../api/userApi";
import { useUser } from "../hooks/useUser";
import AuthShell from "../components/AuthShell";

/**
 * Chỉ cho quay lại những đường dẫn nội bộ. state.from đến từ URL do người dùng
 * điều khiển được, để "//evil.com" lọt vào đây thì navigate sẽ đưa khách ra
 * khỏi app theo cách không ai chủ đích.
 */
function safeRedirect(from) {
  return typeof from === "string" && from.startsWith("/") && !from.startsWith("//")
    ? from
    : "/";
}

export default function LoginPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { login } = useUser();

  const [form, setForm] = useState({ account: "", password: "" });
  const [fieldErrors, setFieldErrors] = useState({});
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const notice = location.state?.message ?? "";
  const from = safeRedirect(location.state?.from);

  function update(event) {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
    setFieldErrors((previous) => {
      if (!previous[name]) return previous;
      const next = { ...previous };
      delete next[name];
      return next;
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (loading) return;

    const errors = {};
    if (!form.account.trim()) errors.account = "Nhập username hoặc email";
    if (!form.password) errors.password = "Nhập mật khẩu";
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return;
    }

    setFieldErrors({});
    setMessage("");
    setLoading(true);

    try {
      // Server tra ve { token, user }: khach vua o dang dung truoc gio hang, bat
      // nhap lai mat khau lan nua la mat don.
      const session = await loginUser(form);
      login(session);
      // replace: quay lui từ trang thanh toán không được đưa khách về lại form
      // đăng nhập vừa dùng xong.
      navigate(from, { replace: true });
    } catch (error) {
      setMessage(error.message || "Không đăng nhập được.");
      setLoading(false);
    }
  }

  return (
    <AuthShell
      title="Đăng nhập"
      lead="Đăng nhập để đặt hàng và giữ sẵn thông tin người nhận cho những lần sau."
      footer={
        <>
          Chưa có tài khoản? <Link to="/register">Đăng ký</Link>
        </>
      }
    >
      {notice ? <p className="banner">{notice}</p> : null}

      <form className="auth__form" onSubmit={handleSubmit} noValidate>
        <label className="field" htmlFor="account">
          <span className="field__label">Username hoặc email</span>
          <input
            id="account"
            name="account"
            className="field__input"
            value={form.account}
            onChange={update}
            autoComplete="username"
            aria-invalid={fieldErrors.account ? "true" : undefined}
            required
          />
          {fieldErrors.account ? (
            <span className="field__error">{fieldErrors.account}</span>
          ) : null}
        </label>

        <label className="field" htmlFor="password">
          <span className="field__label">Mật khẩu</span>
          <input
            id="password"
            name="password"
            type="password"
            className="field__input"
            value={form.password}
            onChange={update}
            autoComplete="current-password"
            aria-invalid={fieldErrors.password ? "true" : undefined}
            required
          />
          {fieldErrors.password ? (
            <span className="field__error">{fieldErrors.password}</span>
          ) : null}
        </label>

        {message ? (
          <div className="banner banner--danger" role="alert">
            {message}
          </div>
        ) : null}

        <button type="submit" className="btn btn--primary" disabled={loading}>
          {loading ? "Đang đăng nhập…" : "Đăng nhập"}
        </button>
      </form>
    </AuthShell>
  );
}
