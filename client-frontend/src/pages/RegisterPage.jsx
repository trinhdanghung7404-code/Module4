import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { registerUser } from "../api/userApi";
import { useUser } from "../hooks/useUser";
import { validateAccount } from "../utils/orderPayload";
import Field from "../components/Field";
import AuthShell from "../components/AuthShell";

const EMPTY = {
  username: "",
  fullName: "",
  email: "",
  password: "",
  confirmPassword: "",
};

function safeRedirect(from) {
  return typeof from === "string" && from.startsWith("/") && !from.startsWith("//")
    ? from
    : "/";
}

export default function RegisterPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { login } = useUser();

  const [form, setForm] = useState(EMPTY);
  const [fieldErrors, setFieldErrors] = useState({});
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

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

    const errors = validateAccount(form);
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setMessage("");
      return;
    }

    setFieldErrors({});
    setMessage("");
    setLoading(true);

    try {
      // Server tra ve { token, user } sau khi dang ky: khach vua o dang dung
      // truoc gio hang, bat nhap lai mat khau lan nua la mat don.
      const session = await registerUser(form);
      login(session);
      navigate(from, { replace: true });
    } catch (error) {
      setMessage(error.message || "Không đăng ký được.");
      setLoading(false);
    }
  }

  return (
    <AuthShell
      title="Tạo tài khoản"
      lead="Một lần điền thông tin, những lần sau chỉ chọn người nhận là đặt được hàng."
      footer={
        <>
          Đã có tài khoản? <Link to="/login">Đăng nhập</Link>
        </>
      }
    >
      <form className="auth__form" onSubmit={handleSubmit} noValidate>
        <Field
          id="username"
          name="username"
          label="Username"
          value={form.username}
          onChange={update}
          error={fieldErrors.username}
          hint="4-50 ký tự, không dấu cách"
          maxLength={50}
          autoComplete="username"
          required
        />
        <Field
          id="fullName"
          name="fullName"
          label="Họ tên của bạn"
          value={form.fullName}
          onChange={update}
          error={fieldErrors.fullName}
          hint="Dùng để gợi ý sẵn khi tạo người nhận đầu tiên"
          maxLength={100}
          autoComplete="name"
          required
        />
        <Field
          id="email"
          name="email"
          type="email"
          label="Email"
          value={form.email}
          onChange={update}
          error={fieldErrors.email}
          maxLength={100}
          autoComplete="email"
          required
        />
        <Field
          id="password"
          name="password"
          type="password"
          label="Mật khẩu"
          value={form.password}
          onChange={update}
          error={fieldErrors.password}
          hint="Tối thiểu 8 ký tự"
          autoComplete="new-password"
          required
        />
        <Field
          id="confirmPassword"
          name="confirmPassword"
          type="password"
          label="Xác nhận mật khẩu"
          value={form.confirmPassword}
          onChange={update}
          error={fieldErrors.confirmPassword}
          autoComplete="new-password"
          required
        />

        {message ? (
          <div className="banner banner--danger" role="alert">
            {message}
          </div>
        ) : null}

        <button type="submit" className="btn btn--primary" disabled={loading}>
          {loading ? "Đang tạo tài khoản…" : "Đăng ký"}
        </button>
      </form>
    </AuthShell>
  );
}
