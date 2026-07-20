import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { registerAdmin } from "../api/adminApi";
import AuthLayout from "../components/AuthLayout";

const INITIAL_FORM = {
  fullName: "",
  username: "",
  email: "",
  password: "",
  confirmPassword: "",
};

function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState(INITIAL_FORM);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage("");

    if (form.password !== form.confirmPassword) {
      setMessage("Mật khẩu xác nhận không khớp");
      return;
    }

    setLoading(true);

    try {
      await registerAdmin(form);
      setForm(INITIAL_FORM);
      navigate("/login", {
        replace: true,
        state: {
          message: "Đăng ký thành công. Hãy đăng nhập.",
        },
      });
    } catch (error) {
      setMessage(error.message || "Không thể kết nối đến máy chủ");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout>
      <form onSubmit={handleSubmit}>
        <div className="form-heading">
          <h2>Tạo tài khoản Admin</h2>
          <p>Điền đầy đủ thông tin để đăng ký tài khoản.</p>
        </div>

        <div className="form-group">
          <label htmlFor="fullName">Họ và tên</label>
          <input
            id="fullName"
            name="fullName"
            type="text"
            value={form.fullName}
            onChange={handleChange}
            placeholder="Nguyễn Văn A"
            minLength={2}
            maxLength={100}
            required
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              name="username"
              type="text"
              value={form.username}
              onChange={handleChange}
              placeholder="admin01"
              autoComplete="username"
              minLength={4}
              maxLength={50}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              name="email"
              type="email"
              value={form.email}
              onChange={handleChange}
              placeholder="admin@email.com"
              autoComplete="email"
              maxLength={100}
              required
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="register-password">Mật khẩu</label>
            <input
              id="register-password"
              name="password"
              type="password"
              value={form.password}
              onChange={handleChange}
              placeholder="Tối thiểu 8 ký tự"
              autoComplete="new-password"
              minLength={8}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">Xác nhận mật khẩu</label>
            <input
              id="confirmPassword"
              name="confirmPassword"
              type="password"
              value={form.confirmPassword}
              onChange={handleChange}
              placeholder="Nhập lại mật khẩu"
              autoComplete="new-password"
              required
            />
          </div>
        </div>

        <button
          className="submit-button"
          type="submit"
          disabled={loading}
        >
          {loading ? "Đang đăng ký..." : "Đăng ký"}
        </button>

        {message && <div className="message error">{message}</div>}
      </form>
    </AuthLayout>
  );
}

export default RegisterPage;
