import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { loginAdmin } from "../api/adminApi";
import AuthLayout from "../components/AuthLayout";

function LoginPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [form, setForm] = useState({account: "",password: "",});
  const [message, setMessage] = useState(location.state?.message ?? "");
  const [messageType, setMessageType] = useState(location.state?.message ? "success" : "");
  const [loading, setLoading] = useState(false);
  

  const handleChange = (event) => {const { name, value } = event.target;
    setForm((previous) => ({...previous,[name]: value,}));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage("");
    setLoading(true);

    try {
      await loginAdmin(form);

      localStorage.setItem("adminLoggedIn", "true");

      navigate("/admin", {
        replace: true,
      });
    } catch (error) {
      setMessage(error.message || "Không thể kết nối đến máy chủ");
      setMessageType("error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout>
      <form onSubmit={handleSubmit}>
        <div className="form-heading">
          <h2>Chào mừng trở lại</h2>
          <p>Đăng nhập để tiếp tục vào trang quản trị.</p>
        </div>

        <div className="form-group">
          <label htmlFor="account">Username hoặc email</label>
          <input
            id="account"
            name="account"
            type="text"
            value={form.account}
            onChange={handleChange}
            placeholder="Nhập username hoặc email"
            autoComplete="username"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="login-password">Mật khẩu</label>
          <input
            id="login-password"
            name="password"
            type="password"
            value={form.password}
            onChange={handleChange}
            placeholder="Nhập mật khẩu"
            autoComplete="current-password"
            required
          />
        </div>

        <button
          className="submit-button"
          type="submit"
          disabled={loading}
        >
          {loading ? "Đang đăng nhập..." : "Đăng nhập"}
        </button>

        {message && (
          <div className={`message ${messageType}`}>{message}</div>
        )}
      </form>
    </AuthLayout>
  );
}

export default LoginPage;
