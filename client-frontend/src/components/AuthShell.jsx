import { Link } from "react-router-dom";

/**
 * Khung chung cho trang đăng nhập / đăng ký của shop.
 *
 * Trang quan tri co AuthLayout cua no (dieu huong khac, mau sac khac), nen shop
 * giu mot chiec shell rieng thay vi chung dung va phai qua lai lai.
 */
export default function AuthShell({ title, lead, children, footer }) {
  return (
    <section className="auth">
      <div className="auth__card">
        <div className="auth__head">
          <h1>{title}</h1>
          {lead ? <p>{lead}</p> : null}
        </div>
        {children}
        {footer ? <p className="auth__foot">{footer}</p> : null}
      </div>

      <p className="auth__back">
        <Link to="/">← Quay lại cửa hàng</Link>
      </p>
    </section>
  );
}
