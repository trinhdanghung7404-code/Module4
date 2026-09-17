import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <section className="not-found">
      <h1>Không tìm thấy trang</h1>
      <p>Đường dẫn bạn truy cập không tồn tại hoặc đã bị gỡ.</p>
      <Link to="/" className="btn btn--primary">
        Về cửa hàng
      </Link>
    </section>
  );
}
