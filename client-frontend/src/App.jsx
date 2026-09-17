import { Route, Routes, useParams } from "react-router-dom";
import ShopLayout from "./components/ShopLayout";
import ShopPage from "./pages/ShopPage";
import ProductDetailPage from "./pages/ProductDetailPage";
import CartPage from "./pages/CartPage";
import CheckoutPage from "./pages/CheckoutPage";
import OrderSuccessPage from "./pages/OrderSuccessPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import AccountPage from "./pages/AccountPage";
import RequireUser from "./components/RequireUser";
import NotFoundPage from "./pages/NotFoundPage";
import "./styles/layout.css";
import "./styles/pages.css";
import "./styles/cart.css";
import "./styles/account.css";
import "./styles/orders.css";

function ProductDetailRoute() {
  const { id } = useParams();

  // key={id} gives every product its own mount, so the selected gallery angle and the
  // quantity stepper of a previously viewed product cannot leak into the next one.
  return <ProductDetailPage key={id} />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<ShopLayout />}>
        <Route index element={<ShopPage />} />
        <Route path="product/:id" element={<ProductDetailRoute />} />
        <Route path="cart" element={<CartPage />} />
        <Route path="login" element={<LoginPage />} />
        <Route path="register" element={<RegisterPage />} />
        {/* Thanh toán và sổ người nhận cần có tài khoản. RequireUser giữ lại
            đường dẫn khách đang muốn vào, nên đăng ký xong thì quay về đúng
            chỗ thay vì bị ném về trang chủ mất giỏ hàng. */}
        <Route element={<RequireUser />}>
          <Route path="checkout" element={<CheckoutPage />} />
          <Route path="account" element={<AccountPage />} />
        </Route>
        <Route path="order/success" element={<OrderSuccessPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
