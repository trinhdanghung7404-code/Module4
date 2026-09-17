import { Outlet } from "react-router-dom";
import ShopHeader from "./ShopHeader";
import ShopFooter from "./ShopFooter";

export default function ShopLayout() {
  return (
    <div className="shop-shell">
      <ShopHeader />
      <main className="shop-main">
        <Outlet />
      </main>
      <ShopFooter />
    </div>
  );
}

