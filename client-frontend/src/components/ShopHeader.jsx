import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useCategories, useShopUi } from "../hooks/useShopUi";
import { useCart } from "../hooks/useCart";
import ShopAccountMenu from "./ShopAccountMenu";

export default function ShopHeader() {
  const { filter, setFilter } = useShopUi();
  const { categories, error, reload } = useCategories();
  const { count } = useCart();
  const location = useLocation();
  const navigate = useNavigate();

  // The input keeps its own draft tagged with the committed query it was typed
  // against. Reading it that way lets an external change (clearing the search chip)
  // win automatically, with no effect syncing two states.
  const [draft, setDraft] = useState({
    for: filter.search,
    value: filter.search,
  });
  const term = draft.for === filter.search ? draft.value : filter.search;

  function applyFilter(patch) {
    setFilter(patch);
    if (location.pathname !== "/") navigate("/");
  }

  function handleSearch(event) {
    event.preventDefault();
    applyFilter({ search: term.trim() });
  }

  return (
    <header className="shop-header">
      <div className="shop-header__bar">
        <Link to="/" className="shop-brand">
          <span className="shop-brand__mark" aria-hidden="true">
            ▣
          </span>
          <span>Cửa hàng</span>
        </Link>

        <form className="shop-search" role="search" onSubmit={handleSearch}>
          <input
            type="search"
            value={term}
            onChange={(event) =>
              setDraft({ for: filter.search, value: event.target.value })
            }
            placeholder="Tìm sản phẩm…"
            aria-label="Tìm sản phẩm"
          />
          <button type="submit" className="btn btn--primary">
            Tìm
          </button>
        </form>

        <Link to="/cart" className="shop-cart">
          Giỏ hàng
          {count > 0 ? <span className="shop-cart__badge">{count}</span> : null}
        </Link>

        <ShopAccountMenu />
      </div>

      <nav className="shop-cats" aria-label="Danh mục sản phẩm">
        <button
          type="button"
          className={`chip${filter.categoryId === 0 ? " is-active" : ""}`}
          onClick={() => applyFilter({ categoryId: 0 })}
        >
          Tất cả
        </button>

        {error ? (
          <span className="shop-cats__error">
            {error}
            <button type="button" className="link-btn" onClick={reload}>
              Thử lại
            </button>
          </span>
        ) : (
          categories.map((category) => (
            <button
              key={category.id}
              type="button"
              className={`chip${
                filter.categoryId === category.id ? " is-active" : ""
              }`}
              onClick={() =>
                applyFilter({
                  categoryId:
                    filter.categoryId === category.id ? 0 : category.id,
                })
              }
            >
              {category.name}
              <span className="chip__count">{category.productCount}</span>
            </button>
          ))
        )}

        {filter.search ? (
          <button
            type="button"
            className="chip chip--clear"
            onClick={() => applyFilter({ search: "" })}
          >
            × “{filter.search}”
          </button>
        ) : null}
      </nav>
    </header>
  );
}
