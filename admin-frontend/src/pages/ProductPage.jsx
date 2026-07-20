import { useEffect, useState } from "react";
import { getProducts } from "../api/productApi";
import ProductModal from "../components/ProductModal";
import "../styles/product.css";

function ProductThumbnail({ product }) {
  const [failed, setFailed] = useState(false);

  const frontImage = product.images?.find(
    (image) => image.imageType === "FRONT"
  );

  if (!frontImage?.imageUrl || failed) {
    return (
      <div className="image-placeholder">
        Không có ảnh
      </div>
    );
  }

  return (
    <img
      className="product-image"
      src={frontImage.imageUrl}
      alt={product.name}
      onError={() => setFailed(true)}
    />
  );
}

function ViewIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function EditIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z" />
    </svg>
  );
}

function DeleteIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6 18 20H6L5 6" />
      <path d="M10 11v5" />
      <path d="M14 11v5" />
    </svg>
  );
}

function ProductPage() {
  const [products, setProducts] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("");

  useEffect(() => {
    getProducts()
      .then(setProducts)
      .catch((error) => {
        setMessage(error.message);
        setMessageType("error");
      });
  }, []);

  const handleCreated = (product) => {
    setProducts((previous) => [
      product,
      ...previous,
    ]);

    setMessage("Thêm sản phẩm thành công");
    setMessageType("success");
  };

  const handleView = (product) => {
    setMessage(`Đang chọn xem sản phẩm: ${product.name}`);
    setMessageType("success");
  };

  const handleEdit = (product) => {
    setMessage(`Đang chọn sửa sản phẩm: ${product.name}`);
    setMessageType("success");
  };

  const handleDelete = (product) => {
    setMessage(`Đang chọn xóa sản phẩm: ${product.name}`);
    setMessageType("error");
  };

  const formatPrice = (price) =>
    new Intl.NumberFormat("vi-VN", {
      style: "currency",
      currency: "VND",
    }).format(Number(price));

  return (
    <main className="product-page">
      <div className="product-container">
        <header className="product-page-header">
          <div>
            <h1>Danh sách sản phẩm</h1>
            <p>Quản lý sản phẩm trong hệ thống.</p>
          </div>

          <button
            type="button"
            className="add-product-button"
            onClick={() => {
              setMessage("");
              setMessageType("");
              setModalOpen(true);
            }}
          >
            + Thêm sản phẩm
          </button>
        </header>

        {message && (
          <div className={`page-message ${messageType}`}>
            {message}
          </div>
        )}

        <section className="product-table-card">
          {products.length === 0 ? (
            <div className="empty-products">
              Chưa có sản phẩm nào.
            </div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Ảnh trước</th>
                    <th>Tên sản phẩm</th>
                    <th>SKU</th>
                    <th>Danh mục</th>
                    <th>Giá</th>
                    <th>Số lượng</th>
                    <th className="actions-header">Thao tác</th>
                  </tr>
                </thead>

                <tbody>
                  {products.map((product) => (
                    <tr key={product.id}>
                      <td>
                        <ProductThumbnail product={product} />
                      </td>

                      <td>{product.name}</td>
                      <td>{product.sku}</td>
                      <td>{product.categoryName}</td>
                      <td>{formatPrice(product.price)}</td>
                      <td>{product.quantity}</td>

                      <td className="actions-cell">
                        <button
                          type="button"
                          className="action-button view-action"
                          title="Xem chi tiết"
                          aria-label={`Xem ${product.name}`}
                          onClick={() => handleView(product)}
                        >
                          <ViewIcon />
                        </button>

                        <button
                          type="button"
                          className="action-button edit-action"
                          title="Chỉnh sửa"
                          aria-label={`Sửa ${product.name}`}
                          onClick={() => handleEdit(product)}
                        >
                          <EditIcon />
                        </button>

                        <button
                          type="button"
                          className="action-button delete-action"
                          title="Xóa"
                          aria-label={`Xóa ${product.name}`}
                          onClick={() => handleDelete(product)}
                        >
                          <DeleteIcon />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <ProductModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onCreated={handleCreated}
      />
    </main>
  );
}

export default ProductPage;
