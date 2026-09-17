import { useEffect, useState } from "react";
import {
  deleteProduct,
  getProducts,
} from "../api/productApi";
import ProductModal from "../components/ProductModal";
import {
  DeleteIcon,
  EditIcon,
  ErrorIcon,
  SuccessIcon,
  ViewIcon,
} from "../components/TableIcons";
import { formatPrice } from "../utils/format";
import "../styles/product.css";

function ProductThumbnail({ product }) {
  const [failed, setFailed] = useState(false);

  const frontImage = product.images?.find(
    (image) => image.imageType === "FRONT"
  );

  if (!frontImage?.imageUrl || failed) {
    return <div className="image-placeholder">Không có ảnh</div>;
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

const initialModal = {
  open: false,
  mode: "create",
  product: null,
};

const initialToast = {
  message: "",
  type: "success",
};

function ProductPage() {
  const [products, setProducts] = useState([]);
  const [modal, setModal] = useState(initialModal);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [toast, setToast] = useState(initialToast);
  const [loading, setLoading] = useState(true);

  const showToast = (message, type = "success") => {
    setToast({ message, type });
  };

  const closeToast = () => {
    setToast(initialToast);
  };

  useEffect(() => {
    getProducts()
      .then(setProducts)
      .catch((error) => showToast(error.message, "error"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!toast.message) {
      return undefined;
    }

    const timer = window.setTimeout(closeToast, 3500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!deleteTarget) {
      return undefined;
    }

    const handleKeyDown = (event) => {
      if (event.key === "Escape" && deletingId === null) {
        setDeleteTarget(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [deleteTarget, deletingId]);

  const openModal = (mode, product = null) => {
    setModal({ open: true, mode, product });
  };

  const closeModal = () => {
    setModal(initialModal);
  };

  const handleCreated = (product) => {
    setProducts((previous) => [product, ...previous]);
    showToast("Thêm sản phẩm thành công");
  };

  const handleUpdated = (product) => {
    setProducts((previous) =>
      previous.map((item) =>
        item.id === product.id ? product : item
      )
    );

    showToast("Cập nhật sản phẩm thành công");
  };

  const openDeleteModal = (product) => {
    setDeleteTarget(product);
  };

  const closeDeleteModal = () => {
    if (deletingId !== null) {
      return;
    }

    setDeleteTarget(null);
  };

  const handleDelete = async () => {
    if (!deleteTarget) {
      return;
    }

    const product = deleteTarget;
    setDeletingId(product.id);

    try {
      await deleteProduct(product.id);
      setProducts((previous) =>
        previous.filter((item) => item.id !== product.id)
      );
      setDeleteTarget(null);
      showToast("Xóa sản phẩm thành công");
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="product-page">
      <div className="product-container">
        <header className="product-page-header">
          <div>
            <h1>Danh sách sản phẩm</h1>
            <p>Quản lý sản phẩm trong hệ thống.</p>
          </div>

          <button
            type="button"
            className="add-product-button"
            onClick={() => openModal("create")}
          >
            + Thêm sản phẩm
          </button>
        </header>

        <section className="product-table-card">
          {loading ? (
            <div className="empty-products">
              Đang tải danh sách sản phẩm…
            </div>
          ) : products.length === 0 ? (
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
                          onClick={() => openModal("view", product)}
                        >
                          <ViewIcon />
                        </button>

                        <button
                          type="button"
                          className="action-button edit-action"
                          title="Chỉnh sửa"
                          aria-label={`Sửa ${product.name}`}
                          onClick={() => openModal("edit", product)}
                        >
                          <EditIcon />
                        </button>

                        <button
                          type="button"
                          className="action-button delete-action"
                          title="Xóa"
                          aria-label={`Xóa ${product.name}`}
                          onClick={() => openDeleteModal(product)}
                          disabled={deletingId === product.id}
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

      {toast.message && (
        <div
          className={`product-toast ${toast.type}`}
          role="status"
          aria-live="polite"
        >
          <span className="product-toast-icon">
            {toast.type === "error" ? <ErrorIcon /> : <SuccessIcon />}
          </span>

          <span className="product-toast-message">{toast.message}</span>

          <button
            type="button"
            className="product-toast-close"
            onClick={closeToast}
            aria-label="Đóng thông báo"
          >
            ×
          </button>
        </div>
      )}

      {deleteTarget && (
        <div
          className="delete-confirm-overlay"
          onMouseDown={closeDeleteModal}
        >
          <div
            className="delete-confirm-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-confirm-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="delete-confirm-icon">
              <DeleteIcon />
            </div>

            <div className="delete-confirm-content">
              <h2 id="delete-confirm-title">Xác nhận xóa sản phẩm</h2>
              <p>
                Bạn có chắc muốn xóa sản phẩm
                {" "}
                <strong>“{deleteTarget.name}”</strong>?
              </p>
              <span>Thao tác này không thể hoàn tác.</span>
            </div>

            <div className="delete-confirm-actions">
              <button
                type="button"
                className="delete-cancel-button"
                onClick={closeDeleteModal}
                disabled={deletingId !== null}
              >
                Hủy
              </button>

              <button
                type="button"
                className="delete-confirm-button"
                onClick={handleDelete}
                disabled={deletingId !== null}
              >
                {deletingId !== null ? "Đang xóa..." : "Xóa sản phẩm"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* `key` đổi theo mỗi lần mở: đóng modal là component bị gỡ, mở lại là một
          ProductModal hoàn toàn mới — không còn form hay ảnh blob còn sót từ lần
          trước, và vì thế ProductModal không cần effect reset nữa (effect đó gọi
          setState ngay lúc chạy nên React coi là một lần render thừa). */}
      {modal.open && (
        <ProductModal
          key={`${modal.mode}-${modal.product?.id ?? "new"}`}
          mode={modal.mode}
          product={modal.product}
          onClose={closeModal}
          onCreated={handleCreated}
          onUpdated={handleUpdated}
        />
      )}
    </div>
  );
}

export default ProductPage;
