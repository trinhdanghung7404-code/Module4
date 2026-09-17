import { useEffect, useState } from "react";
import {
  createCategory,
  deleteCategory,
  getCategories,
  updateCategory,
} from "../api/categoryApi";
import {
  DeleteIcon,
  EditIcon,
  ErrorIcon,
  SuccessIcon,
} from "../components/TableIcons";
import "../styles/product.css";

const MAX_NAME_LENGTH = 20;

const initialToast = {
  message: "",
  type: "success",
};

function CategoryPage() {
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [toast, setToast] = useState(initialToast);

  const showToast = (message, type = "success") => {
    setToast({ message, type });
  };

  const closeToast = () => {
    setToast(initialToast);
  };

  useEffect(() => {
    getCategories()
      .then(setCategories)
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

  const handleCreate = async (event) => {
    event.preventDefault();

    const name = newName.trim();

    if (!name) {
      showToast("Tên danh mục không được để trống", "error");
      return;
    }

    setCreating(true);

    try {
      const created = await createCategory(name);

      setCategories((previous) => [...previous, created]);
      setNewName("");
      showToast("Thêm danh mục thành công");
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setCreating(false);
    }
  };

  const startEdit = (category) => {
    setEditingId(category.id);
    setEditingName(category.name);
  };

  const cancelEdit = () => {
    if (saving) {
      return;
    }

    setEditingId(null);
    setEditingName("");
  };

  const handleSaveEdit = async (category) => {
    if (saving) {
      return;
    }

    const name = editingName.trim();

    if (!name) {
      showToast("Tên danh mục không được để trống", "error");
      return;
    }

    if (name === category.name) {
      cancelEdit();
      return;
    }

    setSaving(true);

    try {
      const updated = await updateCategory(category.id, name);

      setCategories((previous) =>
        previous.map((item) => (item.id === updated.id ? updated : item))
      );
      setEditingId(null);
      setEditingName("");
      showToast("Đổi tên danh mục thành công");
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setSaving(false);
    }
  };

  const openDeleteModal = (category) => {
    setDeleteTarget(category);
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

    const category = deleteTarget;
    setDeletingId(category.id);

    try {
      await deleteCategory(category.id);

      setCategories((previous) =>
        previous.filter((item) => item.id !== category.id)
      );
      setDeleteTarget(null);
      showToast("Xóa danh mục thành công");
    } catch (error) {
      setDeleteTarget(null);
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
            <h1>Quản lý danh mục</h1>
            <p>Tạo mới, đổi tên và xóa danh mục sản phẩm.</p>
          </div>

          <form
            className="quick-category category-create"
            onSubmit={handleCreate}
          >
            <input
              type="text"
              value={newName}
              maxLength={MAX_NAME_LENGTH}
              placeholder="Tên danh mục mới"
              aria-label="Tên danh mục mới"
              disabled={creating}
              onChange={(event) => setNewName(event.target.value)}
            />

            <button type="submit" disabled={creating || !newName.trim()}>
              {creating ? "Đang thêm..." : "+ Thêm danh mục"}
            </button>
          </form>
        </header>

        <section className="product-table-card">
          {loading ? (
            <div className="empty-products">Đang tải danh mục…</div>
          ) : categories.length === 0 ? (
            <div className="empty-products">Chưa có danh mục nào.</div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Tên danh mục</th>
                    <th>Số sản phẩm</th>
                    <th className="actions-header">Thao tác</th>
                  </tr>
                </thead>

                <tbody>
                  {categories.map((category, index) => (
                    <tr key={category.id}>
                      <td>{index + 1}</td>

                      <td className="category-name-cell">
                        {editingId === category.id ? (
                          <div className="quick-category">
                            <input
                              type="text"
                              value={editingName}
                              maxLength={MAX_NAME_LENGTH}
                              aria-label={`Tên mới của ${category.name}`}
                              disabled={saving}
                              autoFocus
                              onChange={(event) =>
                                setEditingName(event.target.value)
                              }
                              onKeyDown={(event) => {
                                if (event.key === "Enter") {
                                  event.preventDefault();
                                  handleSaveEdit(category);
                                }

                                if (event.key === "Escape") {
                                  cancelEdit();
                                }
                              }}
                            />

                            <button
                              type="button"
                              disabled={saving}
                              onClick={() => handleSaveEdit(category)}
                            >
                              {saving ? "Đang lưu..." : "Lưu"}
                            </button>

                            <button
                              type="button"
                              disabled={saving}
                              onClick={cancelEdit}
                            >
                              Hủy
                            </button>
                          </div>
                        ) : (
                          category.name
                        )}
                      </td>

                      <td>
                        {category.productCount > 0 ? (
                          category.productCount
                        ) : (
                          <span className="category-empty-count">
                            Chưa có
                          </span>
                        )}
                      </td>

                      <td className="actions-cell">
                        <button
                          type="button"
                          className="action-button edit-action"
                          title="Đổi tên"
                          aria-label={`Đổi tên ${category.name}`}
                          disabled={editingId !== null || saving}
                          onClick={() => startEdit(category)}
                        >
                          <EditIcon />
                        </button>

                        <button
                          type="button"
                          className="action-button delete-action"
                          title={
                            category.productCount > 0
                              ? "Danh mục còn sản phẩm, không thể xóa"
                              : "Xóa"
                          }
                          aria-label={`Xóa ${category.name}`}
                          disabled={category.productCount > 0}
                          onClick={() => openDeleteModal(category)}
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
        <div className="delete-confirm-overlay" onMouseDown={closeDeleteModal}>
          <div
            className="delete-confirm-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="category-delete-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="delete-confirm-icon">
              <DeleteIcon />
            </div>

            <div className="delete-confirm-content">
              <h2 id="category-delete-title">Xác nhận xóa danh mục</h2>
              <p>
                Bạn có chắc muốn xóa danh mục{" "}
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
                {deletingId !== null ? "Đang xóa..." : "Xóa danh mục"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CategoryPage;