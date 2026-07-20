import { useEffect, useRef, useState } from "react";
import { createProduct } from "../api/productApi";
import {
  createCategory,
  getCategories,
} from "../api/categoryApi";

const MAX_IMAGE_SIZE = 5 * 1024 * 1024;

const ALLOWED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
];

const initialForm = {
  name: "",
  sku: "",
  price: "",
  quantity: "",
  categoryId: "",
  description: "",
};

const initialFiles = {
  frontImage: null,
  backImage: null,
  leftImage: null,
  rightImage: null,
};

const initialPreviews = {
  frontImage: "",
  backImage: "",
  leftImage: "",
  rightImage: "",
};

const imageFields = [
  {
    name: "frontImage",
    label: "Ảnh trước",
  },
  {
    name: "backImage",
    label: "Ảnh sau",
  },
  {
    name: "leftImage",
    label: "Ảnh trái",
  },
  {
    name: "rightImage",
    label: "Ảnh phải",
  },
];

function ProductModal({ open, onClose, onCreated }) {
  const [form, setForm] = useState(initialForm);
  const [files, setFiles] = useState(initialFiles);
  const [previews, setPreviews] = useState(initialPreviews);
  const [categories, setCategories] = useState([]);
  const [newCategory, setNewCategory] = useState("");
  const [showCategoryForm, setShowCategoryForm] =
    useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const fileInputRefs = useRef({});

  useEffect(() => {
    if (!open) {
      return;
    }

    setMessage("");

    getCategories()
      .then(setCategories)
      .catch((error) => setMessage(error.message));
  }, [open]);

  useEffect(() => {
    return () => {
      Object.values(previews).forEach((preview) => {
        if (preview) {
          URL.revokeObjectURL(preview);
        }
      });
    };
  }, [previews]);

  if (!open) {
    return null;
  }

  const resetForm = () => {
    Object.values(previews).forEach((preview) => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    });

    setForm(initialForm);
    setFiles(initialFiles);
    setPreviews(initialPreviews);
    setNewCategory("");
    setShowCategoryForm(false);
    setMessage("");

    Object.values(fileInputRefs.current).forEach((input) => {
      if (input) {
        input.value = "";
      }
    });
  };

  const handleClose = () => {
    if (loading) {
      return;
    }

    resetForm();
    onClose();
  };

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleFileChange = (event) => {
    const { name, files: selectedFiles } = event.target;
    const file = selectedFiles?.[0];

    if (!file) {
      return;
    }

    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      setMessage("Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_IMAGE_SIZE) {
      setMessage("Mỗi ảnh không được vượt quá 5 MB");
      event.target.value = "";
      return;
    }

    setMessage("");

    setPreviews((previous) => {
      if (previous[name]) {
        URL.revokeObjectURL(previous[name]);
      }

      return {
        ...previous,
        [name]: URL.createObjectURL(file),
      };
    });

    setFiles((previous) => ({
      ...previous,
      [name]: file,
    }));
  };

  const handleCreateCategory = async () => {
    const name = newCategory.trim();

    if (!name) {
      setMessage("Nhập tên danh mục");
      return;
    }

    try {
      const category = await createCategory(name);

      setCategories((previous) => [
        ...previous,
        category,
      ]);

      setForm((previous) => ({
        ...previous,
        categoryId: String(category.id),
      }));

      setNewCategory("");
      setShowCategoryForm(false);
      setMessage("");
    } catch (error) {
      setMessage(error.message);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage("");

    const missingImage = imageFields.find(
      ({ name }) => !files[name]
    );

    if (missingImage) {
      setMessage(`${missingImage.label} không được để trống`);
      return;
    }

    const formData = new FormData();

    formData.append("name", form.name.trim());
    formData.append("sku", form.sku.trim());
    formData.append("price", form.price);
    formData.append("quantity", form.quantity);
    formData.append("categoryId", form.categoryId);
    formData.append("description", form.description.trim());

    imageFields.forEach(({ name }) => {
      formData.append(name, files[name]);
    });

    setLoading(true);

    try {
      const product = await createProduct(formData);

      onCreated(product);
      resetForm();
      onClose();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="modal-overlay"
      onMouseDown={handleClose}
    >
      <div
        className="product-modal"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2>Thêm sản phẩm</h2>
            <p>Nhập thông tin và tải đủ bốn ảnh sản phẩm.</p>
          </div>

          <button
            type="button"
            className="modal-close"
            onClick={handleClose}
            disabled={loading}
            aria-label="Đóng"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="product-form-grid">
            <div className="form-group full-column">
              <label htmlFor="product-name">Tên sản phẩm</label>
              <input
                id="product-name"
                name="name"
                value={form.name}
                onChange={handleChange}
                maxLength={150}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="product-sku">SKU</label>
              <input
                id="product-sku"
                name="sku"
                value={form.sku}
                onChange={handleChange}
                maxLength={50}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="product-quantity">Số lượng</label>
              <input
                id="product-quantity"
                name="quantity"
                type="number"
                min="0"
                step="1"
                value={form.quantity}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="product-price">Giá</label>
              <input
                id="product-price"
                name="price"
                type="number"
                min="0.01"
                step="0.01"
                value={form.price}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="product-category">Danh mục</label>

              <div className="category-control">
                <select
                  id="product-category"
                  name="categoryId"
                  value={form.categoryId}
                  onChange={handleChange}
                  required
                >
                  <option value="">Chọn danh mục</option>

                  {categories.map((category) => (
                    <option
                      key={category.id}
                      value={category.id}
                    >
                      {category.name}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  className="small-button"
                  onClick={() =>
                    setShowCategoryForm(
                      (previous) => !previous
                    )
                  }
                >
                  + Mới
                </button>
              </div>
            </div>

            {showCategoryForm && (
              <div className="quick-category full-column">
                <input
                  value={newCategory}
                  onChange={(event) =>
                    setNewCategory(event.target.value)
                  }
                  placeholder="Tên danh mục mới"
                  maxLength={100}
                />

                <button
                  type="button"
                  onClick={handleCreateCategory}
                >
                  Lưu danh mục
                </button>
              </div>
            )}

            <div className="image-upload-section full-column">
              <div className="image-upload-heading">
                <label>Ảnh sản phẩm</label>
                <span>JPG, PNG hoặc WEBP; tối đa 5 MB/ảnh</span>
              </div>

              <div className="image-upload-grid">
                {imageFields.map(({ name, label }) => (
                  <label
                    className="image-upload-box"
                    key={name}
                  >
                    {previews[name] ? (
                      <img
                        src={previews[name]}
                        alt={`Xem trước ${label.toLowerCase()}`}
                      />
                    ) : (
                      <div className="image-upload-placeholder">
                        <strong>+</strong>
                        <span>{label}</span>
                      </div>
                    )}

                    <input
                      ref={(element) => {
                        fileInputRefs.current[name] = element;
                      }}
                      name={name}
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      onChange={handleFileChange}
                      required
                    />

                    <span className="image-upload-label">
                      {files[name]?.name || "Chọn ảnh"}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group full-column">
              <label htmlFor="product-description">Mô tả</label>
              <textarea
                id="product-description"
                name="description"
                rows="4"
                value={form.description}
                onChange={handleChange}
                maxLength={5000}
              />
            </div>
          </div>

          {message && (
            <div className="product-error">
              {message}
            </div>
          )}

          <div className="modal-actions">
            <button
              type="button"
              className="cancel-button"
              onClick={handleClose}
              disabled={loading}
            >
              Hủy
            </button>

            <button
              type="submit"
              className="save-button"
              disabled={loading}
            >
              {loading ? "Đang tải ảnh và lưu..." : "Lưu sản phẩm"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ProductModal;
