import { useEffect, useRef, useState } from "react";
import {
  createProduct,
  updateProduct,
} from "../api/productApi";
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
    imageType: "FRONT",
  },
  {
    name: "backImage",
    label: "Ảnh sau",
    imageType: "BACK",
  },
  {
    name: "leftImage",
    label: "Ảnh trái",
    imageType: "LEFT",
  },
  {
    name: "rightImage",
    label: "Ảnh phải",
    imageType: "RIGHT",
  },
];

function revokeBlobPreviews(previews) {
  Object.values(previews).forEach((preview) => {
    if (preview?.startsWith("blob:")) {
      URL.revokeObjectURL(preview);
    }
  });
}

function getProductPreviews(product) {
  if (!product?.images) {
    return initialPreviews;
  }

  return imageFields.reduce(
    (result, { name, imageType }) => {
      const image = product.images.find(
        (item) => item.imageType === imageType
      );

      result[name] = image?.imageUrl ?? "";
      return result;
    },
    { ...initialPreviews }
  );
}

function getProductForm(product) {
  if (!product) {
    return initialForm;
  }

  return {
    name: product.name ?? "",
    sku: product.sku ?? "",
    price: product.price ?? "",
    quantity: product.quantity ?? "",
    categoryId: String(
      product.categoryId ?? product.category?.id ?? ""
    ),
    description: product.description ?? "",
  };
}

/**
 * Modal sản phẩm — chi duoc render khi mo (ProductPage quyet dieu do), nen
 * "mo" cung nghia voi "mounted": form duoc khoi tao tu props ngay lan render dau
 * tien, va khong con effect nao reset lai nua.
 *
 * Vi sao khong giu effect reset: go setState trong luc effect chay xong khien
 * React render hai lan lien tiep cho mot lan mo modal — lan dau voi form cua san
 * pham tru do, lan hai moi dung form cua san pham dang mo. Voi mot modal gan 600
 * dong thi do la cai nhip nhoi that su, va `react-hooks/set-state-in-effect` khong
 * phan doi thich vu: chi dung cho. Cach React khuyen dung cho "reset state khi
 * prop doi" la remount — tuc la mot prop `key` o cho goi den, dat trong
 * ProductPage.
 */
function ProductModal({
  mode = "create",
  product,
  onClose,
  onCreated,
  onUpdated,
}) {
  const isView = mode === "view";
  const isEdit = mode === "edit";

  const [form, setForm] = useState(() =>
    isView || isEdit ? getProductForm(product) : initialForm
  );
  const [files, setFiles] = useState(initialFiles);
  const [previews, setPreviews] = useState(() =>
    isView || isEdit ? getProductPreviews(product) : initialPreviews
  );
  const [categories, setCategories] = useState([]);
  const [newCategory, setNewCategory] = useState("");
  const [showCategoryForm, setShowCategoryForm] =
    useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const fileInputRefs = useRef({});
  const previewsRef = useRef(previews);

  // Danh muc phai xin tu server, nen day la effect duy nhat o day: no goi ra ben
  // ngoai React, va setState chi xay ra trong callback tra ve — khong phai trong
  // luc than effect chay.
  useEffect(() => {
    if (isView) {
      return undefined;
    }

    let ignored = false;

    getCategories()
      .then((list) => {
        if (!ignored) {
          setCategories(list);
        }
      })
      .catch((error) => {
        if (!ignored) {
          setMessage(error.message);
        }
      });

    return () => {
      ignored = true;
    };
  }, [isView]);

  useEffect(() => {
    previewsRef.current = previews;
  }, [previews]);

  useEffect(() => {
    return () => revokeBlobPreviews(previewsRef.current);
  }, []);

  const resetForm = () => {
    revokeBlobPreviews(previews);
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
      if (previous[name]?.startsWith("blob:")) {
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

      setCategories((previous) => [...previous, category]);
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

    if (isView) {
      return;
    }

    setMessage("");

    const missingImage = imageFields.find(
      ({ name }) => !files[name] && !previews[name]
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
      if (files[name]) {
        formData.append(name, files[name]);
      }
    });

    setLoading(true);

    try {
      if (isEdit) {
        const updatedProduct = await updateProduct(
          product.id,
          formData
        );
        onUpdated(updatedProduct);
      } else {
        const createdProduct = await createProduct(formData);
        onCreated(createdProduct);
      }

      resetForm();
      onClose();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  const title = isView
    ? "Chi tiết sản phẩm"
    : isEdit
      ? "Chỉnh sửa sản phẩm"
      : "Thêm sản phẩm";

  const description = isView
    ? "Thông tin và hình ảnh hiện tại của sản phẩm."
    : isEdit
      ? "Cập nhật thông tin; chỉ chọn lại ảnh cần thay đổi."
      : "Nhập thông tin và tải đủ bốn ảnh sản phẩm.";

  return (
    <div className="modal-overlay" onMouseDown={handleClose}>
      <div
        className="product-modal"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2>{title}</h2>
            <p>{description}</p>
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
                disabled={isView}
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
                disabled={isView}
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
                disabled={isView}
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
                disabled={isView}
              />
            </div>

            <div className="form-group">
              <label htmlFor="product-category">Danh mục</label>

              {isView ? (
                <input
                  id="product-category"
                  value={product?.categoryName ?? ""}
                  disabled
                  readOnly
                />
              ) : (
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
              )}
            </div>

            {showCategoryForm && !isView && (
              <div className="quick-category full-column">
                <input
                  value={newCategory}
                  onChange={(event) =>
                    setNewCategory(event.target.value)
                  }
                  placeholder="Tên danh mục mới"
                  maxLength={20}
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
                {!isView && (
                  <span>
                    JPG, PNG hoặc WEBP; tối đa 5 MB/ảnh
                  </span>
                )}
              </div>

              <div className="image-upload-grid">
                {imageFields.map(({ name, label }) => (
                  <label
                    className={`image-upload-box${
                      isView ? " view-only" : ""
                    }`}
                    key={name}
                  >
                    {previews[name] ? (
                      <img
                        src={previews[name]}
                        alt={label}
                      />
                    ) : (
                      <div className="image-upload-placeholder">
                        <strong>+</strong>
                        <span>{label}</span>
                      </div>
                    )}

                    {!isView && (
                      <input
                        ref={(element) => {
                          fileInputRefs.current[name] = element;
                        }}
                        name={name}
                        type="file"
                        accept="image/jpeg,image/png,image/webp"
                        onChange={handleFileChange}
                        required={mode === "create"}
                      />
                    )}

                    <span className="image-upload-label">
                      {isView
                        ? label
                        : files[name]?.name ||
                          (isEdit ? `Đổi ${label.toLowerCase()}` : "Chọn ảnh")}
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
                disabled={isView}
              />
            </div>
          </div>

          {message && (
            <div className="product-error">{message}</div>
          )}

          <div className="modal-actions">
            <button
              type="button"
              className="cancel-button"
              onClick={handleClose}
              disabled={loading}
            >
              {isView ? "Đóng" : "Hủy"}
            </button>

            {!isView && (
              <button
                type="submit"
                className="save-button"
                disabled={loading}
              >
                {loading
                  ? isEdit
                    ? "Đang cập nhật..."
                    : "Đang tải ảnh và lưu..."
                  : isEdit
                    ? "Lưu thay đổi"
                    : "Lưu sản phẩm"}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

export default ProductModal;
