import { request } from "./client";

/**
 * Mọi đường ở đây đều cần token admin — client.js gắn header giúp, file này không
 * được phép tự gọi fetch.
 *
 * createProduct/updateProduct gửi FormData: phải truyền `form: true` để client.js
 * KHÔNG tự đặt Content-Type application/json (boundary của multipart do trình duyệt
 * viết, đoán tay là sai).
 */
export function getProducts() {
  return request("/products");
}

export function createProduct(formData) {
  return request("/products", { method: "POST", body: formData, form: true });
}

export function updateProduct(productId, formData) {
  return request(`/products/${productId}`, {
    method: "PUT",
    body: formData,
    form: true,
  });
}

export function deleteProduct(productId) {
  return request(`/products/${productId}`, { method: "DELETE" });
}

