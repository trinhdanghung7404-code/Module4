import { request } from "./client";

/**
 * loginAdmin trả về nguyên { token, admin } — client không còn nhận một cái hồ sơ
 * trơ rồi tự coi đó là bằng chứng đã đăng nhập như bản cũ.
 *
 * Đăng nhập vẫn là đường công khai duy nhất; /me, /logout và mọi endpoint khác đều
 * phải kèm token (AdminSessionInterceptor).
 */
export function loginAdmin(payload) {
  return request("/login", { method: "POST", body: payload });
}

/** Vẫn công khai — xem ghi chú về lỗ hổng này trong AdminController. */
export function registerAdmin(payload) {
  return request("/register", { method: "POST", body: payload });
}

/** Thu hồi phiên trên server. 401 đã được client.js xử tập trung. */
export function logoutAdmin() {
  return request("/logout", { method: "POST" });
}

/** Kiểm tra phiên còn sống hay đã hết hạn. */
export function fetchAdminProfile() {
  return request("/me");
}

