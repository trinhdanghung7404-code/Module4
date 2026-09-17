import { request } from "./shopApi.js";

/**
 * Tai khoan khach + so nguoi nhan + don cua chinh minh.
 *
 * Moi duong deu co dang "/users/me/...": khong con cho nao de dien id cua mot
 * tai khoan khac vao duong dan. Danh tinh nam trong header Authorization do
 * shopApi gan tu phien dang nhap, nen cac ham o day khong phai nhan tham so
 * "userId" nao nua — bo tham so do di la bo luon kha nang gui sai.
 */

function json(method, body) {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

/** Chi gui dung nhung field backend khai bao, khong lan truyen key thua cua form. */
function recipientBody({ name, phone, address, label, isDefault } = {}) {
  return {
    name: String(name ?? "").trim(),
    phone: String(phone ?? "").trim(),
    address: String(address ?? "").trim(),
    label: String(label ?? "").trim(),
    isDefault: Boolean(isDefault),
  };
}

/**
 * Dang nhap / dang ky tra ve { token, user }.
 *
 * `account` duoc phep la username hoac email — dung ten cua no o day cung la
 * cach phat hanh cua `userApi` cu, giu nguyen de form khoi doi.
 */
export function loginUser({ account, password }) {
  return request("/users/login", json("POST", { account, password }));
}

export function registerUser({ username, fullName, email, password, confirmPassword }) {
  return request(
    "/users/register",
    json("POST", { username, fullName, email, password, confirmPassword })
  );
}

/**
 * Thu hoi phien tren may chu.
 *
 * Goi nay co the that bai (mat mang, token da bi xoa o tab khac): noi dung nao
 * cung van phai xoa phien o machine locally, neu khong thi khach bam "Dang xuat"
 * ma van con ten cua minh tren header.
 */
export function logoutUser() {
  return request("/users/logout", { method: "POST" });
}

/** Client goi day de biet token con song khong (xem hooks/useUser.js). */
export function fetchProfile() {
  return request("/users/me");
}

export function fetchRecipients() {
  return request("/users/me/recipients");
}

export function addRecipient(payload) {
  return request("/users/me/recipients", json("POST", recipientBody(payload)));
}

export function updateRecipient(recipientId, payload) {
  return request(
    `/users/me/recipients/${encodeURIComponent(recipientId)}`,
    json("PUT", recipientBody(payload))
  );
}

/** Server trả 204, request() dịch thành null. */
export function deleteRecipient(recipientId) {
  return request(`/users/me/recipients/${encodeURIComponent(recipientId)}`, {
    method: "DELETE",
  });
}

/** Trả về cả danh sách mới, không cần gọi lại fetchRecipients(). */
export function setDefaultRecipient(recipientId) {
  return request(`/users/me/recipients/${encodeURIComponent(recipientId)}/default`, {
    method: "PUT",
  });
}

/** "Đơn hàng của tôi": 50 đơn gần nhất, mới nhất trước. */
export function fetchMyOrders() {
  return request("/orders/mine");
}
