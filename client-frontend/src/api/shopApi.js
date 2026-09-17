import { labelForField } from "../utils/orderPayload.js";
import { clearUserSession, getUserToken } from "../utils/userSession.js";

// Optional chaining keeps this module importable outside Vite (plain Node tests).
const API_URL = import.meta.env?.VITE_API_URL || "http://localhost:8080/api";

/** Sort slugs accepted by the backend whitelist. Keep in sync with ProductService.SORT_MODES. */
export const SORT_OPTIONS = [
  { value: "newest", label: "Mới nhất" },
  { value: "price-asc", label: "Giá thấp → cao" },
  { value: "price-desc", label: "Giá cao → thấp" },
  { value: "name-asc", label: "Tên A → Z" },
  { value: "quantity-desc", label: "Còn nhiều nhất" },
];

export const DEFAULT_PAGE_SIZE = 12;

/**
 * Mọi gọi API của shop đi qua đây: một chỗ duy nhất quyết định base URL, một chỗ
 * duy nhất biến body lỗi của backend thành thông báo đọc được + gắn error.status,
 * va mot cho duy nhat gan token dang nhap vao header. userApi.js dùng lại đúng
 * hàm này để hai bên không bất đồng cách xử lý lỗi.
 *
 * Header gan o day chu khong o tung goi cu the: them endpoint can dang nhap moi
 * thi khong ai phai nho gan Authorization, va quen gan la kieu loi khong the phat
 * hien bang mat thuong.
 */
export async function request(path, options = {}) {
  const headers = { ...(options.headers ?? {}) };
  const token = getUserToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...options, headers });
  } catch {
    // fetch chỉ reject khi trình duyệt KHÔNG đọc được response: backend chưa chạy,
    // sai địa chỉ, hoặc CORS từ chối (kể cả khi backend đã trả về dữ liệu). Nó
    // không có nghĩa "sản phẩm đã bị xoá": callers dựa vào error.status vắng mặt để
    // phân biệt hai chuyện đó.
    //
    // Nói rõ cả vế CORS vì một request mang Authorization không phải "simple
    // request": browser gửi OPTIONS preflight trước. Nếu preflight không đạt 2xx
    // thì browser báo network error, và "backend không chạy" là kết luận sai —
    // backend vẫn sống, chỉ origin này bị từ chối. Kèm origin để paste thẳng vào
    // app.cors.allowed-origins mà không phải đoán.
    const origin = typeof window === "undefined" ? "(ngoai trinh duyet)" : window.location.origin;
    throw new Error(
      `Không gọi được ${API_URL} từ origin ${origin}. ` +
        `Backend chưa chạy, sai địa chỉ, hoặc trình duyệt chặn CORS: origin này ` +
        `không nằm trong app.cors.allowed-origins, hoặc request bị chặn ngay ở bước ` +
        `preflight (OPTIONS). Mở DevTools > Network để xem request nào đỏ.`
    );
  }

  if (!response.ok) {
    // 401 = may chu khong con biet token nay. Giu lai cai ten trong localStorage
    // thi moi lan goi sau cung phai chiu them mot 401 nua va hien tin nhan "dang
    // nhap lai" khac khong, nen xoa phien ngay tai day — SESSION_EVENT se dua
    // khach ve man dang nhap qua RequireUser.
    if (response.status === 401) clearUserSession();

    let message = `Không gọi được API (HTTP ${response.status})`;
    try {
      message = extractApiError(await response.json(), response.status);
    } catch {
      // Error body was not JSON (Spring's default error payload); keep the generic message.
    }
    const error = new Error(message);
    // status phải được giữ nguyên: useCartVerification dựa vào nó để phân biệt
    // "server từ chối id này" (4xx -> gỡ dòng khỏi giỏ) với "không liên lạc được"
    // (thì không được xoá gì cả).
    error.status = response.status;
    throw error;
  }

  if (response.status === 204) return null;
  return response.json();
}


/**
 * Builds "/products?..." and drops empty values so the URL stays readable and the
 * backend receives its "no filter" sentinels instead of blank strings.
 */
export function buildProductQuery({
  page = 0,
  size = DEFAULT_PAGE_SIZE,
  categoryId = 0,
  search = "",
  sort = "newest",
} = {}) {
  const params = new URLSearchParams();
  params.set("page", String(page));
  params.set("size", String(size));
  params.set("sort", sort);

  const trimmed = search.trim();
  if (categoryId > 0) params.set("categoryId", String(categoryId));
  if (trimmed) params.set("search", trimmed);

  return `/products?${params.toString()}`;
}

export function fetchProducts(params) {
  return request(buildProductQuery(params));
}

export function fetchProduct(id) {
  return request(`/products/${encodeURIComponent(id)}`);
}

export function fetchCategories() {
  return request("/categories");
}

/**
 * Backend trả về hai hình thái lỗi khác nhau, và chỉ một trong hai có key "message":
 *   - BusinessException        -> { message: "SKU đã tồn tại" }
 *   - MethodArgumentNotValid   -> { customerName: "...", phone: "..." }
 * Nếu chỉ đọc body.message thì mọi lỗi validate của form đặt hàng hiện ra thành
 * "Không gọi được API (HTTP 400)" và khách không biết mình sai ở ô nào.
 */
export function extractApiError(body, status) {
  const fallback = `Không gọi được API (HTTP ${status})`;
  if (!body || typeof body !== "object") return fallback;

  if (typeof body.message === "string" && body.message.trim()) {
    return body.message.trim();
  }

  // Bỏ qua các field kỹ thuật của Spring, chỉ giữ field lỗi người dùng đọc được.
  const technical = new Set([
    "timestamp",
    "status",
    "error",
    "exception",
    "trace",
    "path",
    "details",
    "message",
  ]);

  const fieldErrors = Object.entries(body)
    .filter(([key, value]) => !technical.has(key) && typeof value === "string" && value.trim())
    .map(([key, value]) => `${labelForField(key)}: ${value.trim()}`);

  return fieldErrors.length > 0 ? fieldErrors.join("\n") : fallback;
}

/**
 * Đặt hàng. Body chỉ mang id + số lượng (xem buildOrderPayload): giá và tồn kho
 * do server chốt lại từ DB, và toàn bộ đơn bị rollback nếu một sản phẩm hết hàng.
 */
export function createOrder(orderPayload) {
  return request("/orders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(orderPayload),
  });
}

