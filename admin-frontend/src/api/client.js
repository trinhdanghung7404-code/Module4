import { clearAdminSession, getAdminToken } from "../utils/adminSession";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080/api/admin";

/** Sự kiện AdminLayout nghe để đưa admin về màn đăng nhập khi server nói "phiên hết hạn". */
export const UNAUTHORIZED_EVENT = "admin:unauthorized";

/**
 * Chỗ DUY NHẤT gọi /api/admin/**.
 *
 * Trước đây mỗi file api tự viết fetch riêng, và chỉ cần một file quên gắn
 * Authorization là trang đó báo lỗi khó hiểu. Từ vòng này server đòi token cho mọi
 * đường /api/admin/** (AdminSessionInterceptor), nên "header nằm ở đâu" không được
 * phép là quyết định của từng file nữa.
 *
 * Không dùng filter/interceptor của fetch ở tầng component: dễ quên, và lỗi 401 rơi
 * rải rác ở từng trang thay vì xử lý một lần.
 */
export async function request(path, options = {}) {
  const { method = "GET", body, form = false, headers = {} } = options;

  const init = { method, headers: { ...headers } };

  const token = getAdminToken();
  if (token) {
    init.headers.Authorization = `Bearer ${token}`;
  }

  if (form) {
    // FormData: KHÔNG tự set Content-Type, trình duyệt phải tự viết boundary của multipart.
    init.body = body;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  let response;

  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch (cause) {
    throw new Error(
      `Không gọi được máy chủ tại ${API_URL}${path}. ` +
        "Kiểm tra backend đã chạy chưa và origin hiện tại có nằm trong app.cors.allowed-origins không.",
      { cause }
    );
  }

  const text = await response.text();
  let data = text;

  try {
    data = JSON.parse(text);
  } catch {
    // Backend vẫn trả chuỗi ở vài endpoint (register), nên text là kết quả hợp lệ.
  }

  if (!response.ok) {
    if (response.status === 401) {
      // Phiên đã hết hạn: xóa ngay để ProtectedRoute không còn cho đi qua bằng một
      // localStorage cùn, rồi báo để AdminLayout chuyển màn hình.
      clearAdminSession();
      window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
    }

    const error = new Error(extractMessage(data, response));
    error.status = response.status;
    throw error;
  }

  return data;
}

function extractMessage(data, response) {
  if (typeof data === "string") {
    return data.trim() || `Máy chủ trả về ${response.status}`;
  }

  if (data?.message) {
    return data.message;
  }

  // Body mặc định của Spring cho request sai kiểu: {status, error, path} — không có
  // "message" theo khuôn của GlobalExceptionHandler.
  if (data?.error) {
    return `${data.error} (${response.status})`;
  }

  if (data && typeof data === "object") {
    const first = Object.values(data)[0];

    if (typeof first === "string") {
      return first;
    }
  }

  return `Có lỗi xảy ra (${response.status})`;
}

/** Ghép query string, bỏ qua tham số rỗng để URL trên Network tab còn đọc được. */
export function buildQuery(params = {}) {
  const search = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  });

  const query = search.toString();
  return query ? `?${query}` : "";
}
