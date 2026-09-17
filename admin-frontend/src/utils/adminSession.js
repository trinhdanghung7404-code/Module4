const SESSION_KEY = "adminSession";
const LEGACY_FLAG_KEY = "adminLoggedIn";

/**
 * Một phiên admin = token + hồ sơ, chứa trong cùng một object.
 *
 * Giữ nguyên SESSION_KEY nhưng buộc phải có token. Hệ quả có chủ đích: phiên do bản
 * cũ lưu lại (chỉ có hồ sơ, không có token) bị coi như chưa đăng nhập và admin phải
 * đăng nhập lại một lần. Đó là điều đúng — cái hồ sơ cũ không chứng minh được gì với
 * server, mà trước đây ProtectedRoute lại tin nó.
 *
 * Lưu ý đã biết: bất kỳ đoạn JS nào chạy trên trang cũng đọc được localStorage, nên
 * một lỗi XSS trên trang quản trị lấy được token này. Chặn hẳn phải chuyển sang
 * cookie HttpOnly + SameSite kèm CSRF token; đó là việc lớn hơn nhiều so với vòng này.
 */
function isValidSession(value) {
  if (value === null || typeof value !== "object") {
    return false;
  }

  return (
    typeof value.token === "string" &&
    value.token.length > 0 &&
    typeof value.username === "string" &&
    value.username.length > 0
  );
}

/** Nhận đúng { token, admin } như AdminAuthResponse trả về. */
export function saveAdminAuth(auth) {
  const token = auth?.token;
  const admin = auth?.admin;

  if (
    typeof token !== "string" ||
    token.length === 0 ||
    typeof admin !== "object" ||
    admin === null
  ) {
    throw new Error("Server không trả về phiên đăng nhập hợp lệ");
  }

  localStorage.setItem(SESSION_KEY, JSON.stringify({ ...admin, token }));
  localStorage.removeItem(LEGACY_FLAG_KEY);
}

export function getAdminSession() {
  const raw = localStorage.getItem(SESSION_KEY);

  if (!raw) {
    return null;
  }

  try {
    const session = JSON.parse(raw);
    return isValidSession(session) ? session : null;
  } catch {
    // Dữ liệu cũ bị hỏng hoặc không phải JSON -> coi như chưa đăng nhập.
    localStorage.removeItem(SESSION_KEY);
    return null;
  }
}

export function getAdminToken() {
  return getAdminSession()?.token ?? null;
}

export function clearAdminSession() {
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(LEGACY_FLAG_KEY);
}

export function isLoggedIn() {
  return getAdminSession() !== null;
}

export function getInitials(profile) {
  const source = (profile?.fullName || profile?.username || "").trim();

  if (!source) {
    return "AD";
  }

  const words = source.split(/\s+/);

  if (words.length === 1) {
    return words[0].slice(0, 2).toUpperCase();
  }

  const firstLetter = words[0].charAt(0);
  const lastLetter = words[words.length - 1].charAt(0);

  return (firstLetter + lastLetter).toUpperCase();
}
