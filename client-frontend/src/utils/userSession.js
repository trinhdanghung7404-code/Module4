/**
 * Phiên đăng nhập của khách: { token, user }.
 *
 * Token la thu server kiem tra duoc, con profile chi de hien ten. Tu day moi goi
 * API thuoc ve khach deu kem header "Authorization: Bearer &lt;token&gt;" (gan o
 * api/shopApi.js, mot cho duy nhat), va server tra loi "ai dang goi" tu token chu
 * khong tu body hay duong dan nua.
 *
 * Luu localStorage chu khong sessionStorage: khach dat don tren may ca nhan va
 * mo lai sau mot tuan thi khong phai nhap lai mat khau. Gia phai: may dung chung
 * trinh duyet se giu lai token, va ai cam duoc no co toan bo quyen cua tai khoan
 * den het han (mac dinh 30 ngay, xem app.user.session-ttl-days). Muon chat hon
 * thi doi cai do trong application.properties, khong can doi file nay.
 *
 * Key la v2: du lieu cua kieu cu (chi co profile, khong co token) bi loai thay vi
 * doc lai, de khach dung tab mo san khong bi coi la dang nhap trong khi moi request
 * deu bi tu choi.
 */

const SESSION_KEY = "shop.userSession.v2";

/** Sự kiện nội bộ: báo cho UI rằng phiên vừa đổi (đăng nhập/đăng xuất/hết hạn). */
export const SESSION_EVENT = "shop:user-session";

/** Token that la 43 ky tu base64url; canh duoi nay chi de chan chuoi ro rang gia. */
const MIN_TOKEN_LENGTH = 20;

/**
 * id phai la so nguyen duong va username phai la chuoi: ca hai thuoc duoc dung
 * de chao ten khach va de nhat du lieu ca nhan vao UI. Mot localStorage bi ghi
 * tay "id": "abc" thi tot nhat la bi coi nhu chua dang nhap.
 */
function isValidProfile(profile) {
  return (
    profile !== null &&
    typeof profile === "object" &&
    Number.isInteger(profile.id) &&
    profile.id > 0 &&
    typeof profile.username === "string" &&
    profile.username.length > 0
  );
}

/** Ca token lan profile deu phai on: thieu mot trong hai la phien vo nghia. */
function isValidSession(session) {
  return (
    session !== null &&
    typeof session === "object" &&
    typeof session.token === "string" &&
    session.token.trim().length >= MIN_TOKEN_LENGTH &&
    isValidProfile(session.user)
  );
}

function hasStorage() {
  return typeof window !== "undefined" && !!window.localStorage;
}

function announce() {
  // ShopHeader dung listener nay de cap nhat ngay khi tab khac dang nhap, thay
  // vi hien "hung" gia trang su den khi khach tai lai trang.
  if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
    window.dispatchEvent(new Event(SESSION_EVENT));
  }
}

/**
 * Nhan dung { token, user } nhu /users/login tra ve.
 *
 * Nem loi khi hinh dang sai: goi nay den tu form dang nhap, va "dang nhap thanh
 * cong" ma khong luu duoc token thi lan cuoc tiep se bi tu choi — tot hon la bao
 * that cho khach biet ngay bay gio.
 */
export function saveUserSession(session) {
  if (!isValidSession(session)) {
    throw new Error("Không nhận được phiên đăng nhập từ máy chủ");
  }
  if (!hasStorage()) return;
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  announce();
}

export function getUserSession() {
  if (!hasStorage()) return null;

  const raw = window.localStorage.getItem(SESSION_KEY);
  if (!raw) return null;

  try {
    const session = JSON.parse(raw);
    if (isValidSession(session)) return session;
  } catch {
    // JSON hong: dung nem loi giua luc doc trang, chi coi la chua dang nhap.
  }

  window.localStorage.removeItem(SESSION_KEY);
  return null;
}

export function clearUserSession() {
  if (!hasStorage()) return;
  window.localStorage.removeItem(SESSION_KEY);
  announce();
}

export function getUserProfile() {
  return getUserSession()?.user ?? null;
}

/** null khi chua dang nhap — shopApi dua vao do de quyet co gan header khong. */
export function getUserToken() {
  return getUserSession()?.token ?? null;
}

export function isLoggedIn() {
  return getUserSession() !== null;
}

export function getInitials(profile) {
  const source = (profile?.fullName || profile?.username || "").trim();
  if (!source) return "KH";

  const words = source.split(/\s+/);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();

  return (words[0].charAt(0) + words[words.length - 1].charAt(0)).toUpperCase();
}
