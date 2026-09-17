/**
 * Payload + validation cho việc đặt hàng.
 *
 * Tách ra module thuần (không React, không fetch) để scripts/verify-logic.mjs test
 * được bằng Node: quy tắc quan trọng nhất ở đây là "không bao giờ gửi giá sang
 * server" mà không có test thì rất dễ bị ai đó thêm line.price vào payload lúc
 * refactor.
 */

const FIELD_LABELS = {
  customerName: "Họ tên người nhận",
  phone: "Số điện thoại",
  address: "Địa chỉ",
  note: "Ghi chú",
  items: "Danh sách sản phẩm",
  // Field cua form dang ky / dang nhap va cua so nguoi nhan.
  username: "Username",
  email: "Email",
  password: "Mật khẩu",
  confirmPassword: "Xác nhận mật khẩu",
  account: "Username hoặc email",
  name: "Tên người nhận",
  label: "Nhãn người nhận",
};

// Trùng khớp @Pattern của OrderCreateRequest phía backend.
const PHONE_PATTERN = /^\+?\d[\d .-]{6,18}$/;

/**
 * Giới độ dài của khối "người nhận" — khớp cột trong bảng orders/recipient.
 *
 * Client dung mot hinh thai duy nhat { name, phone, address }; `name` chi tro
 * thanh `customerName` luc dung body gui di (buildOrderPayload). Nay mai them
 * hinh thai thu hai vao form la the nao cung co mot chiec form dung sai.
 */
export const RECIPIENT_LIMITS = {
  name: 120,
  phone: 20,
  address: 255,
  note: 500,
};

export const ACCOUNT_LIMITS = {
  username: 50,
  fullName: 100,
  email: 100,
  password: 72,
  label: 30,
};

export function labelForField(field) {
  const base = field.split(/[.[]/)[0];
  return FIELD_LABELS[base] ?? base;
}

/**
 * Kiểm phía client để hiện lỗi ngay tại ô nhập; server vẫn là trọng tài cuối.
 *
 * Ba field này là NGƯỜI NHẬN HÀNG, không phải chủ tài khoản: mua quà tặng thì
 * người nhận là một người hoàn toàn khác, và đó chính là lý do tách hai khái niệm.
 */
export function validateRecipient(recipient = {}) {
  const errors = {};

  for (const field of ["name", "phone", "address"]) {
    const value = String(recipient[field] ?? "").trim();
    if (!value) {
      errors[field] = `${labelForField(field)} không được để trống`;
    } else if (value.length > RECIPIENT_LIMITS[field]) {
      errors[field] = `${labelForField(field)} tối đa ${RECIPIENT_LIMITS[field]} ký tự`;
    }
  }

  if (!errors.phone && !PHONE_PATTERN.test(String(recipient.phone ?? "").trim())) {
    errors.phone = "Số điện thoại không hợp lệ";
  }

  const note = String(recipient.note ?? "").trim();
  if (note.length > RECIPIENT_LIMITS.note) {
    errors.note = `Ghi chú tối đa ${RECIPIENT_LIMITS.note} ký tự`;
  }

  return errors;
}

/**
 * Validation của form đăng ký tài khoản.
 *
 * Chi co muc do nay vi backend (@Size/@Email/@Valid) moi la trong tai cuoi;
 * client lo phan "sai ro rang tai o nhap" con "username da ton tai" thi khong
 * the nao biet duoc truoc khi goi.
 */
export function validateAccount(form = {}) {
  const errors = {};
  const rules = {
    username: "Username",
    fullName: "Họ tên",
    email: "Email",
    password: "Mật khẩu",
  };

  for (const [field, label] of Object.entries(rules)) {
    const value = String(form[field] ?? "").trim();
    if (!value) errors[field] = `${label} không được để trống`;
  }

  const username = String(form.username ?? "").trim();
  if (username && username.length < 4) {
    errors.username = "Username phải từ 4-50 ký tự";
  }

  const email = String(form.email ?? "").trim();
  // Chỉ chặn "không có @ và không có dấu chấm sau @". Backend @Email là chuẩn
  // cuối, client không cần đoán trước cả đặc tả RFC.
  if (email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    errors.email = "Email không hợp lệ";
  }

  const password = String(form.password ?? "");
  if (password && password.length < 8) {
    errors.password = "Mật khẩu phải từ 8-72 ký tự";
  } else if (password.length > ACCOUNT_LIMITS.password) {
    errors.password = "Mật khẩu phải từ 8-72 ký tự";
  }

  if (form.confirmPassword !== undefined || password) {
    if (String(form.confirmPassword ?? "") !== password) {
      errors.confirmPassword = "Mật khẩu xác nhận không khớp";
    }
  }

  return errors;
}

/**
 * Đổi giỏ hàng thành body của POST /api/orders.
 *
 * Ba thứ duy nhất đi qua server: productId, quantity, và bộ ba người nhận.
 * GIÁ KHÔNG BAO GIỜ đi qua — giá, tên, ảnh là server tự lấy từ DB. Nếu cho client
 * gửi giá thì chỉ cần mở DevTools là mua được hàng giá 0đ.
 *
 * Ba field customerName/phone/address là NGUOI NHAN, không phải thông tin tài
 * khoản: đặt đơn tặng người khác thì hai thứ này khác nhau hoàn toàn.
 *
 * Khong con userId trong body: don gan voi tai khoan nao do SERVER quyet tu header
 * Authorization (xem ShopOrderController). Client khong con cho nao de tu khai
 * "don nay cua nguoi khac", cung khong con kieu loi "dat roi ma don khong thuoc ve
 * minh" vi quen truyen id.
 */
export function buildOrderPayload(lines = [], recipient = {}) {
  const items = lines
    .map((line) => ({
      productId: Number(line.productId),
      quantity: Number(line.quantity),
    }))
    .filter((item) => Number.isInteger(item.productId) && item.productId > 0)
    .map((item) => ({
      productId: item.productId,
      quantity: Number.isInteger(item.quantity) && item.quantity > 0 ? item.quantity : 1,
    }));

  const payload = {
    // Form dung `name`, hop dong voi server la `customerName`: phep dich nay
    // nam o bien API de mot hinh thai du dung cho ca so nguoi nhan va dat hang.
    customerName: String(recipient.customerName ?? recipient.name ?? "").trim(),
    phone: String(recipient.phone ?? "").trim(),
    address: String(recipient.address ?? "").trim(),
    items,
  };

  const note = String(recipient.note ?? "").trim();
  if (note) payload.note = note;

  return payload;
}

const RECEIPT_KEY = "shop.lastOrder.v1";

/**
 * Biên lai đơn hàng vừa tạo, giữ trong sessionStorage chứ không phải localStorage:
 * đây là thông tin một lần, để ở localStorage thì lần sau mở tab vẫn còn hiện đơn
 * cũ như vừa mua xong.
 */
export function saveOrderReceipt(order) {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(RECEIPT_KEY, JSON.stringify(order));
  } catch {
    // Tab chặn storage: trang cảm ơn sẽ hiện trạng thái "không tìm thấy đơn",
    // còn đơn hàng thì vẫn đã được lưu phía server.
  }
}

export function readOrderReceipt() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(RECEIPT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && parsed.orderCode ? parsed : null;
  } catch {
    return null;
  }
}
