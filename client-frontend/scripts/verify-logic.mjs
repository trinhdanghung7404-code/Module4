// Behaviour checks for the pure client logic (no DOM, no network).
// Run: npm run verify
import assert from "node:assert/strict";

const memory = new Map();
const sessionMemory = new Map();
globalThis.window = {
  localStorage: {
    getItem: (key) => (memory.has(key) ? memory.get(key) : null),
    setItem: (key, value) => memory.set(key, String(value)),
    removeItem: (key) => memory.delete(key),
  },
  sessionStorage: {
    getItem: (key) => (sessionMemory.has(key) ? sessionMemory.get(key) : null),
    setItem: (key, value) => sessionMemory.set(key, String(value)),
    removeItem: (key) => sessionMemory.delete(key),
  },
  addEventListener: () => {},
};

let failures = 0;
function test(name, fn) {
  try {
    fn();
    console.log("  ok   " + name);
  } catch (error) {
    failures += 1;
    console.log("  FAIL " + name + "  ::  " + error.message);
  }
}

// Seed garbage BEFORE import: cartStore reads localStorage at module evaluation.
memory.set("shop.cart.v1", "{ not json");
const { cartStore } = await import("../src/utils/cartStore.js");
const { buildProductQuery, extractApiError } = await import("../src/api/shopApi.js");
const {
  buildOrderPayload,
  validateRecipient,
  validateAccount,
  saveOrderReceipt,
  readOrderReceipt,
} = await import("../src/utils/orderPayload.js");
const {
  saveUserSession,
  getUserSession,
  clearUserSession,
  getUserProfile,
  getUserToken,
  isLoggedIn,
  getInitials,
} = await import("../src/utils/userSession.js");
const {
  buildSteps,
  cancelInfo,
  isCancelled,
  reachedMap,
  statusLabel: orderStatusLabel,
} = await import("../src/utils/orderFlow.js");

console.log("cartStore");
test("corrupted localStorage recovers as an empty cart", () => {
  assert.deepEqual(cartStore.getSnapshot(), []);
});

cartStore.clear();
const product5 = { id: 7, name: "A", price: 10, quantity: 5 };

test("add() stores a snapshot line", () => {
  assert.equal(cartStore.add(product5, 2).ok, true);
  const line = cartStore.getSnapshot()[0];
  assert.deepEqual(
    { id: line.productId, qty: line.quantity, price: line.price },
    { id: 7, qty: 2, price: 10 }
  );
});

test("add() clamps to stock instead of over-selling", () => {
  const result = cartStore.add(product5, 4);
  assert.equal(result.ok, true);
  assert.equal(result.clampedTo, 5);
  assert.equal(cartStore.getSnapshot()[0].quantity, 5);
});

test("add() refuses when the line is already at stock", () => {
  const result = cartStore.add(product5, 1);
  assert.equal(result.ok, false);
  assert.match(result.reason, /5/);
  assert.equal(cartStore.getSnapshot()[0].quantity, 5);
});

test("setQuantity() caps at stock and rejects garbage", () => {
  cartStore.setQuantity(7, 99);
  assert.equal(cartStore.getSnapshot()[0].quantity, 5);
  cartStore.setQuantity(7, -3);
  cartStore.setQuantity(7, 0);
  cartStore.setQuantity(7, Number.NaN);
  cartStore.setQuantity(7, "");
  assert.equal(cartStore.getSnapshot().length, 1);
  assert.equal(cartStore.getSnapshot()[0].quantity, 5);
});

test("add() ignores a product without a usable id", () => {
  assert.equal(cartStore.add({ name: "no id", price: 1 }, 1).ok, false);
});

console.log("syncWithProducts");
cartStore.clear();
cartStore.add({ id: 8, name: "B", price: 20, quantity: 3 }, 1);
cartStore.add(product5, 1);

test("a partial catalog does NOT prune (the paged-grid trap)", () => {
  const removed = cartStore.syncWithProducts([{ id: 7, name: "A", price: 99, quantity: 5 }]);
  assert.deepEqual(removed, []);
  assert.equal(cartStore.getSnapshot().length, 2);
});

test("sync refreshes price from live data", () => {
  assert.equal(cartStore.getSnapshot().find((l) => l.productId === 7).price, 99);
});

test("full coverage + pruneMissing drops deleted lines", () => {
  const removed = cartStore.syncWithProducts(
    [{ id: 7, name: "A", price: 99, quantity: 5 }],
    { pruneMissing: true }
  );
  assert.deepEqual(removed, [8]);
  assert.equal(cartStore.getSnapshot().length, 1);
});

test("cart survives a reload: persisted JSON is re-normalized", () => {
  const raw = JSON.parse(memory.get("shop.cart.v1"));
  assert.equal(raw[0].productId, 7);
});

console.log("buildProductQuery");
test("defaults keep the backend sentinels out of the URL", () => {
  assert.equal(
    buildProductQuery({}),
    "/products?page=0&size=12&sort=newest"
  );
});

test("search is trimmed, categoryId 0 is omitted", () => {
  const url = buildProductQuery({ search: "  hung ", categoryId: 0, page: 2 });
  assert.match(url, /page=2/);
  assert.match(url, /search=hung/);
  assert.doesNotMatch(url, /categoryId/);
});

test("Vietnamese search is percent-encoded", () => {
  assert.match(buildProductQuery({ search: "điện" }), /search=%C4%91i%E1%BB%87n/);
});

console.log("order payload");

/** Hinh thai form o checkout: `name`, KHONG phai `customerName`. */
const buyer = {
  name: "  Nguyễn Văn A ",
  phone: " 0901 234 567 ",
  address: " 67 Lê Lợi ",
  note: "  ",
};

test("payload ships ids and quantities only — never a price", () => {
  const payload = buildOrderPayload(
    [{ productId: 7, quantity: 2, price: 99, name: "A" }],
    buyer
  );
  assert.deepEqual(payload, {
    // `name` cua form duoc doi thanh `customerName` o bien API — ten khac nhau
    // la co that, nen phai co mot test noi ro phep doi chieu nay.
    customerName: "Nguyễn Văn A",
    phone: "0901 234 567",
    address: "67 Lê Lợi",
    items: [{ productId: 7, quantity: 2 }],
  });
  // Giá nằm trong payload nghĩa là DevTools sửa được hoá đơn: phải không có chỗ nào
  // để "price" lọt vào, kể cả sau này ai đó thêm field vào line của cart.
  assert.equal(JSON.stringify(payload).includes("price"), false);
  assert.equal("note" in payload, false); // ghi chú rỗng thì không gửi
});

test("payload mang thong tin nguoi nhan, khong mang bat ky danh tinh nao", () => {
  // userId khong con la field cua OrderCreateRequest nua: server lay tai khoan
  // tu header Authorization. Nen neu mot dong gio hoac mot form lo ra "userId"
  // thi no PHAI bi bo lai buildOrderPayload, dung de co mot duong gui len.
  const stray = 4242;
  const payload = buildOrderPayload(
    [{ productId: 7, quantity: 1, userId: stray }],
    { ...buyer, userId: stray }
  );

  assert.equal("userId" in payload, false);
  assert.equal(JSON.stringify(payload).includes(String(stray)), false);
});

test("one cart line = one item; merging duplicates is the server's job", () => {
  const payload = buildOrderPayload(
    [
      { productId: 7, quantity: 2 },
      { productId: 7, quantity: 3 },
    ],
    buyer
  );
  assert.deepEqual(payload.items, [
    { productId: 7, quantity: 2 },
    { productId: 7, quantity: 3 },
  ]);
});

test("garbage ids and quantities are normalised, not forwarded", () => {
  const payload = buildOrderPayload(
    [
      { productId: 0, quantity: 5 },
      { productId: "8", quantity: "3" },
      { productId: 9, quantity: Number.NaN },
      { productId: null, quantity: 1 },
    ],
    buyer
  );
  assert.deepEqual(payload.items, [
    { productId: 8, quantity: 3 },
    { productId: 9, quantity: 1 },
  ]);
});

console.log("recipient validation");

test("accepts both phone shapes the backend @Pattern allows", () => {
  assert.deepEqual(
    validateRecipient({ name: "A", phone: "0901234567", address: "x", note: "" }),
    {}
  );
  assert.deepEqual(
    validateRecipient({ name: "A", phone: "+84 901 234 567", address: "x" }),
    {}
  );
});

test("blocks blanks, junk phone numbers and over-long text", () => {
  const errors = validateRecipient({
    name: "  ",
    phone: "abc",
    address: "",
    note: "x".repeat(501),
  });
  assert.ok(errors.name);
  assert.ok(errors.phone);
  assert.ok(errors.address);
  assert.ok(errors.note);
});

console.log("account form validation");

test("register form mirrors the backend @Size bounds", () => {
  assert.deepEqual(
    validateAccount({
      username: "hung.nguyen",
      fullName: "Nguyễn Văn Hùng",
      email: "hung@example.com",
      password: "secret123",
      confirmPassword: "secret123",
    }),
    {}
  );

  const errors = validateAccount({
    username: "hu",
    fullName: "",
    email: "not-an-email",
    password: "short",
    confirmPassword: "short2",
  });
  assert.ok(errors.username);
  assert.ok(errors.fullName);
  assert.ok(errors.email);
  assert.ok(errors.password);
  assert.ok(errors.confirmPassword);

  // BCrypt cat mat khau o 72 byte: phai chan tu phia form, khong de khach tuong
  // minh dang co mot mat khau dai hon the.
  assert.ok(validateAccount({ username: "abcd", fullName: "A B", email: "a@b.co",
    password: "x".repeat(80), confirmPassword: "x".repeat(80) }).password);
});

console.log("user session");

const TOKEN = "Zm9vYmFyLXRva2VuLXZhbHVlLTRkLWJ5dGVzLWFjdHVhbA";
const PROFILE = { id: 7, username: "hung", fullName: "Hùng", email: "h@x.co" };

test("a saved session reads back as token + profile", () => {
  assert.equal(isLoggedIn(), false);
  saveUserSession({ token: TOKEN, user: PROFILE });
  assert.equal(isLoggedIn(), true);
  assert.equal(getUserToken(), TOKEN);
  assert.equal(getUserProfile().id, 7);
  assert.equal(getUserSession().user.username, "hung");
  clearUserSession();
  assert.equal(getUserToken(), null);
  assert.equal(getUserProfile(), null);
});

test("tampered or hand-written storage is treated as signed out", () => {
  // token phai la chuoi dai du so cho that, con user phai co id nguyen duong +
  // username: hai thu do quyet dinh moi goi API co header Authorization hop le.
  const bad = [
    "{ not json",
    "null",
    "[]",
    // Kieu cu (profile khong token): doc duoc JSON lanh but khong goi duoc API
    // nao, nen thà coi là chưa đăng nhập còn hơn để UI hiện tên khách giả.
    JSON.stringify(PROFILE),
    JSON.stringify({ token: TOKEN }),
    JSON.stringify({ token: "", user: PROFILE }),
    JSON.stringify({ token: "short", user: PROFILE }),
    JSON.stringify({ token: TOKEN, user: { id: "7", username: "hung" } }),
    JSON.stringify({ token: TOKEN, user: { id: 0, username: "hung" } }),
    JSON.stringify({ token: TOKEN, user: { id: 7, username: "" } }),
  ];

  for (const raw of bad) {
    memory.set("shop.userSession.v2", raw);
    assert.equal(getUserSession(), null, `phải coi là chưa đăng nhập: ${raw}`);
    assert.equal(isLoggedIn(), false);
  }

  memory.delete("shop.userSession.v2");
});

test("a session without a usable token is refused instead of being stored", () => {
  // Dang nhap "thanh cong" ma khong luu duoc token thi lan cuoc sau bi tu choi:
  // that bai o day phai nem ra ngoai cho form hien len, khong im lang.
  assert.throws(() => saveUserSession({ user: PROFILE }));
  assert.throws(() => saveUserSession({ token: "abc", user: PROFILE }));
  assert.throws(() => saveUserSession(PROFILE));
  assert.equal(isLoggedIn(), false);
});

test("initials fall back through fullName then username", () => {
  assert.equal(getInitials({ fullName: "Nguyễn Văn A" }), "NA");
  assert.equal(getInitials({ username: "hung" }), "HU");
  assert.equal(getInitials(null), "KH");
});

console.log("order receipt");

test("receipt survives F5 through sessionStorage; tampered JSON is ignored", () => {
  saveOrderReceipt({ orderCode: "DH-000042", totalAmount: 1250000, items: [] });
  assert.equal(readOrderReceipt().orderCode, "DH-000042");
  sessionMemory.set("shop.lastOrder.v1", "{ not json");
  assert.equal(readOrderReceipt(), null);
});

console.log("api error mapping");

test("BusinessException body surfaces its message", () => {
  assert.equal(
    extractApiError({ message: "Sản phẩm A chỉ còn 5 cái" }, 400),
    "Sản phẩm A chỉ còn 5 cái"
  );
});

// GlobalExceptionHandler trả validation error dưới dạng Map<field, message> — KHÔNG
// có key "message", nên nếu chỉ đọc body.message thì form đặt hàng hiện ra đúng một
// câu "Không gọi được API (HTTP 400)" vô dụng.
test("field-error map becomes one readable line per field", () => {
  const text = extractApiError(
    {
      customerName: "Họ tên không được để trống",
      phone: "Số điện thoại không hợp lệ",
    },
    400
  );
  assert.deepEqual(text.split("\n"), [
    // Backend goi nguoi nhan la "customerName" (cot trong bang orders), client goi
    // la "name" — ca hai deu phai doc ra tieng nguoi, khong "customerName: ...".
    "Họ tên người nhận: Họ tên không được để trống",
    "Số điện thoại: Số điện thoại không hợp lệ",
  ]);
});

test("recipient-book and account errors read in Vietnamese too", () => {
  assert.equal(
    extractApiError({ name: "Tên người nhận không được để trống" }, 400),
    "Tên người nhận: Tên người nhận không được để trống"
  );
  // `userId` khong con trong FIELD_LABELS nua va cung khong con trong bat ky body
  // nao gui di (chu don do server lay tu token), nen API khong the tra ve field
  // error mang ten nay. Giu lai mot key la de chac chan no thoat ra nguyen ten
  // chu khong bi che duoi mot nhan tieng Viet da khong con y nghia.
  assert.equal(
    extractApiError({ userId: "Tài khoản đặt hàng không hợp lệ" }, 400),
    "userId: Tài khoản đặt hàng không hợp lệ"
  );
  assert.equal(
    extractApiError({ email: "Email không hợp lệ" }, 400),
    "Email: Email không hợp lệ"
  );
});

test("nested item errors fall back to their base field label", () => {
  assert.equal(
    extractApiError({ "items[1].quantity": "phải lớn hơn 0" }, 400),
    "Danh sách sản phẩm: phải lớn hơn 0"
  );
});

test("non-object or technical-only bodies keep the generic HTTP line", () => {
  assert.equal(extractApiError(null, 500), "Không gọi được API (HTTP 500)");
  assert.equal(
    extractApiError("Internal Server Error", 500),
    "Không gọi được API (HTTP 500)"
  );
  assert.equal(
    extractApiError({ timestamp: "2026-09-16T10:00:00", status: 404, error: "Not Found" }, 404),
    "Không gọi được API (HTTP 404)"
  );
});

console.log("orderFlow (timeline của khách, dựng từ dữ liệu server)");

test("a fresh order shows four steps with only PENDING passed", () => {
  const steps = buildSteps({ status: "PENDING", createdAt: "2026-09-17T08:00:00" });

  assert.deepEqual(
    steps.map((step) => step.status),
    ["PENDING", "CONFIRMED", "SHIPPING", "DELIVERED"]
  );
  assert.deepEqual(
    steps.map((step) => step.done),
    [true, false, false, false]
  );
  assert.deepEqual(
    steps.map((step) => step.at),
    ["2026-09-17T08:00:00", null, null, null]
  );
});

test("step labels come from the server payload, not from a second client dictionary", () => {
  const shipping = {
    status: "SHIPPING",
    statusLabel: "Đang vận chuyển",
    createdAt: "2026-09-17T08:00:00",
    history: [
      { toStatus: "CONFIRMED", toStatusLabel: "Chờ lấy hàng", createdAt: "2026-09-17T09:00:00" },
      { toStatus: "SHIPPING", toStatusLabel: "Đang vận chuyển", createdAt: "2026-09-17T10:00:00" },
    ],
  };

  assert.deepEqual(
    buildSteps(shipping).map((step) => step.label),
    ["Chờ xác nhận", "Chờ lấy hàng", "Đang vận chuyển", "Giao hàng thành công"]
  );
  assert.equal(orderStatusLabel(shipping), "Đang vận chuyển");
});

test("steps before the current one count as passed without inventing a timestamp", () => {
  const steps = buildSteps({
    status: "DELIVERED",
    createdAt: "2026-09-17T08:00:00",
    statusUpdatedAt: "2026-09-17T11:00:00",
  });

  assert.deepEqual(
    steps.map((step) => step.done),
    [true, true, true, true]
  );
  assert.deepEqual(
    steps.map((step) => step.at),
    ["2026-09-17T08:00:00", null, null, "2026-09-17T11:00:00"]
  );
});

test("a cancelled order keeps only PENDING as a passed step", () => {
  assert.equal(isCancelled({ status: "CANCELLED" }), true);
  assert.equal(isCancelled({ status: "PENDING" }), false);

  assert.deepEqual(
    buildSteps({ status: "CANCELLED", createdAt: "2026-09-17T08:00:00" }).map(
      (step) => step.done
    ),
    [true, false, false, false]
  );
});

test("cancelInfo reads the cancellation row instead of claiming the stock came back", () => {
  const cancel = cancelInfo({
    status: "CANCELLED",
    statusUpdatedAt: "2026-09-17T12:00:00",
    history: [
      { toStatus: "CONFIRMED", createdAt: "2026-09-17T09:00:00" },
      {
        toStatus: "CANCELLED",
        createdAt: "2026-09-17T10:30:00",
        note: "Không trả được kho: Điện thoại (sản phẩm đã bị xóa)",
      },
    ],
  });

  // Đúng dòng lịch sử chứ không phải statusUpdatedAt: dấu trạng thái bị các bước
  // sau đó ghi đè, còn dòng CANCELLED mới là lúc việc hủy xảy ra.
  assert.deepEqual(cancel, {
    at: "2026-09-17T10:30:00",
    note: "Không trả được kho: Điện thoại (sản phẩm đã bị xóa)",
  });

  // Đơn hủy đời trước, không còn dòng lịch sử nào.
  assert.deepEqual(
    cancelInfo({ status: "CANCELLED", statusUpdatedAt: "2026-09-17T12:00:00" }),
    { at: "2026-09-17T12:00:00", note: null }
  );

  assert.equal(cancelInfo({ status: "PENDING" }), null);
  assert.equal(cancelInfo(undefined), null);
});

test("the history row wins over the guess made from the current status", () => {
  const reached = reachedMap({
    status: "CONFIRMED",
    createdAt: "2026-09-17T08:00:00",
    statusUpdatedAt: "2026-09-17T12:00:00",
    history: [{ toStatus: "CONFIRMED", createdAt: "2026-09-17T09:00:00" }],
  });

  assert.equal(reached.get("CONFIRMED"), "2026-09-17T09:00:00");
});

test("an order without history, and an empty order, still render four steps", () => {
  assert.equal(buildSteps(undefined).length, 4);
  assert.deepEqual(
    buildSteps({}).map((step) => step.done),
    [false, false, false, false]
  );
});

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILED`);
process.exitCode = failures === 0 ? 0 : 1;
