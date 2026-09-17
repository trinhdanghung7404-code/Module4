 client-frontend

Cửa hàng công khai (shop) đọc từ backend `/api/products` + `/api/categories`.
Tách khỏi `admin-frontend` để hai bên không đụng route, không đụng CSS.

## Chạy

```bash
npm install
npm run dev      # http://localhost:5174
```

Cần backend Spring Boot chạy ở `http://localhost:8080` (xem `../backend`).
Địa chỉ API lấy từ `.env` → `VITE_API_URL`.

```bash
npm run build    # production bundle vào dist/
npm run lint     # ESLint (react-hooks v7, rule set-state-in-effect đang bật)
npm run verify   # 37 kiểm tra logic thuần: cart, query builder, order payload, phiên khách, error map, timeline đơn
```

## Port

`vite.config.js` **ghim cứng 5174** (`strictPort: true`). Danh sách CORS của backend
(`app.cors.allowed-origins` trong `application.properties`) chỉ có `5173` và `5174`;
nếu Vite tự nhảy sang port khác thì mọi response `/api/**` bị trình duyệt chặn và
look like a backend bug.

## Cấu trúc

```
src/
  api/shopApi.js          fetch + buildProductQuery + createOrder + extractApiError
                          (một chỗ duy nhất gắn Authorization và một chỗ clear session khi 401)
  api/userApi.js          đăng ký / đăng nhập / logout / hồ sơ / sổ người nhận theo token
  utils/cartStore.js      localStorage cart, external store
  utils/orderPayload.js   buildOrderPayload / validateRecipient / validateAccount + receipt
  utils/orderFlow.js      nhãn trạng thái + timeline 4 mốc, dựng từ history do server gửi
  utils/userSession.js    phiên khách trong localStorage + event "shop:user-session"
  utils/shopUiStore.js    category list + filter/page, external store
  utils/format.js         VND price, date
  hooks/useCart.js        useSyncExternalStore wrapper
  hooks/useCartVerification.js  đối chiếu giỏ với server (Cart + Checkout dùng chung)
  hooks/useUser.js        phiên đăng nhập, tự cập nhật khi tab khác đổi localStorage
  hooks/useRecipients.js  sổ người nhận của một tài khoản
  hooks/useOrders.js      đơn của tài khoản đang đăng nhập (GET /api/orders/mine)
  hooks/useShopUi.js      ""
  components/             ShopLayout, ShopHeader, ShopFooter, ProductCard,
                          ProductImage, Pagination, StatusPanel, Price,
                          Field (ô nhập + lỗi dùng chung), AuthShell, RequireUser,
                          ShopAccountMenu (dropdown tài khoản trên header),
                          OrderHistory (khung danh sách "Đơn hàng của tôi"),
                          OrderCard (một thẻ đơn: timeline 4 mốc + người nhận + tổng tiền)
  pages/                  ShopPage, ProductDetailPage, CartPage, CheckoutPage,
                          OrderSuccessPage, LoginPage, RegisterPage,
                          AccountPage (menu ?tab=: sổ người nhận | đơn hàng),
                          NotFoundPage
```

Không dùng React Context: state dùng external store + `useSyncExternalStore`, nên
header và grid tự đồng bộ, và file chỉ export component (fast refresh sạch).

## Hợp đồng API đã được kiểm chứng bằng chạy thật

| Endpoint | Ghi chú |
| --- | --- |
| `GET /api/products` | envelope `{content, page, size, totalElements, totalPages, last}` |
 | | card chỉ có `id, name, price, quantity, categoryName, imageUrl` |
| `GET /api/products/{id}` | `ProductResponse` đầy đủ: `description, sku, createdAt, images[]` |
| `GET /api/categories` | `[{id, name, productCount}]` |
| `POST /api/orders` | **cần đăng nhập**. **201** → `{orderCode, totalAmount, status, items[]}`; **400** → hoặc `{message}` (business) hoặc `{field: message}` (validation); **401** → `{message}` |
| `POST /api/users/register` | **201** → `{token, user:{id, username, fullName, email, createdAt}}` — **không** có mật khẩu. Đăng ký xong là có phiên luôn, không phải nhập lại |
| `POST /api/users/login` | **200** → `{token, user}`. Sai mật khẩu hay không có tài khoản đều trả **cùng một thông báo** để không dò được email nào đã đăng ký |
| `POST /api/users/logout` | **204**, thu hồi đúng token vừa gửi. Token lạ/hết hạn cũng im lặng — nên client cứ clear `localStorage` bất kể kết quả |
| `GET /api/users/me` | profile của tài khoản **suy ra từ token**; **401** khi thiếu token hoặc phiên đã hết hạn |
| `GET\|POST /api/users/me/recipients` | sổ người nhận; người đầu tiên tự thành mặc định; trùng lặp → 400 |
| `PUT\|DELETE /api/users/me/recipients/{rid}` | sửa / xoá. Xoá người mặc định thì người kế được đẩy lên. `PUT .../{rid}/default` đổi mặc định và trả cả danh sách mới |
| `GET /api/orders/mine` | đơn của chính tài khoản đang gọi, mới nhất trước; mỗi đơn kèm `statusLabel`, `nextStatuses`, `shippingUnit`, `trackingCode`, `statusUpdatedAt` và `history[]` (`fromStatus/toStatus/toStatusLabel/actorType/actorLabel/note/createdAt`) — xem `dto/response/OrderResponse.java`. **Không có** `?userId=` để đổi chỗ đoán id lấy được cả tên+SĐT+địa chỉ trong đơn |

* Mọi đường dẫn thuộc về khách đều mang header `Authorization: Bearer <token>`; token do
  `POST /users/register|login` trả về và được client cất trong `localStorage`
  (`shop.userSession.v2`). Không còn endpoint nào dạng `/api/users/{id}/...`.
* Token là chuỗi opaque 43 ký tự (32 byte `SecureRandom`, base64url). DB chỉ lưu **SHA-256**
  của token trong bảng `user_session` cùng `expires_at` = hiện tại +
  `app.user.session-ttl-days` (30). Hết hạn/đăng xuất là **xoá dòng**, nên token chết ngay
  lập tức; đổi lại mỗi request phải tra DB một lần.
* Một tài khoản có nhiều phiên song song: đăng xuất ở máy này không đá khách ở máy khác.
* Interceptor (`UserSessionInterceptor`, đăng ký trong `WebConfig`) chặn
  `/api/users/**` (trừ `login`/`register`), `/api/orders` và `/api/orders/**`. Nó là
  `HandlerInterceptor` chứ không phải filter để 401 vẫn nằm sau bộ máy CORS — thiếu
  `Access-Control-Allow-Origin` thì trình duyệt chỉ báo "không gọi được máy chủ" thay vì
  cho client biết phiên đã hết hạn.

* `page` bắt đầu từ **0**.
* `size` mặc định 12, backend **chặn tối đa 48**.
* `sort` whitelist: `newest, oldest, price-asc, price-desc, name-asc, name-desc, quantity-asc, quantity-desc`. Giá trị lạ **không báo lỗi**, backend tự rơi về `newest`.
* Ảnh đại diện lấy theo `imageType = FRONT`; sản phẩm chưa có ảnh FRONT → `imageUrl: null`.
* `POST /api/orders` trả `201` kèm `orderCode` dạng `DH-000001`; dòng trùng sản phẩm được
  cộng gộp **trước** khi kiểm tra tồn kho; trừ kho nguyên tử cấp DB và rollback cả đơn
  nếu một sản phẩm không đủ hàng.

## Khối tài khoản trên header

`components/ShopAccountMenu.jsx` thay cho hàng nút "avatar + tên + Đăng xuất" trước
đây: nút bấm chỉ còn avatar + tên (tên bị ẩn dưới 720px để hết cắt "B22DCVT230_Trịnh …"),
mọi nội dung khác nằm trong dropdown mở ra từ chính nút đó. Chưa đăng nhập thì khối
này vẫn là hai đường dẫn "Đăng nhập / Đăng ký" như cũ.

| phần trong dropdown | nội dung | nguồn dữ liệu |
| --- | --- | --- |
| summary | tên đầy đủ + email | `userSession` — `{token, user}` do `/users/login` trả về |
| details | username, "Tham gia từ" | `user.username`, `formatDate(user.createdAt)` |
| links | "Tài khoản của tôi" → `/account`, "Đơn hàng của tôi" → `/account?tab=orders` | — |
| hành động | **Đăng xuất** | `useUser().logout()` |

Bốn điểm cần biết trước khi sửa tiếp:

* **Giỏ hàng không nằm trong dropdown.** Nó đã là một nút riêng của header, ngay cạnh
  khối tài khoản, và khách chưa đăng nhập cũng phải bấm được. Lặp lại nó trong menu
  chỉ thêm một chỗ phải nhớ cập nhật mà chẳng thêm được đường đi nào.
* **"Đơn hàng của tôi" thì có**, vì nó không có nút nào khác trên header và là một
  đường đi thật. Link trỏ thẳng `?tab=orders` để vào đúng mục, khỏi cuộn qua sổ người
  nhận — cùng một trang `/account`, khác mục.

* Đăng xuất vẫn do `useUser` quyết định thứ tự (gọi `/users/logout` thu hồi token
  trước, xoá `localStorage` sau). Component chỉ gọi `logout()`, rồi đưa khách về `/`
  nếu đang đứng ở `/checkout` hoặc `/account` — hai trang ấy không còn nghĩa gì khi
  phiên vừa bị xoá. `AccountPage` **không** lặp lại nút Đăng xuất nữa: một hành động
  đặt hai chỗ cạnh nhau khiến khách đoán là hai việc khác nhau.
* "Mở hay đóng" là giá trị **suy ra** (`openForPath === location.pathname`), không
  phải effect gọi `setOpen(false)` mỗi lần đổi trang — react-hooks v7 chặn
  `set-state-in-effect`. Bấm link trong menu vẫn có `closeMenu()` tường minh, vì đứng
  ở `/account` mà bấm "Tài khoản của tôi" thì pathname không đổi và menu sẽ treo lại.

## Khi hiện "Không gọi được ... từ origin ..."

Câu đó chỉ được sinh ra ở **một chỗ duy nhất**: `request()` trong `src/api/shopApi.js`,
`catch` của `fetch`. Nghĩa là **trình duyệt không đọc được response**, chứ KHÔNG phải
backend trả 4xx/5xx — một cú 401 hay 400 luôn có message riêng và luôn hiện đúng message
đó. Ba nguyên nhân thật:

1. Backend chưa chạy, hoặc `VITE_API_URL` trỏ sai cổng (`.env` là `http://localhost:8080/api`;
   `8081`/`8082` chỉ là instance smoke test).
2. **Origin của tab đang mở không nằm trong `app.cors.allowed-origins`.** CORS so sánh
   nguyên chuỗi host+port: `http://127.0.0.1:5174` ≠ `http://localhost:5174`, và cổng
   5175 trở đi (Vite tự nhảy khi 5173/5174 đã bị chiếm) cũng không được chấp nhận.
3. **Preflight `OPTIONS` bị chặn.** Đây là cái bẫy đắt nhất, vì nó chỉ xảy ra trong
   trình duyệt: một request mang header `Authorization` không còn là "simple request",
   nên browser gửi `OPTIONS` trước và **chỉ gửi tiếp khi `OPTIONS` trả về 2xx**. Bản
   `OPTIONS` đó không bao giờ mang `Authorization` (nó chỉ liệt kê header sắp gửi trong
   `Access-Control-Request-Headers`), mà Spring lại **giữ nguyên interceptor chain** khi
   xử lý preflight (`AbstractHandlerMapping#getCorsHandlerExecutionChain`). Kết quả:
   `UserSessionInterceptor` chặn preflight bằng 401, browser báo "không gọi được máy
   chủ", và mọi endpoint khách đều chết — **trong khi `/users/login` vẫn chạy** vì nó
   nằm ngoài interceptor. Dấu hiệu nhận biết đúng bệnh này: đăng nhập được, khối "Tài
   khoản đặt hàng" vẫn hiện (đọc `localStorage`, không cần mạng), nhưng mọi gọi API sau
   đó đều đỏ. Fix: `CorsUtils.isPreFlightRequest(request)` → cho qua, nằm trong
   `UserSessionInterceptor#preHandle`.

Vì sao smoke test không bắt được: PowerShell gọi thẳng GET/POST, **không bao giờ gửi
preflight**. Nên `backend/backend/scripts/user-check.ps1` giờ có hẳn một mục
`== preflight CORS ==` gửi `OPTIONS` thật tới `/users/me`, `/orders`, `/orders/mine`,
`/users/me/recipients`, `/users/login` và đòi 2xx kèm `Access-Control-Allow-Origin`,
cả chiều ngược lại (origin lạ phải bị từ chối và không được echo header).

Thao tác chẩn đoán 30 giây:

* DevTools → Network: thấy một dòng `OPTIONS` đỏ → đúng nguyên nhân 3 (hoặc 2).
* Nhìn thanh địa chỉ: có phải `http://localhost:5173|5174` không → nguyên nhân 2.
* `curl -i -X OPTIONS http://localhost:8080/api/users/me -H "Origin: http://localhost:5174" -H "Access-Control-Request-Method: GET" -H "Access-Control-Request-Headers: authorization"`
  → phải là `200`, không phải `401/500`.
* Sửa Java xong là phải **restart backend**; `npm run dev` không liên quan tới code backend.


## Cart

* Lưu `localStorage` key `shop.cart.v1`, mỗi dòng kèm **snapshot** `name/price/imageUrl`.
  → giỏ vẫn hiển thị khi API chết, nhưng giá **không tự cập nhật** cho tới khi trang
  Giỏ hàng đối chiếu lại.
* Số lượng bị **chặn theo tồn kho**; gõ rỗng/0 không xoá dòng (chỉ nút × mới xoá).
* Hai tab tự đồng bộ qua event `storage`.
* Trang Giỏ hàng đối chiếu lại từng `productId`; sản phẩm bị admin xoá thì bị gỡ khỏi giỏ.
  Lỗi mạng/5xx chỉ được báo "không kiểm tra được", **không** tự ý xoá dòng.
* **Checkout** `/checkout` cần đăng nhập (`RequireUser`), và chia làm hai khối tách biệt:
  * **Tài khoản đặt hàng** — chỉ hiển thị, không nhập: đơn được gắn `orders.user_id`.
  * **Người nhận hàng** — chọn từ sổ đã lưu (bấm một cái là điền xong tên/SĐT/địa chỉ),
    hoặc "＋ Người nhận khác" để nhập tay. Sửa thông tin của người đã chọn thì hiện
    checkbox "Lưu người nhận này vào sổ".
* Body đặt hàng: `{customerName, phone, address, note?, items:[{productId, quantity}]}`
  — **không gửi giá** (server tính lại từ DB; test thật: đổi giá trong DB thì tổng tiền vẫn
  theo DB) và **không gửi `userId`**: chủ đơn do server lấy từ token. `buildOrderPayload`
  lọc bỏ mọi `userId`/`price`/`lineTotal`/`totalAmount` lọt vào từ cart hay từ form, nên
  không còn đường nào để client tự khai đơn của mình là của người khác.
* Trang **Tài khoản** (`/account`) là một **menu, mỗi mục một màn** — không xếp chồng
  mọi khối xuống cùng một trang dài như trước:

  | mục (`?tab=`) | nội dung | nguồn dữ liệu |
  | --- | --- | --- |
  | `recipients` (mặc định) | sổ người nhận + form thêm/sửa | `/api/recipient` |
  | `orders` | **Đơn hàng của tôi** (`OrderHistory`) | 50 đơn gần nhất từ `/api/orders/mine` |

  Danh sách mục khai báo một lần ở mảng `TABS` trong `pages/AccountPage.jsx`; thêm mục
  mới là thêm một dòng `TABS` + một nhánh render. Mục đang xem đọc từ URL bằng
  `useSearchParams` (không phải `useState`), nên chia sẻ link, F5 và nút quay lại của
  trình duyệt vẫn giữ đúng mục; `?tab=<lạ>` rơi về `recipients` thay vì màn trắng.
  Chưa có tra đơn theo mã công khai.
* Trong app, "người nhận" luôn là `{name, phone, address}`; chỉ đúng một `name` được đổi
  thành `customerName` khi dựng body (`buildOrderPayload`), nên sổ người nhận và form đặt
  hàng dùng chung được một validator.
* Ghi chú thuộc về **đơn**, không thuộc về người nhận — nên nó đứng ngoài sổ người nhận.
* Nút "Thanh toán" trong giỏ và nút "Đặt hàng" đều **khoá cho tới khi đối chiếu xong**
  với server (`useCartVerification`), vì giá trong `localStorage` chỉ là snapshot.
* Giỏ hàng **chỉ bị xoá sau khi nhận 201**; đơn bị từ chối (thiếu field / hết hàng) thì
  giỏ giữ nguyên để khách sửa rồi đặt lại.
* `/order/success` đọc biên lai từ `sessionStorage` (F5 vẫn còn, đóng tab là mất) — chỉ là
  bản ghi phía người dùng, nguồn thật nằm trong bảng `orders`.

## Nhãn trạng thái và timeline trong "Đơn hàng của tôi"

Toàn bộ logic nằm trong `src/utils/orderFlow.js`. `OrderCard.jsx` (một thẻ đơn, tách
khỏi `OrderHistory.jsx` là khung danh sách) chỉ gọi `statusLabel`,
`buildSteps` và `cancelInfo` rồi vẽ; `isCancelled` vẫn được export vì `cancelInfo` dùng nó
và verify gọi thẳng, nhưng UI thì không tự kiểm tra trạng thái nữa. CSS của cả khối
nằm riêng ở `src/styles/orders.css`; menu mục (`?tab=`) ở trên thì nằm trong
`src/styles/account.css` (`.account__menu`, `.account__menu-item.is-active`,
`.account__panel`).

* **Nhãn do server gửi**: `history[].toStatusLabel` cho từng bước đã qua, `statusLabel` cho
  trạng thái hiện tại. `FALLBACK_LABELS` trong file chỉ là ô đỡ cho dữ liệu **không có** lịch
  sử, và được chép nguyên văn từ `OrderStatusRules.labels()` ở backend — hai bản dịch lệch
  nhau là cách chắc nhất để khách nghĩ đơn mình bị chậm.
* **Bốn mốc luôn hiện**, kể cả mốc chưa tới ("Chờ xác nhận → Chờ lấy hàng → Đang vận chuyển →
  Giao hàng thành công"). Ẩn mốc tương lai thì khách không còn biết đơn mình đang ở đâu trong
  luồng, nên chúng được đánh dấu "Chưa tới" thay vì bị bỏ.
* **Mốc đã qua suy từ `history`**, không suy từ trí nhớ: đơn ở `SHIPPING` mà lịch sử chỉ có
  một dòng `PENDING → SHIPPING` (admin xác nhận miệng rồi bấm luôn bàn giao) thì các mốc đứng
  trước vẫn được tính là đã qua, vì đơn **chỉ đi một chiều** theo `OrderStatusRules`.
* **Không bịa timestamp.** Mốc đã qua mà không có dòng lịch sử nào ghi lại thì để trống, không
  lấy `createdAt` của đơn gán cho bước giữa luồng — làm vậy là cho khách một giờ cụ thể chưa
  từng tồn tại ở đâu cả.
* **`CANCELLED` thì không vẽ timeline**: bốn mốc "Chưa tới" phía sau một đơn đã hủy chỉ là chữ
  vô nghĩa, thay bằng dòng "Đơn đã hủy <ngày>" + ghi chú của bước hủy (`cancelInfo(order)` đọc
  đúng dòng `CANCELLED` trong lịch sử, không lấy `statusUpdatedAt` khi lịch sử còn nguyên).
  Client **không tự khẳng định "hàng đã quay lại kho"**: server trả hàng lúc hủy, nhưng sản
  phẩm đã bị xoá vật lý thì không còn gì để trả, và server ghi thẳng điều đó vào ghi chú bước
  hủy (`AdminOrderService#restoreStock`) — khách đọc được đúng sự thật, không đọc được lời hứa.
* `nextStatuses` có trong payload nhưng **không có nút đổi trạng thái cho khách** — khách không
  có quyền đó; nếu sau này cho khách tự hủy thì vẫn phải hỏi server, đừng lặp luật ở client.

## Hạn chế đang biết

1. Backend **không có cột ẩn/hiện sản phẩm**, nên mọi sản phẩm trong DB đều hiện, gồm cả
   sản phẩm test (`hung`, `jiji`).
2. Tìm kiếm không phân biệt hoa/thường nhưng **không bỏ dấu**: `dien thoai` không ra
   `Điện thoại`. Cần PostgreSQL `unaccent` hoặc chuẩn hoá chuỗi nếu muốn bỏ dấu.
3. Backend trả **400** (không phải 404) cho "không tìm thấy", vì `GlobalExceptionHandler`
   map mọi `BusinessException` về `badRequest()`. Client đang xử theo khoảng 4xx.
4. Error response ở chế độ dev có thể kèm `trace`. (Admin endpoints **đã** có auth riêng từ
   vòng luồng đơn: `security/AdminSessionInterceptor.java` chặn mọi `/api/admin/**`, token
   do `POST /api/admin/login` trả và cất trong bảng `admin_session`.)
5. **Xác thực khách đã thật, nhưng mới ở mức tối thiểu.** Token opaque + bảng `user_session`
   + `HandlerInterceptor`: `POST /api/orders` và `/api/users/**` trả **401** khi thiếu token,
   chủ đơn do server quyết (smoke test: `backend/backend/scripts/user-check.ps1`). Còn thiếu:
   * `localStorage` thì JS nào cũng đọc được → một lỗi XSS ăn trọn token. Muốn chặn hẳn phải
     chuyển sang cookie `HttpOnly` + `SameSite` (kèm CSRF token);
   * chưa có **đổi mật khẩu**. Khi thêm thì phải xoá luôn các dòng `user_session` của tài
     khoản đó, không thì token cũ vẫn sống sau khi khách đổi mật khẩu vì bị đánh cắp;
   * `login` **không giới hạn số lần thử**, nên dò mật khẩu chậm không bị chặn;
   * interceptor của khách **không che `/api/admin/**`** — admin đi chain riêng
     (`AdminSessionInterceptor`, mục 4); hai chain độc lập nên một token khách không có cửa
     nào gọi vào endpoint admin và ngược lại, nhưng cũng **không có** khái niệm quyền nào
     ngoài "đúng chủ của dữ liệu" / "đúng admin đã đăng nhập";
   * mỗi request tra DB một lần để đổi token ra user. Muốn bỏ chi phí đó thì cần cache, và
     cache làm token thu hồi còn sống tới khi cache hết hạn — đánh đổi, không phải free.
6. Vì `orderCode` đánh số theo `id` nên **đừng** thêm endpoint GET tra đơn theo mã cho người
   chưa đăng nhập: `DH-000042` kể cho người gọi biết mã của người bên cạnh là gì, và một
   đơn thì kèm cả tên + SĐT + địa chỉ nhận hàng. Khách chỉ xem được đơn của mình qua
   `/api/orders/mine`. Muốn có "tra đơn bằng mã" thì phải đổi mã sang dạng không đoán được
   (UUID/base36 ngẫu nhiên) trước đã.
7. Chưa có cổng thanh toán thật: đơn chỉ được ghi nhận `PENDING`, không tiền nào bị trừ.
