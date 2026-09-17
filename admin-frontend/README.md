# admin-frontend

Trang quản trị (React + Vite): đăng nhập admin, sản phẩm, danh mục, và xử lý đơn
hàng. Mọi số liệu lấy từ `/api/admin/**` của backend Spring Boot
(`../backend/backend`) — trang này không giữ nghiệp vụ nào trong đầu.

## Chạy

```powershell
npm install
npm run dev      # http://localhost:5173
```

| lệnh | tác dụng |
| --- | --- |
| `npm run dev` | dev server, mặc định cổng 5173 |
| `npm run build` | bundle vào `dist/` |
| `npm run lint` | ESLint (react-hooks v7 + react-refresh) |
| `npm run preview` | xem thử bản `dist/` |

`VITE_API_URL` trong `.env` là địa chỉ gốc của API admin, mặc định
`http://localhost:8080/api/admin`. Backend phải chạy trước (IntelliJ, hay
`../backend/backend/scripts/run-backend.ps1`).

**Đừng đổi cổng tuỳ hứng.** Danh sách CORS backend chấp nhận
(`app.cors.allowed-origins` trong `application.properties`) chỉ có
`http://localhost:5173` và `http://localhost:5174`. Chạy cổng khác thì trình duyệt
chặn response, và câu lỗi hiện ra trông hệt như lỗi backend.

## Phiên đăng nhập

`src/utils/adminSession.js` cất **một** object trong `localStorage` dưới key
`adminSession`: hồ sơ admin kèm `token` do `POST /admin/login` trả về. Không có
token là chưa đăng nhập. Hệ quả có chủ đích: phiên do bản cũ lưu (chỉ có hồ sơ,
không token) bị từ chối và admin phải đăng nhập lại một lần — cái hồ sơ đó không
chứng minh được gì với server, mà bản cũ thì lại tin nó.

`src/api/client.js` là chỗ **duy nhất** gọi HTTP:

* gắn `Authorization: Bearer <token>` vào mọi request — trước đây mỗi file api tự
  viết `fetch` riêng, và chỉ cần một file quên gắn header là trang đó báo lỗi khó
  hiểu trong khi server đòi token cho mọi đường `/api/admin/**`;
* gặp **401** thì xoá phiên ngay và bắn event `admin:unauthorized` để
  `AdminLayout` đưa về màn đăng nhập, thay vì để từng trang tự lo;
* `extractMessage()` ưu tiên `message` của `GlobalExceptionHandler` (tiếng Việt),
  rồi mới tới `error` của body mặc định Spring — nên lỗi nghiệp vụ hiện nguyên câu
  server viết, không bị dịch lại ở client.

Thêm endpoint mới thì dùng `request()` trong `src/api/*Api.js`; đừng viết `fetch`
ở trong component.

> Đã biết: token nằm trong `localStorage` nghĩa là bất kỳ script nào chạy trên
> trang đọc được nó, nên một lỗi XSS ở trang quản trị là ăn trọn token. Chặn hẳn
> phải chuyển sang cookie `HttpOnly` + `SameSite` kèm CSRF token — việc lớn hơn
> nhiều so với những gì vòng này làm.

## Đơn hàng

| file | việc |
| --- | --- |
| `src/api/orderApi.js` | danh sách (lọc theo trạng thái, tìm theo mã đơn/tên/SĐT, phân trang), số đếm theo trạng thái, chi tiết một đơn, đổi trạng thái |
| `src/pages/OrdersPage.jsx` | bảng đơn, chip đếm theo trạng thái, nút đổi trạng thái |
| `src/components/OrderDetailModal.jsx` | lịch sử từng bước + đổi trạng thái cho một đơn |

Máy trạng thái **nằm ở server** (`service/OrderStatusRules.java`). Mỗi đơn trả về
kèm `nextStatuses` — những bước hợp lệ tiếp theo — và UI chỉ vẽ nút từ danh sách
đó. Nhãn trạng thái cũng lấy từ payload (`statusLabel`, `history[].toStatusLabel`,
`summary.labels`); `FALLBACK_LABELS` trong hai file kia chỉ là chỗ đỡ khi số liệu
chưa về kịp, và được chép nguyên văn từ `OrderStatusRules.labels()` để hai bên
khỏi gọi tên khác nhau cho cùng một trạng thái.

Hai chỗ luật của server mà UI phải phụ họa theo, không được tự đoán:

* Chuyển sang **Đang vận chuyển** bắt buộc có `shippingUnit`, và server trả **400**
  kèm nguyên văn *"Phải ghi rõ đơn vị vận chuyển trước khi bàn giao hàng."* khi thiếu
  — modal hiển lại đúng câu đó. Tên đơn vị là chuỗi tự do (`@Size(max = 100)`), backend
  **không** có danh sách nhà vận chuyển nào để chọn, nên không có dropdown nào ở đây;
  `trackingCode` (mã vận đơn, `max = 60`) là tuỳ chọn và khách thấy nó trong đơn.
  Ở các bước khác, gửi hai trường này lên cũng bị **400**
  (*"Chỉ nhập đơn vị vận chuyển và mã vận đơn ở bước Đang vận chuyển."*) — server không
  âm thầm bỏ qua dữ liệu admin đã gõ.
* **Đã hủy** trả hàng về kho và chỉ trả **một lần**; đơn đã `DELIVERED` hoặc đã
  `CANCELLED` là hết đường đổi (`nextStatuses` rỗng → không có nút nào hiện ra).
  Hàng hoàn không lấy lại được (sản phẩm đã bị xoá khỏi `product`) thì server ghi thẳng
  vào note của bước hủy, và cột "Ghi chú gần nhất" của bảng đơn là chỗ duy nhất để nhìn
  thấy dòng đó — đừng xoá nó khi sửa bảng.

## Modal sản phẩm

`ProductPage` chỉ render `ProductModal` **khi mở**, kèm `key` gồm chế độ + id sản phẩm
(xem dòng `key=` chỗ render `ProductModal`). Mỗi lần mở vì thế là một component
hoàn toàn mới, form khởi tạo thẳng từ props, không cần state reset.

Trước đây modal sống thường trực và một `useEffect` reset state mỗi lần `open` đổi — cách đó
bắt React render hai lần cho một lần mở trên một file gần 600 dòng, và bị
`react-hooks/set-state-in-effect` bắt lỗi. Luật đó không vô duyên: nó báo đúng chỗ setState
đồng bộ trong effect là thừa.

## Kiểm tra

```powershell
npm run lint
npm run build
```

Đường đi thật của một đơn (đăng nhập → danh sách → lọc/tìm → chi tiết → đổi trạng
thái → hoàn kho → phân quyền 401/403 → preflight CORS) được kiểm bằng script smoke
chạy **với backend đang bật**:

```powershell
cd ..\backend\backend\scripts
$env:SMOKE_API_BASE = 'http://localhost:8080/api'
powershell -NoProfile -ExecutionPolicy Bypass -File .\admin-order-flow.ps1
```

Script tạo admin tạm `smokeadm*` và đơn probe, rồi tự xoá khi kết thúc; nếu nó chết
giữa chừng thì còn lại vài dòng trong DB, xoá bằng `psql`:

```powershell
$env:PGPASSWORD='123'; & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost -d shop_management_db -c "delete from admin where username like 'smokeadm%'"
```

Phía khách có `.\user-check.ps1` (đăng ký/đăng nhập/sổ người nhận/đặt đơn + preflight
CORS), và `.\cleanup-probe.ps1` trả lại tồn kho cho các đơn probe của script đó.

Cần dữ liệu để sửa UI thì `.\seed-products.ps1` chèn 20 sản phẩm `SEED-DT-01…20` vào
danh mục đầu tiên, ảnh lấy từ `picsum.photos/seed/<sku>` (chạy lại không sinh bản thứ
hai). `.\seed-products.ps1 -Clean` xoá đúng các dòng `SEED-DT-%`; ảnh là URL ngoài nên
không upload gì lên Cloudinary và không chiếm dung lượng repo.

