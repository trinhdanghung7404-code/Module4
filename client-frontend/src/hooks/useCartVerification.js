import { useEffect, useMemo, useState } from "react";
import { fetchProduct } from "../api/shopApi";

/**
 * Đối chiếu giỏ hàng với dữ liệu sống trên server.
 *
 * Tách khỏi CartPage vì trang thanh toán phải gate đúng một điều kiện như vậy:
 * chưa đối chiếu thì chưa cho gửi đơn. Giá trong cart chỉ là snapshot từ
 * localStorage; server tính lại tiền từ DB, nên người dùng cần thấy số của server
 * trước khi bấm đặt hàng chứ không phải nhận lỗi sau khi gửi.
 *
 * Trả về:
 *  - membership: "key" của tập product id hiện tại, dạng "7|8"
 *  - isVerified: đã đối chiếu xong cho đúng tập id hiện tại
 *  - validated:  { key, removed, unreachable } của lần đối chiếu gần nhất
 */
export function useCartVerification(lines, syncWithProducts) {
  const [validated, setValidated] = useState(null);

  // Only the cart *membership* triggers a catalog lookup — editing a quantity must not
  // fire a request per keystroke.
  const membership = useMemo(
    () => lines.map((line) => line.productId).join("|"),
    [lines]
  );

  useEffect(() => {
    if (!membership) return undefined;

    let cancelled = false;
    const ids = membership.split("|").map(Number);

    Promise.allSettled(ids.map((productId) => fetchProduct(productId))).then(
      (results) => {
        if (cancelled) return;

        const found = [];
        let unreachable = 0;

        results.forEach((result) => {
          if (result.status === "fulfilled") {
            found.push(result.value);
            return;
          }
          // A 4xx answer means the server definitively refuses this product id, so the
          // line is dropped. The range check (not `=== 404`) is required because this
          // backend maps "not found" to 400 via GlobalExceptionHandler + BusinessException.
          // Anything else — backend down, CORS block, 5xx — is only "unknown", and
          // unknown must never be silently turned into "delete the user's cart line".
          const status = result.reason?.status;
          if (status >= 400 && status < 500) return;
          unreachable += 1;
        });

        const removed = syncWithProducts(found, {
          pruneMissing: unreachable === 0,
        });

        setValidated({ key: membership, removed: removed.length, unreachable });
      }
    );

    return () => {
      cancelled = true;
    };
  }, [membership, syncWithProducts]);

  // "Checking" is derived: verification has not settled for the *current* set of ids.
  return { membership, validated, isVerified: validated?.key === membership };
}
