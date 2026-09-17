package com.example.backend.security;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.HexFormat;

/**
 * Sinh token đăng nhập và đổi token ra giá trị mang đi so sánh.
 *
 * Tách ra khỏi SessionService vì dự án có HAI loại phiên — khách hàng (SessionService)
 * và quản trị (AdminSessionService). "Token là 256 bit ngẫu nhiên, DB chỉ lưu SHA-256"
 * là luật bảo mật, mà luật bảo mật có hai bản cài đặt thì kiểu gì cũng lệch: một bên
 * quên chặn độ dài, một bên đổi sang MD5 lúc nào không ai để ý. Ở đây chỉ một chỗ.
 *
 * Không phải bean Spring: không trạng thái, không cấu hình, gọi tĩnh cho gọn.
 */
public final class TokenCodec {

    /** 32 byte = 256 bit; base64url không padding -> 43 ký tự, an toàn trong header. */
    private static final int TOKEN_BYTES = 32;

    /** Chặn một request gửi vào chuỗi dài vô hạn chỉ để ăn phí hash. */
    public static final int MAX_TOKEN_LENGTH = 200;

    private static final HexFormat HEX = HexFormat.of();
    private static final SecureRandom RANDOM = new SecureRandom();

    private TokenCodec() {
    }

    /**
     * Token mới, xác suất trùng về số không. Trả về đúng một lần cho caller để gửi
     * cho client — đây là giá trị duy nhất không bao giờ được lưu lại.
     */
    public static String randomToken() {
        byte[] bytes = new byte[TOKEN_BYTES];
        RANDOM.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    /**
     * SHA-256 hex (64 ký tự thường) — giá trị đem lưu và đem so sánh.
     *
     * Không cần bcrypt ở đây: bcrypt chậm là vì mật khẩu do người chọn rất nghèo
     * nàn, dò hết không gian chỉ bằng cách thử. Token đã là 256 bit ngẫu nhiên đều
     * thì không có gì để dò, nên hash nhanh là đủ để kẻ đọc trộm bảng không dựng
     * lại được token hợp lệ.
     */
    public static String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HEX.formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            // SHA-256 nằm trong đặc tả của mọi JDK, nên đây là lỗi môi trường chứ
            // không phải nhánh chạy được của chương trình.
            throw new IllegalStateException("JDK không hỗ trợ SHA-256", e);
        }
    }

    /** Đúng định dạng token do randomToken() sinh ra; chữ thừa ở hai đầu được cắt. */
    public static boolean looksUsable(String token) {
        if (token == null) {
            return false;
        }
        String trimmed = token.trim();
        return !trimmed.isEmpty() && trimmed.length() <= MAX_TOKEN_LENGTH;
    }
}
