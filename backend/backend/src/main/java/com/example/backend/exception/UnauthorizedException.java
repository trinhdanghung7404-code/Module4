package com.example.backend.exception;

/**
 * Caller không được biết ai là ai: thiếu token, token sai, hoặc token hết hạn.
 *
 * Tách khỏi BusinessException vì BusinessException map ra 400 — báo "yêu cầu của
 * bạn có dữ liệu sai". Ở đây dữ liệu không sai, chỉ là chưa được phép vào; client
 * cần đúng mã 401 để biết mà xoá phiên và đưa khách về màn đăng nhập thay vì hiện
 * "dữ liệu không hợp lệ" cho một lần hết hạn bình thường.
 */
public class UnauthorizedException extends RuntimeException {

    public UnauthorizedException(String message) {
        super(message);
    }
}
