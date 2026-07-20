package com.example.backend.service;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import com.example.backend.dto.response.CloudinaryUploadResponse;
import com.example.backend.exception.BusinessException;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;
import java.util.Set;

@Service
public class CloudinaryService {

    private static final long MAX_FILE_SIZE = 5L * 1024 * 1024;

    private static final Set<String> ALLOWED_TYPES = Set.of(
            "image/jpeg",
            "image/png",
            "image/webp"
    );

    private final Cloudinary cloudinary;

    public CloudinaryService(Cloudinary cloudinary) {
        this.cloudinary = cloudinary;
    }

    public CloudinaryUploadResponse upload(MultipartFile file) {
        validate(file);

        try {
            Map<?, ?> result = cloudinary.uploader().upload(
                    file.getBytes(),
                    ObjectUtils.asMap(
                            "folder", "products",
                            "resource_type", "image"
                    )
            );

            return new CloudinaryUploadResponse(
                    String.valueOf(result.get("secure_url")),
                    String.valueOf(result.get("public_id"))
            );
        } catch (Exception exception) {
            throw new BusinessException("Upload ảnh thất bại");
        }
    }

    public void delete(String publicId) {
        if (publicId == null || publicId.isBlank()) {
            return;
        }

        try {
            cloudinary.uploader().destroy(
                    publicId,
                    ObjectUtils.asMap(
                            "resource_type", "image",
                            "invalidate", true
                    )
            );
        } catch (Exception exception) {
            throw new BusinessException("Xóa ảnh thất bại");
        }
    }

    public void deleteQuietly(String publicId) {
        try {
            delete(publicId);
        } catch (RuntimeException ignored) {
            // Best-effort cleanup: không che mất lỗi gốc.
        }
    }

    private void validate(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException("Ảnh không được để trống");
        }

        if (file.getSize() > MAX_FILE_SIZE) {
            throw new BusinessException("Ảnh không được vượt quá 5 MB");
        }

        String contentType = file.getContentType();

        if (contentType == null || !ALLOWED_TYPES.contains(contentType)) {
            throw new BusinessException(
                    "Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP"
            );
        }
    }
}
