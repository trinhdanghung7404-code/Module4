package com.example.backend.service;

import com.example.backend.dto.RecipientRequest;
import com.example.backend.dto.UserLoginRequest;
import com.example.backend.dto.UserRegisterRequest;
import com.example.backend.dto.response.RecipientResponse;
import com.example.backend.dto.response.UserAuthResponse;
import com.example.backend.dto.response.UserProfileResponse;
import com.example.backend.entity.Recipient;
import com.example.backend.entity.User;
import com.example.backend.exception.BusinessException;
import com.example.backend.repository.RecipientRepository;
import com.example.backend.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Locale;

/**
 * Tài khoản khách + sổ người nhận hàng.
 *
 * KY LUAT khi cham vao Recipient: repository.clearDefaults() la JPQL UPDATE co
 * clearAutomatically, nghia la no XOA SACH persistence context ngay luc chay.
 * Moi entity doc truoc do tro thanh detached va sua no se khong duoc ghi xuong
 * DB, nen sau goi do luc nao cung phai doc lai entity.
 */
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final RecipientRepository recipientRepository;
    private final PasswordEncoder passwordEncoder;
    private final SessionService sessionService;

    /**
     * Đăng ký. Trả về cả token lẫn profile — không chỉ báo "thành công" như phía
     * admin: khách vừa bấm đăng ký là đang đứng trước giỏ hàng, bắt nhập lại mật
     * khẩu lần nữa thì mất đơn.
     */
    @Transactional
    public UserAuthResponse register(UserRegisterRequest request) {
        String username = normalizeUsername(request.getUsername());
        String email = normalizeEmail(request.getEmail());

        if (userRepository.existsByUsername(username)) {
            throw new BusinessException("Username đã tồn tại");
        }
        if (userRepository.existsByEmail(email)) {
            throw new BusinessException("Email đã tồn tại");
        }
        if (!request.getPassword().equals(request.getConfirmPassword())) {
            throw new BusinessException("Mật khẩu xác nhận không khớp");
        }

        User user = User.builder()
                .username(username)
                .email(email)
                .fullName(request.getFullName().trim())
                .password(passwordEncoder.encode(request.getPassword()))
                .build();

        User saved = userRepository.save(user);
        return authenticate(saved);
    }

    @Transactional
    public UserAuthResponse login(UserLoginRequest request) {
        String account = request.getAccount().trim();

        // Chuan hoa ca hai nhan dung nhu luc dang ky: neu khong thi go "PROBE01"
        // se bi bao "mat khau khong chinh xac" trong khi mat khau van dung.
        User user = userRepository
                .findByUsernameOrEmail(normalizeUsername(account), normalizeEmail(account))
                .orElse(null);

        // MỘT thông báo cho cả "không có tài khoản" lẫn "sai mật khẩu".
        // AdminService tách làm hai ("Tài khoản không tồn tại"), tức là cho người
        // ngoài dò được email nào đã có mặt ở đây; khách hàng thì không nên lộ.
        if (user == null || !passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new BusinessException("Tài khoản hoặc mật khẩu không chính xác");
        }

        return authenticate(user);
    }

    /**
     * Đăng xuất = xoá phiên, không phải "client tự xoá localStorage".
     *
     * Token đã thu hồi thì tra DB không ra nữa, nên một đoạn script gọi thẳng API
     * bằng token cũ cũng chỉ nhận đúng 401 như mọi token lạ khác.
     */
    public void logout(String token) {
        sessionService.revoke(token);
    }

    /**
     * Một đường duy nhất cho mọi lần mở phiên, để register và login không bất
     * đồng: thiếu một trong hai thì khách đăng ký xong vẫn phải đăng nhập lại.
     */
    private UserAuthResponse authenticate(User user) {
        return UserAuthResponse.builder()
                .token(sessionService.issue(user.getId()))
                .user(UserProfileResponse.from(user))
                .build();
    }

    @Transactional(readOnly = true)
    public UserProfileResponse profile(Integer userId) {
        return UserProfileResponse.from(requireUser(userId));
    }

    @Transactional(readOnly = true)
    public List<RecipientResponse> listRecipients(Integer userId) {
        requireUser(userId);
        return recipientRepository.findByUserIdOrderByIsDefaultDescIdAsc(userId).stream()
                .map(RecipientResponse::from)
                .toList();
    }

    @Transactional
    public RecipientResponse addRecipient(Integer userId, RecipientRequest request) {
        requireUser(userId);

        String name = request.getName().trim();
        String phone = request.getPhone().trim();
        String address = request.getAddress().trim();

        if (recipientRepository.existsByUserIdAndNameAndPhoneAndAddress(userId, name, phone, address)) {
            throw new BusinessException("Người nhận này đã có trong danh sách");
        }

        // Người nhận đầu tiên đương nhiên là mặc định, kể cả client không hỏi.
        boolean first = recipientRepository
                .findByUserIdOrderByIsDefaultDescIdAsc(userId).isEmpty();
        boolean asDefault = first || Boolean.TRUE.equals(request.getIsDefault());

        if (asDefault) {
            recipientRepository.clearDefaults(userId);
        }

        // Doc lai: user cu da bi clearDefaults() lam cho detached.
        User owner = requireUser(userId);

        Recipient recipient = Recipient.builder()
                .user(owner)
                .name(name)
                .phone(phone)
                .address(address)
                .label(blankToNull(request.getLabel()))
                .isDefault(asDefault)
                .build();

        return RecipientResponse.from(recipientRepository.save(recipient));
    }

    @Transactional
    public RecipientResponse updateRecipient(Integer userId, Integer recipientId,
                                             RecipientRequest request) {
        Recipient current = requireRecipient(userId, recipientId);

        String name = request.getName().trim();
        String phone = request.getPhone().trim();
        String address = request.getAddress().trim();
        boolean setAsDefault = Boolean.TRUE.equals(request.getIsDefault());

        // existsBy... cũng đúng khi chính nó đã mang nội dung đó, nên phải loại
        // trường hợp "sửa nguyên bản" ra khỏi va chạm.
        boolean duplicate = recipientRepository
                .existsByUserIdAndNameAndPhoneAndAddress(userId, name, phone, address)
                && !sameContent(current, name, phone, address);
        if (duplicate) {
            throw new BusinessException("Người nhận này đã có trong danh sách");
        }

        if (setAsDefault) {
            recipientRepository.clearDefaults(userId);
        }

        // Doc lai sau moi lenh bulk: current da bi clear.
        Recipient recipient = requireRecipient(userId, recipientId);
        recipient.setName(name);
        recipient.setPhone(phone);
        recipient.setAddress(address);
        recipient.setLabel(blankToNull(request.getLabel()));

        // isDefault == null nghĩa là client không nhắc tới: giữ nguyên cờ cũ,
        // không âm thầm bỏ mặc định của một người nhận chỉ vì sửa lại địa chỉ.
        if (request.getIsDefault() != null) {
            recipient.setIsDefault(setAsDefault);
        }

        return RecipientResponse.from(recipientRepository.save(recipient));
    }

    @Transactional
    public void deleteRecipient(Integer userId, Integer recipientId) {
        Recipient recipient = requireRecipient(userId, recipientId);
        boolean wasDefault = Boolean.TRUE.equals(recipient.getIsDefault());

        recipientRepository.delete(recipient);
        // Lệnh xoá phải xuống DB trước khi dọn người khác lên làm mặc định.
        recipientRepository.flush();

        if (wasDefault) {
            recipientRepository.findByUserIdOrderByIsDefaultDescIdAsc(userId).stream()
                    .findFirst()
                    .ifPresent(next -> next.setIsDefault(true));
        }
    }

    /** Trả về cả danh sách mới để client khỏi phải gọi thêm một lần GET nữa. */
    @Transactional
    public List<RecipientResponse> setDefaultRecipient(Integer userId, Integer recipientId) {
        requireRecipient(userId, recipientId);
        recipientRepository.clearDefaults(userId);

        Recipient recipient = requireRecipient(userId, recipientId);
        recipient.setIsDefault(true);
        recipientRepository.flush();

        return recipientRepository.findByUserIdOrderByIsDefaultDescIdAsc(userId).stream()
                .map(RecipientResponse::from)
                .toList();
    }

    private User requireUser(Integer userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new BusinessException("Tài khoản không tồn tại"));
    }

    /**
     * Mọi đường chạm vào người nhận đều đi qua đây: id người nhận không đủ, lúc
     * nào cũng phải kèm tài khoản sở hữu để không sửa được sổ của người khác.
     */
    private Recipient requireRecipient(Integer userId, Integer recipientId) {
        return recipientRepository.findByIdAndUserId(recipientId, userId)
                .orElseThrow(() -> new BusinessException(
                        "Người nhận không tồn tại hoặc không thuộc tài khoản này"));
    }

    private boolean sameContent(Recipient recipient, String name, String phone, String address) {
        return recipient.getName().equals(name)
                && recipient.getPhone().equals(phone)
                && recipient.getAddress().equals(address);
    }

    private String normalizeEmail(String value) {
        return value.trim().toLowerCase(Locale.ROOT);
    }

    /**
     * Username cũng lowercase như email: PostgreSQL so sánh chuỗi có phân biệt
     * hoa thường, nên "Hung" và "hung" sẽ thành hai tài khoản khác nhau nếu bỏ
     * bước này.
     */
    private String normalizeUsername(String value) {
        return value.trim().toLowerCase(Locale.ROOT);
    }

    private String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
