package com.example.backend.dto.response;

import com.example.backend.entity.Recipient;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class RecipientResponse {

    private final Integer id;
    private final String name;
    private final String phone;
    private final String address;
    private final String label;
    private final Boolean isDefault;
    private final LocalDateTime createdAt;

    public static RecipientResponse from(Recipient recipient) {
        return RecipientResponse.builder()
                .id(recipient.getId())
                .name(recipient.getName())
                .phone(recipient.getPhone())
                .address(recipient.getAddress())
                .label(recipient.getLabel())
                .isDefault(Boolean.TRUE.equals(recipient.getIsDefault()))
                .createdAt(recipient.getCreatedAt())
                .build();
    }
}
