package com.example.backend.repository;

import com.example.backend.entity.Admin;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface AdminRepository extends JpaRepository<Admin, Integer> {

    boolean existsByUsername(String username);

    boolean existsByEmail(String email);

    Optional<Admin> findByUsernameOrEmail(String username, String email);
}