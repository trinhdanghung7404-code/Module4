package com.example.backend.repository;

import com.example.backend.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Integer> {

    boolean existsByUsername(String username);

    boolean existsByEmail(String email);

    /** Dang nhap bang username HOAC email, giong het cach AdminRepository lam. */
    Optional<User> findByUsernameOrEmail(String username, String email);
}
