-- ============================================================================
-- Xac thuc that cho trang quan tri: token cua phien dang nhap admin.
--
-- Vi sao can: truoc day "dang nhap admin" chi la chuyen client giu profile vao
-- localStorage roi ProtectedRoute kiem tra cai key do. Server khong kiem tra gi
-- ca, nen bat ky ai goi thang /api/admin/products (POST/PUT/DELETE) deu duoc,
-- khong can biet mat khau. Luong don hang moi co buoc admin doi trang thai, ma
-- mot endpoint ghi nhu vay ma de trong thi khong the goi la khap kin.
--
-- Cop nguyen cach lam cua 04_user_session.sql: token opaque, DB chi giu SHA-256.
--
-- ddl-auto=validate nen phai chay thu cong:
--   $env:PGCLIENTENCODING='UTF8'; $env:PGPASSWORD='123'
--   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost `
--       -d shop_management_db -v ON_ERROR_STOP=1 -f 05_admin_session.sql
--
-- File idempotent.
-- ============================================================================

create table if not exists admin_session (
    id         integer generated always as identity primary key,

    token_hash varchar(64) not null constraint uq_admin_session_token_hash unique,

    -- Xoa admin thi moi phien cua ho mat theo, khong de lai token cua tai khoan
    -- khong con ton tai.
    admin_id   integer not null
        constraint fk_admin_session_admin references admin(id) on delete cascade,

    created_at timestamp not null default current_timestamp,
    expires_at timestamp not null
);

create index if not exists idx_admin_session_token_hash
    on admin_session (token_hash);

create index if not exists idx_admin_session_expires_at
    on admin_session (expires_at);

create index if not exists idx_admin_session_admin_id
    on admin_session (admin_id);

comment on table admin_session is
    'Phien dang nhap quan tri. Luu SHA-256 cua token, khong bao gio luu token goc.';
