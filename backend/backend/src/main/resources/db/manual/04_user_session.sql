-- ============================================================================
-- Xac thuc that cho tai khoan khach: token cua phien dang nhap.
--
-- Vi sao can: truoc day "dang nhap" chi la chuyen client giu profile vao
-- localStorage roi go /api/users/{id}/... . Server khong kiem tra gi ca, nen
-- bat ky ai doan duoc id deu doc duoc so nguoi nhan (ten + SDT + dia chi) cua
-- nguoi khac. Tu day moi endpoint phai trinh token, va userId lay tu token chu
-- khong lay tu client.
--
-- ddl-auto=validate nen phai chay thu cong:
--   $env:PGCLIENTENCODING='UTF8'; $env:PGPASSWORD='123'
--   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost `
--       -d shop_management_db -v ON_ERROR_STOP=1 -f 04_user_session.sql
--
-- File idempotent.
-- ============================================================================

-- Bang luu HASH cua token, KHONG luu token nguyen ven.
--
-- Token duoc sinh mot lan va tra ve cho client dung lam "mat khau de dang nhap
-- lai"; hash nam o day la bang chung ben goi. Roi ri co DB (khong co ma nguon
-- cua ung dung) thi ke tan cung khong tai tao duoc token cua nguoi khac, vi
-- SHA-256 khong dao duoc.
create table if not exists user_session (
    id         integer generated always as identity primary key,

    token_hash varchar(64) not null constraint uq_user_session_token_hash unique,

    -- Xoa tai khoan thi moi phien dang nhap cua no cung mat theo: khong co chuyen
    -- "tai khoan khong con ton tai ma token van con dung duoc".
    user_id    integer not null
        constraint fk_user_session_user references user_account(id) on delete cascade,

    created_at timestamp not null default current_timestamp,
    expires_at timestamp not null
);

-- Moi lan go API la mot lan tra cu du lieu theo dung gia tri nay.
create index if not exists idx_user_session_token_hash
    on user_session (token_hash);

-- Don day khi het han (SessionService.purgeExpired), va de tim phien cua mot
-- tai khoan khi xoa tai khoan.
create index if not exists idx_user_session_expires_at
    on user_session (expires_at);

create index if not exists idx_user_session_user_id
    on user_session (user_id);

comment on table user_session is
    'Phien dang nhap khach hang. Luu SHA-256 cua token, khong bao gio luu token goc.';
comment on column user_session.token_hash is
    'SHA-256 hex (64 ky tu, chu thuong) cua token da tra ve client.';
comment on column user_session.expires_at is
    'Qua thoi gian nay thi token bi tu choi; dong cu giu lai den khi purgeExpired xoa.';
