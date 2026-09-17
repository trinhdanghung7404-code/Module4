-- ============================================================================
-- Tai khoan khach hang + so nguoi nhan hang.
--
-- Cung ly do voi 02_orders.sql: application.properties dat
-- spring.jpa.hibernate.ddl-auto=validate, Hibernate chi KIEM TRA schema chu khong
-- tao bang. KHONG chay file nay thi backend khong khoi dong duoc nua (validate
-- that bai vi thieu bang user_account / cot orders.user_id).
--
--   $env:PGCLIENTENCODING='UTF8'; $env:PGPASSWORD='123'
--   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost `
--       -d shop_management_db -v ON_ERROR_STOP=1 -f 03_users.sql
--
-- File idempotent: chay lan thu hai khong gay hai lan nao.
-- ============================================================================

-- Ten bang la "user_account" chu khong phai "user". USER la tu khoa danh rieng
-- trong PostgreSQL (dong nghia cua current_user), tao bang "user" thi moi lan
-- tham chieu deu phai boi ngoac kep: select * from "user". Git xa duoc viec do.
create table if not exists user_account (
    id         integer generated always as identity primary key,

    -- Dang nhap duoc bang username HOAC email nen ca hai deu unique. Ca hai deu
    -- duoc lowercase trong UserService truoc khi luu: PostgreSQL phan biet hoa
    -- thuong nen "Hung" va "hung" se thanh hai tai khoan neu khong chuan hoa.
    username   varchar(50)  not null constraint uq_user_account_username unique,
    email      varchar(100) not null constraint uq_user_account_email unique,

    -- Hash BCrypt, 60 ky tu; de 255 cho giong cot cua bang admin.
    password   varchar(255) not null,
    full_name  varchar(100) not null,

    created_at timestamp not null default current_timestamp
);

comment on table user_account is
    'Tai khoan khach hang. Tach khoi nguoi nhan hang: mot tai khoan giao duoc cho nhieu dia chi.';
comment on column user_account.password is
    'Hash BCrypt. Khong bao gio tra ve client duoi bat ky hinh thuc nao.';
comment on column user_account.email is
    'Duoc lowercase luc luu nen "A@B.com" va "a@b.com" khong qua duoc unique.';

-- ----------------------------------------------------------------------------
-- So nguoi nhan hang cua tung tai khoan.
--
-- Day la du lieu de chon nhanh o man thanh toan. DON KHONG luu FK toi bang nay:
-- orders giu snapshot (customer_name/phone/address) tai thoi diat dat, neu khong
-- xoa hoac sua mot nguoi nhan se lam dich noi giao cua don da chot.
-- ----------------------------------------------------------------------------
create table if not exists recipient (
    id         integer generated always as identity primary key,

    user_id    integer not null
        constraint fk_recipient_user references user_account(id) on delete cascade,

    name       varchar(120)  not null,
    phone      varchar(20)   not null,
    address    varchar(255)  not null,

    -- "Nha", "Co quan"... de phan biet khi chon; de trong cung duoc.
    label      varchar(30),
    is_default boolean       not null default false,

    created_at timestamp not null default current_timestamp
);

-- Xoa tai khoan thi xoa luon so nguoi nhan: khong con chu de dung lai, va giu
-- dia chi cua ho trong he thong ma ho da muon xoa la dieu te hai.
create index if not exists idx_recipient_user_id
    on recipient (user_id);

-- Unique MOT phan: moi tai khoan toi da mot nguoi nhan mac dinh. PostgreSQL cho
-- phep index co dieu kien, nen bat nay nam duoc bat bien ma khong can trigger.
-- Service van phai gi gon (clearDefaults truoc khi dat cai moi), day luoi cuoi
-- phong khi co loi o tang khac.
create unique index if not exists uq_recipient_single_default
    on recipient (user_id)
    where is_default;

comment on table recipient is
    'Nguoi nhan hang da luu cua mot tai khoan, dung de prefill man thanh toan.';

-- ----------------------------------------------------------------------------
-- Gan don voi tai khoan da dat no.
--
-- Nullable va ON DELETE SET NULL: don phai con doc duoc sau khi khach xoa tai
-- khoan. Cũ thi khong co gi, nen khong the them NOT NULL.
-- ----------------------------------------------------------------------------
alter table orders
    add column if not exists user_id integer;

alter table orders drop constraint if exists fk_orders_user;
alter table orders
    add constraint fk_orders_user
    foreign key (user_id) references user_account(id) on delete set null;

comment on column orders.user_id is
    'Tai khoan da dat don; null la dat khong dang nhap. Nguoi nhan van nam o customer_name/phone/address.';

-- "Don hang cua toi" loc theo user_id, don theo created_at.
create index if not exists idx_orders_user_id
    on orders (user_id, created_at desc);

-- ============================================================================
-- Kiem tra sau khi chay:
--
--   select u.username, count(r.id) as recipients,
--          coalesce(sum(r.is_default::int), 0) as defaults
--     from user_account u left join recipient r on r.user_id = u.id
--    group by u.id, u.username;
--
--   -- defaults phai luon luon <= 1 voi moi tai khoan
--   select user_id, count(*) from recipient where is_default group by user_id having count(*) > 1;
-- ============================================================================
