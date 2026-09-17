-- ============================================================================
-- Luong thuc hien don hang: thoi diem doi trang thai, don vi van chuyen, va
-- lich su trang thai.
--
-- Vi sao can: EnumStatus da du nam gia tri tu dau (PENDING/CONFIRMED/SHIPPING/
-- DELIVERED/CANCELLED) nen KHONG phai dong them gia tri nao. Thieu ba thing:
--   1) khong co dau vet thoi gian - client chi biet trang thai hien tai, khong
--      biet don duoc xac nhan luc may gio;
--   2) khong co cho nao ghi don vi van chuyen / ma van don;
--   3) khong co ai ghi lai tung buoc doi, nen khong tra loi duoc "ai da huy don
--      nay va huy tu trang thai nao".
--
-- ddl-auto=validate nen phai chay thu cong:
--   $env:PGCLIENTENCODING='UTF8'; $env:PGPASSWORD='123'
--   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost `
--       -d shop_management_db -v ON_ERROR_STOP=1 -f 06_order_fulfillment.sql
--
-- File idempotent.
-- ============================================================================

-- 1) Ba cot moi tren "orders". Ca ba deu null duoc: don cu chua co lich su thi
--    trang thai gan nhat tinh theo created_at.
alter table orders
    add column if not exists status_updated_at timestamp,
    add column if not exists shipping_unit     varchar(100),
    add column if not exists tracking_code     varchar(60);

comment on column orders.status_updated_at is
    'Luc don buoc vao status hien tai. Null voi don tao truoc khi co luong theo doi.';
comment on column orders.shipping_unit is
    'Ten don vi van chuyen do admin nhap. KHONG phai tich hop API that.';
comment on column orders.tracking_code is
    'Ma van don do don vi van chuyen cap, admin ghi tay.';

-- Admin loc danh sach theo trang thai (mac dinh la "cho xac nhan").
create index if not exists idx_orders_status
    on orders (status);

-- Backfill: don da ton tai thi coi nhu trang thai hien tai co tu luc dat.
update orders
   set status_updated_at = created_at
 where status_updated_at is null;

-- 2) Lich su trang thai - mot dong cho MOT buoc doi, khong bao gio sua.
--
-- Bang nay la bang ghi chu thuan: muon biet "don DQ-... bi huy luc nao, boi ai"
-- thi phai co no, va khong the tai dung lai tu cot status (chi giu gia tri cuoi).
create table if not exists order_status_history (
    id          integer generated always as identity primary key,

    order_id    integer not null
        constraint fk_order_status_history_order references orders (id) on delete cascade,

    -- Null o dong dau tien cua mot don (luong chua ghi luc tao don, dong nay duoc
    -- sinh khi don buoc sang trang thai dau tien hoac khi backfill).
    from_status varchar(20),
    to_status   varchar(20)  not null,

    -- Ai tac dong: ADMIN = nhan vien, USER = chinh khach hang, SYSTEM = backfill
    -- hoac tep san pham bi xoa khi thu hoi kho.
    actor_type  varchar(10)  not null default 'ADMIN'
        constraint chk_order_status_history_actor_type
        check (actor_type in ('ADMIN', 'USER', 'SYSTEM')),

    -- Null voi SYSTEM. Khong phai FK: admin bi xoa thi dong lich su van phai con.
    actor_id    integer,

    -- Ten hien thi tai thoi diem hanh dong, chep ra de dong doc lap voi bang
    -- admin/user: sua ten hay xoa tai khoan khong lam dong lich su doi nguoi.
    actor_label varchar(120),

    note        varchar(500),

    created_at  timestamp not null default current_timestamp
);

-- Trang chi tiet don: lay ca lineage theo dung mot cau lenh.
create index if not exists idx_order_status_history_order_id
    on order_status_history (order_id, id);

comment on table order_status_history is
    'Moi dong = mot buoc doi trang thai cua don. Insert-only, khong update, khong delete.';

-- Backfill cho don cu de stepper khong trong tro.
insert into order_status_history
    (order_id, from_status, to_status, actor_type, actor_id, actor_label, note, created_at)
select o.id,
       null,
       o.status,
       'SYSTEM',
       null,
       null,
       'Bo sung cho don tao truoc khi co luong theo doi',
       o.created_at
  from orders o
 where not exists (
           select 1 from order_status_history h where h.order_id = o.id
       );
