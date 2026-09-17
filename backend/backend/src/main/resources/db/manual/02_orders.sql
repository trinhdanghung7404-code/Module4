-- ============================================================================
-- Don hang: bang "orders" va "order_item".
--
-- Repo khong dung Flyway/Liquibase va application.properties dat
-- spring.jpa.hibernate.ddl-auto=validate, nghĩa là Hibernate chỉ KIỂM TRA schema
-- chứ không tạo bảng. Vì vậy file này là nguồn duy nhất của schema đơn hàng và
-- phải được chạy thủ công một lần:
--
--   $env:PGCLIENTENCODING='UTF8'; $env:PGPASSWORD='123'
--   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -U postgres -h localhost `
--       -d shop_management_db -v ON_ERROR_STOP=1 -f 02_orders.sql
--
-- Ghi chu style: giuan nguyen tac cua Product/Category - snake_case,
-- "integer generated always as identity", ten constraint dat tay
-- (uq_/fk_/chk_) de error message de doc.
-- ============================================================================

-- Ten bang la "orders" chu khong phai "order": ORDER la tu khoa danh rieng trong
-- PostgreSQL (ORDER BY), dat "order" thi moi cau lenh deu phai boi ngoac kep.
create table if not exists orders (
    id            integer generated always as identity primary key,

    -- Ma don in ra cho khach, dang DH-000001. Duoc sinh tu id sau khi insert nen
    -- luc insert dau tien khong the biet truoc gia tri: service tam thoi ghi mot
    -- ma ngan han duy nhat roi update lai trong cung transaction.
    --
    -- Identity KHONG rollback theo transaction: don bi tu choi (het hang, validate
    -- fail) van lot mat mot so, nen day DH-xxxxxx khong lien hoan. Da do bang
    -- chung: sau 5 don probe roi don nao quo lai thi count(*) = 0 ma don ke tiep
    -- se la DH-000006. Ma don duoc cham nhu vay de tra cuu khong tranh chap giua
    -- hai request song song, khong phai de dem nhu hoa don.
    order_code    varchar(30)   not null
        constraint uq_orders_order_code unique,

    customer_name varchar(120)  not null,
    phone         varchar(20)   not null,
    address       varchar(255)  not null,
    note          varchar(500),

    -- Tong tien do server tinh, khong bao gan lay tu request.
    total_amount  numeric(12,2) not null
        constraint chk_orders_total_amount check (total_amount >= 0),

    -- EnumType.STRING trong entity: luu chu, khong luu so thu tu, de sau nay them
    -- trang thai khong lam roi du lieu cu.
    status        varchar(20)   not null default 'PENDING',

    created_at    timestamp     not null default current_timestamp
);

create index if not exists idx_orders_created_at
    on orders (created_at desc);

create table if not exists order_item (
    id           integer generated always as identity primary key,

    -- Xoa don hang thi xoan het chi tiet.
    order_id     integer         not null
        constraint fk_order_item_order references orders (id) on delete cascade,

    -- Product duoc phep null vi admin delete cua backend la xoa vat ly: don hang
    -- cu phai ton tai sau khi san pham bi xoa.
    product_id   integer
        constraint fk_order_item_product references product (id) on delete set null,

    -- Snapshot ten va gia tai thoi diat dat hang, de don khong bi "doi tien" khi
    -- admin sua gia hoac doi ten san pham.
    product_name varchar(160)    not null,
    unit_price   numeric(12,2)   not null
        constraint chk_order_item_unit_price check (unit_price >= 0),
    quantity     integer         not null
        constraint chk_order_item_quantity check (quantity > 0)
);

create index if not exists idx_order_item_order_id
    on order_item (order_id);
