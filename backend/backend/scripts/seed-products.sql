-- =============================================================================
-- Seed 20 san pham demo (dien thoai) vao category dau tien, anh dung ngau nhien
-- tu picsum.photos/seed/<sku>. Chay: .\seed-products.ps1  (hoac psql -f file nay).
--
-- Idempotent: chay nhieu lan khong tao trung (guard theo sku), va khong bao gio
-- cham vao du lieu that — chi dong SEED-DT-**, khong update/delete gi khac.
-- Don sieu thi: .\seed-products.ps1 -Clean
-- =============================================================================

begin;

insert into product (name, sku, price, quantity, category_id, description)
select v.name, v.sku, v.price, v.quantity,
       (select id from category order by id limit 1),
       v.description
  from (values
    ('Aurora X5 5G 128GB',        'SEED-DT-01',  6990000,  24, 'Huy anh dem tot, pin 5000mAh, bao hanh 12 thang.'),
    ('Aurora X5 Pro 5G 256GB',    'SEED-DT-02',  9490000,  15, 'Man AMOLED 120Hz, cam bien chinh 50MP.'),
    ('Aurora Lite 4G 64GB',       'SEED-DT-03',  3290000,  40, 'Phien ban gia re, pin dung hai ngay.'),
    ('Nimbus Note 12 5G 256GB',   'SEED-DT-04',  8190000,  12, 'But bi viet truc tiep len man hinh.'),
    ('Nimbus Note 12 Ultra 512GB','SEED-DT-05', 13990000,   6, 'Camera zoom quang 10x, khung titan.'),
    ('Nimbus Pad Phone 5G 128GB', 'SEED-DT-06',  5490000,  30, 'Man lon 6.9 inch, hop cho giai tri.'),
    ('Zephyr S9 5G 256GB',        'SEED-DT-07', 11290000,   9, 'Chong nuoc IP68, sac nhanh 65W.'),
    ('Zephyr S9 FE 5G 128GB',     'SEED-DT-08',  7490000,  18, 'Ban FE gia mem hon, van co 5G.'),
    ('Zephyr Mini 5G 128GB',      'SEED-DT-09',  6290000,  21, 'Nho gon 5.4 inch, trong luong 148g.'),
    ('Vertex One 5G 256GB',       'SEED-DT-10', 15990000,   4, 'Dong cao cap, camera sap toan than.'),
    ('Vertex One Fold 512GB',     'SEED-DT-11', 29990000,   0, 'Man gap 7.6 inch. Het hang, da dat het suat.'),
    ('Vertex Air 5G 128GB',       'SEED-DT-12',  8790000,  16, 'May mong nhat dong, 6.9mm.'),
    ('Lumen K7 4G 128GB',         'SEED-DT-13',  4190000,  35, 'Pho thong, loa kep, gia de tiep can.'),
    ('Lumen K7 Plus 4G 256GB',    'SEED-DT-14',  5790000,  22, 'Cung dong but nhieu hon bo nho.'),
    ('Lumen K9 5G 256GB',         'SEED-DT-15',  9990000,  11, 'Camera 200MP, tong dung luong lon.'),
    ('Solis P30 5G 128GB',        'SEED-DT-16',  6790000,  27, 'Pin 6000mAh, sac dao nguoc.'),
    ('Solis P30 Pro 5G 256GB',    'SEED-DT-17', 10490000,   8, 'Ong kinh periscope, chong rung ky thuat so.'),
    ('Solis Neo 4G 64GB',         'SEED-DT-18',  2790000,  45, 'May cho nguoi lon tuoi, phim lon, am thanh cao.'),
    ('Orion X1 5G 256GB',         'SEED-DT-19', 18490000,   2, 'Dau bang, van xu ly 3nm, sac khong day.'),
    ('Orion X1 Lite 5G 128GB',    'SEED-DT-20',  7990000,   3, 'Ban rut gon cua X1, gan dung nhu ban goc.')
  ) as v(name, sku, price, quantity, description)
 where not exists (select 1 from product p where p.sku = v.sku);

-- Mat truoc: anh ngau nhien nhung co dinh — cung sku thi ra cung anh,
-- de lan chay lai khien sieu thi khong doi hinh cua san pham.
insert into product_image (product_id, image_type, image_url, image_public_id)
select p.id,
       'FRONT',
       'https://picsum.photos/seed/' || lower(p.sku) || '/900/900',
       'seed/' || lower(p.sku) || '-front'
  from product p
 where p.sku like 'SEED-DT-%'
   and not exists (select 1 from product_image pi
                    where pi.product_id = p.id and pi.image_type = 'FRONT');

-- Mat sau: cung bo seed nhung mau xam, de phan biet khi UI hien hai anh.
insert into product_image (product_id, image_type, image_url, image_public_id)
select p.id,
       'BACK',
       'https://picsum.photos/seed/' || lower(p.sku) || 'b/900/900?grayscale',
       'seed/' || lower(p.sku) || '-back'
  from product p
 where p.sku like 'SEED-DT-%'
   and not exists (select 1 from product_image pi
                    where pi.product_id = p.id and pi.image_type = 'BACK');

commit;
