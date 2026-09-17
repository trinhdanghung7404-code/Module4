param([switch]$Verify)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:PGPASSWORD = '123'
$env:PGCLIENTENCODING = 'UTF8'
$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'

function Sql([string[]]$statements) {
    $args = @('-w', '-h', 'localhost', '-U', 'postgres', '-d', 'shop_management_db',
              '-P', 'pager=off', '-v', 'ON_ERROR_STOP=1')
    foreach ($s in $statements) { $args += @('-c', $s) }
    return (& $psql @args) -join "`n"
}

# Cong TRUNG so hang da ban vao lai dung don probe, thay de quy ve so "nen"
# da ghi nho: ne neu ai do vua thay doi thi loi khong bi an di.
$restore = @"
update product p
   set quantity = p.quantity + s.qty
  from (select i.product_id, sum(i.quantity) as qty
          from order_item i
          join orders o on o.id = i.order_id
         where o.phone in ('0900000033','0900000055')
         group by 1) s
 where p.id = s.product_id
"@

Write-Output (Sql @($restore))
Write-Output (Sql @(
    "delete from orders where phone in ('0900000033','0900000055')",
    "delete from user_account where username like 'probe%'"
))

Write-Output '--- trang thai sau don ---'
Write-Output (Sql @(
    "select 'orders=' || (select count(*) from orders)",
    "select 'order_item=' || (select count(*) from order_item)",
    "select 'user_account=' || (select count(*) from user_account)",
    "select 'recipient=' || (select count(*) from recipient)",
    "select 'stock_' || id || '=' || quantity from product where id in (2,4) order by id"
))
