param([switch]$Clean)

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

if ($Clean) {
    # product_image tu xoá theo product (FK ON DELETE CASCADE), nen chi can mot lenh.
    Write-Output (Sql @("delete from product where sku like 'SEED-DT-%'"))
    Write-Output '--- con lai sau khi don ---'
    Write-Output (Sql @(
        "select 'seed_products=' || count(*) from product where sku like 'SEED-DT-%'",
        "select 'all_products=' || count(*) from product",
        "select 'all_images=' || count(*) from product_image"
    ))
    exit 0
}

$sqlFile = Join-Path $PSScriptRoot 'seed-products.sql'
& $psql -w -h localhost -U postgres -d shop_management_db -P pager=off `
        -v ON_ERROR_STOP=1 -f $sqlFile
if ($LASTEXITCODE -ne 0) { throw "psql chay that bai (exit $LASTEXITCODE)" }

Write-Output '--- sau khi seed ---'
Write-Output (Sql @(
    "select 'seed_products=' || count(*) from product where sku like 'SEED-DT-%'",
    "select 'seed_images=' || count(*) from product_image pi
       join product p on p.id = pi.product_id where p.sku like 'SEED-DT-%'",
    "select 'orders referencing seed=' || count(*) from order_item i
       join product p on p.id = i.product_id where p.sku like 'SEED-DT-%'"
))
Write-Output (Sql @("select sku, name, price, quantity from product where sku like 'SEED-DT-%' order by sku"))
