$ErrorActionPreference = 'Stop'
$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
$env:PGPASSWORD = '123'
$env:PGCLIENTENCODING = 'UTF8'
$dbArgs = @('-w', '-h', 'localhost', '-U', 'postgres', '-d', 'shop_management_db', '-P', 'pager=off', '-At')
function Sql($q) { (& $psql @dbArgs -c $q) -join ' ; ' }
$out = New-Object System.Collections.Generic.List[string]
function Say($m) { $script:out.Add($m) }
$base = if ($env:SMOKE_API_BASE) { $env:SMOKE_API_BASE } else { 'http://localhost:8081/api' }

# Tu migration 04_user_session, POST /orders nam sau UserSessionInterceptor:
# moi script smoke phai mo mot tai khoan probe cua rieng minh va dung token cua no.
$probeUser = 'zzprice' + (Get-Random -Maximum 99999)
$regBody = (@{ username = $probeUser; fullName = 'ZZ Price Probe'; email = ($probeUser + '@example.com')
    password = 'zz-probe-pass'; confirmPassword = 'zz-probe-pass' } | ConvertTo-Json -Compress)
$token = (Invoke-RestMethod -Uri ($base + '/users/register') -Method Post `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($regBody))).token
if (-not $token) { throw 'dang ky probe khong tra ve token' }

function Post($body) {
  try {
    $r = Invoke-WebRequest -Uri "$base/orders" -Method Post -ContentType 'application/json' -Body $body `
      -Headers @{ Authorization = ('Bearer ' + $token) } -UseBasicParsing
    return @{ status = [int]$r.StatusCode; text = $r.Content }
  } catch {
    $resp = $_.Exception.Response
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    return @{ status = [int]$resp.StatusCode; text = $sr.ReadToEnd() }
  }
}

Sql "delete from orders where phone in ('0900000071','0900000072','0900000073')" | Out-Null
Sql "delete from product where sku='zzprobe4'" | Out-Null
Sql "insert into product (name,sku,price,quantity,category_id) values ('ZZ Probe Four','zzprobe4',111000,5,1)" | Out-Null
$p = [int](Sql "select id from product where sku='zzprobe4'")
Say "probe id=$p price=111000 stock=5"

$r = Post ('{"customerName":"ZZ Price","phone":"0900000071","address":"A1","items":[{"productId":' + $p + ',"quantity":2}]}')
Say ("A baseline 2 x 111000 -> HTTP {0} total={1}" -f $r.status, ((($r.text | ConvertFrom-Json)).totalAmount))

# Admin doi gia giua luc khach dang o man hinh thanh toan
Sql "update product set price=999000 where id=$p" | Out-Null
$r = Post ('{"customerName":"ZZ Price","phone":"0900000072","address":"A2","items":[{"productId":' + $p + ',"quantity":1}]}')
Say ("B price changed to 999000 -> HTTP {0} total={1} (client still thinks 111000)" -f $r.status, (($r.text | ConvertFrom-Json)).totalAmount)

# Gia su khach mo DevTools va gui kem price/totalAmount gia
$r = Post ('{"customerName":"ZZ Price","phone":"0900000073","address":"A3","totalAmount":0,"items":[{"productId":' + $p + ',"quantity":1,"price":1,"lineTotal":1,"name":"hack"}]}')
Say ("C spoofed price/total in body -> HTTP {0} total={1}" -f $r.status, (($r.text | ConvertFrom-Json)).totalAmount)
Say ("C snapshot name from DB -> " + (($r.text | ConvertFrom-Json)).items[0].productName)
Say ("C stock 5-2-1-1 -> " + (Sql "select quantity from product where id=$p"))

Sql "delete from orders where phone in ('0900000071','0900000072','0900000073')" | Out-Null
Sql "delete from product where sku='zzprobe4'" | Out-Null
# user_session xoa theo tai khoan (FK cascade), khong de lai phien rung.
Sql "delete from user_account where username='$probeUser'" | Out-Null
Say ("AFTER orders=" + (Sql "select count(*) from orders") + " items=" + (Sql "select count(*) from order_item") +
  " baseline=" + (Sql "select id||'='||quantity from product where id in (2,4) order by id"))

[System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot 'price-check.log'), $out, (New-Object System.Text.UTF8Encoding $true))
Write-Host 'written'
