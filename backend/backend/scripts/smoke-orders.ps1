$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
$env:PGPASSWORD = '123'
$env:PGCLIENTENCODING = 'UTF8'
$dbArgs = @('-w', '-h', 'localhost', '-U', 'postgres', '-d', 'shop_management_db', '-P', 'pager=off', '-At')
$base = if ($env:SMOKE_API_BASE) { $env:SMOKE_API_BASE } else { 'http://localhost:8081/api' }
$logPath = Join-Path $PSScriptRoot 'smoke-orders.log'
$log = New-Object System.Collections.Generic.List[string]

function Say($m) { $script:log.Add($m) }
function Sql($q) { (& $psql @dbArgs -c $q) -join ' ; ' }

# Tu khi co user_session, POST /orders nam sau UserSessionInterceptor: script
# phai mo mot tai khoan probe va gui token cua no tren moi request.
$probeUser = 'zzsmoke' + (Get-Random -Maximum 99999)
$regJson = (@{ username = $probeUser; fullName = 'ZZ Smoke Probe'; email = ($probeUser + '@example.com')
  password = 'zz-probe-pass'; confirmPassword = 'zz-probe-pass' } | ConvertTo-Json -Compress -Depth 5)
$token = (Invoke-RestMethod -Uri ($base + '/users/register') -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($regJson))).token
if (-not $token) { throw 'dang ky probe khong tra ve token' }

function Api($path, $body) {
  try {
    # GetBytes: PS 5.1 ma hoa body string theo ASCII bat ke charset trong
    # ContentType, nen tieng Viet trong body bi roi chu o ngay ben gui.
    $r = Invoke-WebRequest -Uri ($base + $path) -Method Post `
      -ContentType 'application/json; charset=utf-8' `
      -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing `
      -Headers @{ Authorization = ('Bearer ' + $token) }
    # .Content trong PS 5.1 giai body theo charset cua Content-Type; Spring tra
    # "application/json" khong kem charset nen tieng Viet bi doc thanh ISO-8859-1.
    # Doc truc tiep raw stream moi thay dung UTF-8 ma server gui.
    $text = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())
    return @{ status = [int]$r.StatusCode; text = $text }
  } catch {
    $resp = $_.Exception.Response
    if ($resp) {
      $sr = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
      return @{ status = [int]$resp.StatusCode; text = $sr.ReadToEnd() }
    }
    return @{ status = 0; text = $_.Exception.Message }
  }
}

function Body($phone, $name, $items, $note) {
  $p = @{ customerName = $name; phone = $phone; address = '1 Test St'; items = $items }
  if ($null -ne $note) { $p.note = $note }
  return ($p | ConvertTo-Json -Compress -Depth 5)
}

Say "=== SMOKE TEST POST /api/orders ==="

# ---- 0. Probe data (never touching product 2 / 4) -------------------------------
Sql "delete from orders where customer_name like 'ZZ Probe%' or customer_name like 'Nguy%ZZ%'" | Out-Null
Sql "delete from product where sku in ('zzprobe1','zzprobe2')" | Out-Null
Sql "insert into product (name, sku, price, quantity, category_id) values
     ('ZZ Probe One','zzprobe1',250000,1,1),('ZZ Probe Two','zzprobe2',100000,5,1)" | Out-Null

$p1 = [int](Sql "select id from product where sku='zzprobe1'")   # stock 1
$p2 = [int](Sql "select id from product where sku='zzprobe2'")   # stock 5
Say "probe p1=$p1 (stock 1, 250000d)   p2=$p2 (stock 5, 100000d)"

# ---- T1 happy path --------------------------------------------------------------
$r = Api '/orders' (Body '0900000001' 'ZZ Probe Buyer' @(@{ productId = $p2; quantity = 2 }) '   ')
$o = $r.text | ConvertFrom-Json
Say ("T1 happy        -> HTTP {0} | code={1} | total={2} | status={3} | lines={4} | qty={5} | blankNote->null={6}" -f `
  $r.status, $o.orderCode, $o.totalAmount, $o.status, $o.items.Count, $o.items[0].quantity, ($null -eq $o.note))
Say ("T1 stock 5->3   -> " + (Sql "select quantity from product where id=$p2"))
Say ("T1 snapshot     -> " + $o.items[0].productName + " @ " + $o.items[0].unitPrice + " | lineTotal=" + $o.items[0].lineTotal)
Say ("T1 createdAt    -> " + $o.createdAt)

# ---- T2 duplicate cart lines must merge into ONE item (2+1=3) -------------------
$r = Api '/orders' (Body '0900000002' 'ZZ Probe Buyer' @(
  @{ productId = $p2; quantity = 2 }, @{ productId = $p2; quantity = 1 }) $null)
$o = $r.text | ConvertFrom-Json
Say ("T2 merge        -> HTTP {0} | lines={1} | mergedQty={2} | total={3} | stockNow={4}" -f `
  $r.status, $o.items.Count, $o.items[0].quantity, $o.totalAmount, (Sql "select quantity from product where id=$p2"))

# ---- T3 oversell: stock is 0 now, 99 must fail with nothing changed -------------
$before = Sql "select quantity from product where id=$p2"
$r = Api '/orders' (Body '0900000003' 'ZZ Probe Buyer' @(@{ productId = $p2; quantity = 99 }) $null)
Say ("T3 oversell     -> HTTP {0} | msg={1}" -f $r.status, $r.text)
Say ("T3 stock same   -> " + (Sql "select quantity from product where id=$p2") + " (was $before)")

# ---- T4 product id that does not exist ------------------------------------------
$r = Api '/orders' (Body '0900000004' 'ZZ Probe Buyer' @(@{ productId = 999999; quantity = 1 }) $null)
Say ("T4 missing prod -> HTTP {0} | msg={1}" -f $r.status, $r.text)

# ---- T5 validation shape (field map, no 'message' key) --------------------------
$r = Api '/orders' (@{ customerName = ''; phone = 'abc'; address = ''; items = @() } | ConvertTo-Json -Compress)
Say ("T5 validation   -> HTTP {0} | body={1}" -f $r.status, $r.text)

# ---- T6 UTF-8 round trip --------------------------------------------------------
$viet = [regex]::Unescape('Nguy\u1EC5n V\u0103n ZZ')
$r = Api '/orders' (Body '0900000006' $viet @(@{ productId = $p1; quantity = 1 }) $null)
$o = $r.text | ConvertFrom-Json
Say ("T6 utf8 name    -> HTTP {0} | echoedOk={1} | code={2}" -f $r.status, ($o.customerName -eq $viet), $o.orderCode)

# ---- T7 concurrency: stock 1, two parallel qty=1 --------------------------------
Sql "update product set quantity=1 where id=$p1" | Out-Null
$b1 = Body '0900000007' 'ZZ Probe Buyer' @(@{ productId = $p1; quantity = 1 }) $null
$b2 = Body '0900000008' 'ZZ Probe Buyer' @(@{ productId = $p1; quantity = 1 }) $null
Add-Type -AssemblyName System.Net.Http
$client = New-Object System.Net.Http.HttpClient
# Hai request dau tranh nhau cung phai mang token, neu khong ca hai chi 401.
$client.DefaultRequestHeaders.Authorization =
  New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $token)
$c1 = New-Object System.Net.Http.StringContent($b1, [System.Text.Encoding]::UTF8, 'application/json')
$c2 = New-Object System.Net.Http.StringContent($b2, [System.Text.Encoding]::UTF8, 'application/json')
$t1 = $client.PostAsync("$base/orders", $c1)
$t2 = $client.PostAsync("$base/orders", $c2)
[System.Threading.Tasks.Task]::WaitAll(@($t1, $t2))
Say ("T7 concurrent   -> codes {0} / {1} (expect exactly one 201)" -f [int]$t1.Result.StatusCode, [int]$t2.Result.StatusCode)
Say ("T7 stock 1->0   -> " + (Sql "select quantity from product where id=$p1"))
Say ("T7 orders won   -> " + (Sql "select count(*) from orders where phone in ('0900000007','0900000008')"))
$client.Dispose()

# ---- T8 rows in DB + no temporary code leaked -----------------------------------
Say ("T8 rows         -> " + (Sql "select o.order_code||'='||o.total_amount||' x'||count(i.id) from orders o join order_item i on i.order_id=o.id group by o.order_code,o.total_amount order by o.order_code"))
Say ("T8 TMP leaked   -> " + (Sql "select count(*) from orders where order_code like 'TMP%'"))
Say ("T8 code format  -> " + (Sql "select bool_and(order_code ~ '^DH-[0-9]{6}$') from orders"))
Say ("T8 bad qty rows -> " + (Sql "select count(*) from order_item where quantity <= 0"))
Say ("T8 orphan items -> " + (Sql "select count(*) from order_item i left join orders o on o.id=i.order_id where o.id is null"))

# ---- Cleanup --------------------------------------------------------------------
Sql "delete from orders where customer_name like 'ZZ Probe%' or customer_name like 'Nguy%ZZ%'" | Out-Null
Sql "delete from product where sku in ('zzprobe1','zzprobe2')" | Out-Null
Sql "delete from user_account where username='$probeUser'" | Out-Null
Say ("AFTER orders    -> " + (Sql "select count(*) from orders"))
Say ("AFTER order_item-> " + (Sql "select count(*) from order_item"))
Say ("AFTER baseline  -> " + (Sql "select id||'='||quantity from product where id in (2,4) order by id"))
Say ("AFTER max id    -> " + (Sql "select coalesce(max(id),0) from orders"))

[System.IO.File]::WriteAllLines($logPath, $log, (New-Object System.Text.UTF8Encoding $true))
Write-Host "written: $logPath"

