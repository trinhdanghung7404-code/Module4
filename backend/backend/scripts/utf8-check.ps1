$ErrorActionPreference = 'Stop'
$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
$env:PGPASSWORD = '123'
$env:PGCLIENTENCODING = 'UTF8'
$dbArgs = @('-w', '-h', 'localhost', '-U', 'postgres', '-d', 'shop_management_db', '-P', 'pager=off', '-At')
function Sql($q) { (& $psql @dbArgs -c $q) -join ' ; ' }

$logPath = Join-Path $PSScriptRoot 'utf8-check.log'
$out = New-Object System.Collections.Generic.List[string]
function Say($m) { $script:out.Add($m) }

Sql "delete from orders where phone in ('0900000061','0900000062')" | Out-Null
Sql "delete from product where sku='zzprobe3'" | Out-Null
Sql "insert into product (name,sku,price,quantity,category_id) values ('ZZ Probe Three','zzprobe3',50000,2,1)" | Out-Null
$pid3 = [int](Sql "select id from product where sku='zzprobe3'")

# Nguyễn Văn ZZ / 12 LÊ LỢI / Gọi trước khi giao
$name = [regex]::Unescape('Nguy\u1EC5n V\u0103n ZZ')
$addr = [regex]::Unescape('12 L\u00EA L\u1EE3i')
$note = [regex]::Unescape('G\u1EA1i tr\u1EDBc khi giao')

$json = (@{ customerName = $name; phone = '0900000061'; address = $addr; note = $note
  items = @(@{ productId = $pid3; quantity = 1 }) } | ConvertTo-Json -Compress -Depth 5)

Say "ConvertTo-Json kept non-ASCII literals = " + [bool]($json -match '[^\x00-\x7F]')

Add-Type -AssemblyName System.Net.Http

# POST /orders thuoc nhom duoc bao ve tu khi co user_session, nen phai dang ky
# mot tai khoan probe va gan Authorization vao HttpClient.
$base = if ($env:SMOKE_API_BASE) { $env:SMOKE_API_BASE } else { 'http://localhost:8081/api' }
$probeUser = 'zzutf8' + (Get-Random -Maximum 99999)
$regJson = (@{ username = $probeUser; fullName = 'ZZ Utf8 Probe'; email = ($probeUser + '@example.com')
  password = 'zz-probe-pass'; confirmPassword = 'zz-probe-pass' } | ConvertTo-Json -Compress -Depth 5)
$token = (Invoke-RestMethod -Uri ($base + '/users/register') -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($regJson))).token
if (-not $token) { throw 'dang ky probe khong tra ve token' }

$client = New-Object System.Net.Http.HttpClient
$client.DefaultRequestHeaders.Authorization =
  New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $token)
$content = New-Object System.Net.Http.StringContent($json, [System.Text.Encoding]::UTF8, 'application/json')
$task = $client.PostAsync($base + '/orders', $content)
[System.Threading.Tasks.Task]::WaitAll(@($task))
$resp = $task.Result
$text = [System.Text.Encoding]::UTF8.GetString($resp.Content.ReadAsByteArrayAsync().Result)
$parsed = $text | ConvertFrom-Json

Say ("http = " + [int]$resp.StatusCode)
Say ("name expected chars = " + $name.Length + " / codepoints = " +
  (($name.ToCharArray() | ForEach-Object { [int]$_ }) -join ' '))
Say ("response decoded as UTF8 -> nameOk = " + ($parsed.customerName -eq $name) +
  " | noteOk = " + ($parsed.note -eq $note) + " | addressOk = " + ($parsed.address -eq $addr))
Say ("DB row -> " + (Sql "select 'nameLen='||length(customer_name)||' char5='||ascii(substring(customer_name,5,1))
  ||' addrLen='||length(address)||' noteLen='||length(note)||' total='||total_amount
  from orders where phone='0900000061'"))
Say ("psql sees codepoints -> " + (Sql "select array(select ascii(x) from unnest(string_to_array(customer_name,null)) x) from orders where phone='0900000061'"))
Say ("stock 2->1 = " + (Sql "select quantity from product where id=$pid3"))
Say ("order_code = " + $parsed.orderCode)
$client.Dispose()

Sql "delete from orders where phone in ('0900000061','0900000062')" | Out-Null
Sql "delete from product where sku='zzprobe3'" | Out-Null
Sql "delete from user_account where username='$probeUser'" | Out-Null
Say ("AFTER orders = " + (Sql "select count(*) from orders") + " | probe left = " +
  (Sql "select count(*) from product where sku like 'zzprobe%'") + " | baseline = " +
  (Sql "select id||'='||quantity from product where id in (2,4) order by id"))

[System.IO.File]::WriteAllLines($logPath, $out, (New-Object System.Text.UTF8Encoding $true))
Write-Host "written: $logPath"
