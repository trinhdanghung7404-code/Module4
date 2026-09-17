$ErrorActionPreference = 'Stop'

# Kiem tra xac thuc theo token that: /users/me, /users/me/recipients, /orders,
# /orders/mine, logout, han session trong bang user_session.
#
# HAI thu phai dung Encoding khi chay file nay:
#   - PS 5.1 doc file .ps1 khong co BOM theo ANSI -> moi chuoi Viet trong file
#     bi hieu sai va so sanh sai du server tra ve dung. File phai luu UTF-8 CO BOM.
#   - PS 5.1 giai ma stdout cua chuong trinh ngoai (psql) theo codepage console,
#     nen OutputEncoding + PGCLIENTENCODING phai la UTF-8.
#
# API base/DB lay tu bien moi truong de script chay duoc o may khac:
#   SMOKE_API_BASE (mac dinh http://localhost:8081/api), PSQL_EXE, PGPASSWORD.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PGCLIENTENCODING = 'UTF8'

$base = if ($env:SMOKE_API_BASE) { $env:SMOKE_API_BASE } else { 'http://localhost:8081/api' }
$psqlExe = if ($env:PSQL_EXE) { $env:PSQL_EXE } else { 'C:\Program Files\PostgreSQL\18\bin\psql.exe' }
if (-not $env:PGPASSWORD) { $env:PGPASSWORD = '123' }

$suffix = Get-Random -Maximum 99999
$userA = 'probe{0:d5}' -f $suffix
$userB = 'probeb{0:d5}' -f $suffix
$mailA = $userA + '@example.com'
$mailB = $userB + '@example.com'
$pass = 'probe-pass-123'
$phoneA = '0900000011'
$phoneB = '0900000022'

$script:passCount = 0
$script:failCount = 0

function Check($name, $cond, $detail) {
    if ($cond) {
        $script:passCount++
        Write-Output ('  ok   ' + $name)
    } else {
        $script:failCount++
        Write-Output ('  FAIL ' + $name + '  ::  ' + $detail)
    }
}

function Read-ErrorBody($response) {
    try {
        $stream = $response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        return $reader.ReadToEnd()
    } catch { return '' }
}

function Get-HeaderOf($headers, $name) {
    if ($null -eq $headers) { return '' }
    try {
        if ($headers -is [System.Collections.Specialized.NameValueCollection]) {
            $v = $headers.Get($name)
            if ($v) { return $v }
            return ''
        }
        # PS 5.1 tra Dictionary[string,string] cho response thanh cong, ma
        # Contains() la explicit interface => goi truc tiep nem MethodException.
        # Doi khi chinh exception do bi bat o ngoai va bien 201 thanh "status=0".
        $value = $null
        if ($headers.TryGetValue($name, [ref]$value)) { return [string]$value }
    } catch { return '' }
    return ''
}

# Invoke-Api: goi HTTP va LUON tra ve { status, json, text, acao } thay vi nem loi,
# de mot test 401/400 duoc viet nhu mot lenh so binh thuong.
# Origin duoc gui trong moi request vi mot test duoi day do header CORS tren chinh
# response 401 - do la toan bo ly do chan access viet la interceptor chu khong phai
# filter: filter chay truoc bo may CORS cua Spring MVC nen 401 khong kem
# Access-Control-Allow-Origin, va trinh duyet chi con bao "goi khong duoc".
function Invoke-Api($method, $path, $body, $token) {
    $headers = @{ Origin = 'http://localhost:5173' }
    if ($token) { $headers['Authorization'] = 'Bearer ' + $token }

    $params = @{
        Uri             = ($base + $path)
        Method          = $method
        UseBasicParsing = $true
        TimeoutSec      = 30
        Headers         = $headers
    }
    if ($null -ne $body) {
        # GetBytes chu khong phai truyen chuoi: PS 5.1 mac dinh ma hoa JSON body
        # bang ASCII, tieng Viet bien thanh dau ? ngay trong request.
        $params.Body = [System.Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 8 -Compress))
        $params.ContentType = 'application/json; charset=utf-8'
    }

    try {
        $r = Invoke-WebRequest @params
        $text = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())
        # ConvertFrom-Json NEM NOI khi body khong phai JSON (noi tra text/plain nhu
        # /api/admin/register). Nam ngay trong return cua try thi no bi ngoai lai
        # thanh catch -> status = 0, nom nhu "khong goi duoc may chu" trong khi
        # request van 200. Bao toan text, json = null khi khong phai JSON.
        $json = $null
        if ($text) { try { $json = $text | ConvertFrom-Json } catch { $json = $null } }
        return @{
            status = [int]$r.StatusCode
            json   = $json
            text   = $text
            acao   = Get-HeaderOf $r.Headers 'Access-Control-Allow-Origin'
        }
    } catch {
        $status = 0
        $text = ''
        $acao = ''
        if ($_.Exception.Response) {
            if ($_.Exception.Response.StatusCode) { $status = [int]$_.Exception.Response.StatusCode }
            $text = Read-ErrorBody $_.Exception.Response
            $acao = Get-HeaderOf $_.Exception.Response.Headers 'Access-Control-Allow-Origin'
        }
        if (-not $text -and $_.ErrorDetails -and $_.ErrorDetails.Message) {
            $text = $_.ErrorDetails.Message
        }
        $json = $null
        if ($text) { try { $json = $text | ConvertFrom-Json } catch { } }
        return @{ status = $status; json = $json; text = $text; acao = $acao }
    }
}

# Invoke-Preflight: mo phong dung buoc trinh duyet gui TRUOC khi goi that.
# Mot goi co header Authorization khong phai "simple request", nen browser gui
# OPTIONS kem Access-Control-Request-Method/Headers va CHI gui tiep khi preflight
# tra ve 2xx. PreFlight khong bao gio mang Authorization, nen neu interceptor chan
# no bang 401/500 thi browser bao loi mang - client chi con "khong goi duoc may
# chu" trong khi login van chay. PowerShell khong tu gui preflight, nen Invoke-Api
# khong the phat hien loai loi nay: phai goi OPTIONS thang.
function Invoke-Preflight($path, $method, $reqHeaders, $origin) {
    if (-not $origin) { $origin = 'http://localhost:5173' }
    $params = @{
        Uri             = ($base + $path)
        Method          = 'Options'
        UseBasicParsing = $true
        TimeoutSec      = 30
        Headers         = @{
            'Origin'                         = $origin
            'Access-Control-Request-Method'  = $method
            'Access-Control-Request-Headers' = $reqHeaders
        }
    }
    try {
        $r = Invoke-WebRequest @params
        return @{
            status = [int]$r.StatusCode
            acao   = Get-HeaderOf $r.Headers 'Access-Control-Allow-Origin'
            acah   = Get-HeaderOf $r.Headers 'Access-Control-Allow-Headers'
            text   = ''
        }
    } catch {
        $status = 0
        $acao = ''
        $acah = ''
        $text = ''
        if ($_.Exception.Response) {
            if ($_.Exception.Response.StatusCode) { $status = [int]$_.Exception.Response.StatusCode }
            $acao = Get-HeaderOf $_.Exception.Response.Headers 'Access-Control-Allow-Origin'
            $acah = Get-HeaderOf $_.Exception.Response.Headers 'Access-Control-Allow-Headers'
            $text = Read-ErrorBody $_.Exception.Response
        }
        return @{ status = $status; acao = $acao; acah = $acah; text = $text }
    }
}

function Sha256Hex($value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($value)) |
        ForEach-Object { $_.ToString('x2') }) -join ''
}

function DbScalar($sql) {
    return (& $psqlExe -w -h localhost -U postgres -d shop_management_db `
        -P pager=off -At -c $sql) -join ''
}

function DbExec($sql) {
    & $psqlExe -w -h localhost -U postgres -d shop_management_db -P pager=off -q -c $sql | Out-Null
}

Write-Output '== dang ky / dang nhap: phien token =='

$reg = Invoke-Api POST '/users/register' @{
    username = $userA; fullName = 'Nguyễn Văn Probe'; email = $mailA
    password = $pass; confirmPassword = $pass
}
Check '201 khi tao tai khoan' ($reg.status -eq 201) ('status=' + $reg.status + ' body=' + $reg.text)
Check 'response la { token, user }' ($reg.json.token.Length -eq 43 -and $reg.json.user.id -gt 0) $reg.text
Check 'user trong response dung tai khoản' ($reg.json.user.username -eq $userA -and $reg.json.user.email -eq $mailA) $reg.text
Check 'khong ro mat kh trong response' ($reg.text -notmatch 'password') $reg.text
$idA = $reg.json.user.id
$tokenA = $reg.json.token

# Khong duoc de chay tiep voi token rong: moi test duoi day se bao "401" va cho
# rang interceptor bi hong, trong khi loi nam o ngay buoc dang ky.
if (-not $tokenA -or -not $idA) {
    Write-Output '  FAIL dang ky khong tra ve du { token, user } - dung tai day'
    Write-Output ('       ' + $reg.text)
    exit 1
}

$dupUser = Invoke-Api POST '/users/register' @{
    username = $userA; fullName = 'Khac'; email = ('other' + $userA + '@example.com')
    password = $pass; confirmPassword = $pass
}
Check 'username trung bi tu choi' ($dupUser.status -eq 400 -and $dupUser.json.message -match 'Username') $dupUser.text

$dupEmail = Invoke-Api POST '/users/register' @{
    username = ($userA + 'x'); fullName = 'Khac'; email = ($userA.ToUpper() + '@EXAMPLE.com')
    password = $pass; confirmPassword = $pass
}
Check 'email trung hoa khong qua duoc unique' ($dupEmail.status -eq 400 -and $dupEmail.json.message -match 'Email') $dupEmail.text

$shortPass = Invoke-Api POST '/users/register' @{
    username = ($userA + 'y'); fullName = 'Khac'; email = ($userA + 'y@example.com')
    password = '123'; confirmPassword = '123'
}
Check 'mat khau ngan bi chan tu @Size' ($shortPass.status -eq 400 -and $shortPass.text -match 'khẩu') $shortPass.text

$mismatch = Invoke-Api POST '/users/register' @{
    username = ($userA + 'z'); fullName = 'Khac'; email = ($userA + 'z@example.com')
    password = $pass; confirmPassword = ($pass + 'x')
}
Check 'xac nhan khop khong qua duoc' ($mismatch.status -eq 400 -and $mismatch.json.message -match 'khớp') $mismatch.text

$badPw = Invoke-Api POST '/users/login' @{ account = $userA; password = 'sai-mat-khau-nay' }
Check 'sai mat khau -> 400' ($badPw.status -eq 400) ('status=' + $badPw.status)
Check 'sai mat khau khong lo bi mat tai khoản ton tai' ($badPw.json.message -notmatch 'tồn tại') ('leak: ' + $badPw.text)

$noSuch = Invoke-Api POST '/users/login' @{ account = ('khongco' + $userA); password = $pass }
Check 'tai khoản khong ton tai cung mot thong bao' ($noSuch.json.message -eq $badPw.json.message) ($noSuch.text + ' vs ' + $badPw.text)

$login = Invoke-Api POST '/users/login' @{ account = $mailA; password = $pass }
Check 'dang nhap bang email duoc' ($login.status -eq 200 -and $login.json.user.id -eq $idA) $login.text
$tokenA2 = $login.json.token

$loginUpper = Invoke-Api POST '/users/login' @{ account = ($userA.ToUpper()); password = $pass }
Check 'dang nhap bang USERNAME VIET HOA van duoc' ($loginUpper.status -eq 200) $loginUpper.text

Check 'hai lan dang nhap ra hai token khac nhau' ($tokenA -ne $tokenA2) ($tokenA + ' vs ' + $tokenA2)
$meSecond = Invoke-Api GET '/users/me' $null $tokenA2
Check 'ca hai phien deu con song (logout may nay khong da may khac)' ($meSecond.status -eq 200 -and $meSecond.json.id -eq $idA) $meSecond.text

$regB = Invoke-Api POST '/users/register' @{
    username = $userB; fullName = 'Trần Thị Probe'; email = $mailB
    password = $pass; confirmPassword = $pass
}
$idB = $regB.json.user.id
$tokenB = $regB.json.token
Check 'tao tai khoản thu hai de do criss-cross' ($idB -gt 0 -and $idB -ne $idA) $regB.text

Write-Output '== DB chi luu hash, khong luu token =='

$hashA = Sha256Hex $tokenA
Check 'token_hash la SHA-256 hex 64 ky tu' ($hashA.Length -eq 64) $hashA
Check 'dong session trung voi hash vua tinh' `
    ((DbScalar ("select count(*) from user_session where token_hash='" + $hashA + "' and user_id=" + $idA)) -eq '1') $hashA
Check 'token goc khong xuat hien o bat ky cot nao' `
    ((DbScalar ("select count(*) from user_session where token_hash='" + $tokenA + "'")) -eq '0') 'token goc da nam trong DB'
# register + login bang email + login bang USERNAME HOA = ba lan phat hanh.
Check 'moi tai khoản co 3 phien sau 3 lan dang nhap' `
    ((DbScalar ("select count(*) from user_session where user_id=" + $idA)) -eq '3') ('count=' + (DbScalar ("select count(*) from user_session where user_id=" + $idA)))

Write-Output '== chan 401 =='

$noToken = Invoke-Api GET '/users/me' $null
Check '/users/me khong token -> 401' ($noToken.status -eq 401) ('status=' + $noToken.status + ' ' + $noToken.text)
Check '401 van kem header CORS (ly do dung interceptor, khong phai filter)' ($noToken.acao -eq 'http://localhost:5173') ('acao=[' + $noToken.acao + ']')
Check '401 bao dung loi chua dang nhap' ($noToken.json.message -match 'Chưa đăng nhập') $noToken.text

$badToken = Invoke-Api GET '/users/me' $null 'day-khong-phai-token-that'
Check 'token la -> 401' ($badToken.status -eq 401) $badToken.text
Check 'token la bao het han chu khong bao chua dang nhap' ($badToken.json.message -match 'không còn hiệu lực') $badToken.text

$emptyBearer = Invoke-Api GET '/users/me' $null '   '
Check 'Bearer rong -> 401' ($emptyBearer.status -eq 401) $emptyBearer.text

$longToken = Invoke-Api GET '/users/me' $null ('x' * 400)
Check 'token dai bat thuong -> 401, khong bi hash o phi' ($longToken.status -eq 401) ('status=' + $longToken.status)

$me = Invoke-Api GET '/users/me' $null $tokenA
Check '/users/me voi token -> 200' ($me.status -eq 200 -and $me.json.id -eq $idA) $me.text
Check '/users/me khong tra mat khau' ($me.text -notmatch 'password') $me.text

$oldPath = Invoke-Api GET ('/users/' + $idA + '/recipients') $null $tokenA
Check 'duong dan cu /users/{id}/recipients khong con ton tai -> 404' ($oldPath.status -eq 404) ('status=' + $oldPath.status + ' ' + $oldPath.text)

$noTokenList = Invoke-Api GET '/users/me/recipients' $null
Check '/users/me/recipients khong token -> 401' ($noTokenList.status -eq 401) ('status=' + $noTokenList.status)

Write-Output '== preflight CORS: buoc trinh duyet goi truoc khi goi that =='

# Preflight phai 2xx. Day la loai loi Invoke-Api khong bao gio thay: PowerShell goi
# thang GET/POST khong kem OPTIONS, con browser thi dung lai toan bo response neu
# OPTIONS bi chan. Mot interceptor chan "duong dan can dang nhap" rat de chan nien
# ca OPTIONS vi do la cung mot duong dan.
$pfMe = Invoke-Preflight '/users/me' 'GET' 'authorization'
Check 'OPTIONS /users/me -> 2xx (khong bi chan thanh 401/500)' ($pfMe.status -ge 200 -and $pfMe.status -lt 300) ('status=' + $pfMe.status + ' body=' + $pfMe.text)
Check 'preflight /users/me tra Access-Control-Allow-Origin' ($pfMe.acao -eq 'http://localhost:5173') ('acao=[' + $pfMe.acao + ']')
Check 'preflight /users/me cho gui header authorization' ($pfMe.acah -match 'authorization') ('ACAH=[' + $pfMe.acah + ']')

$pfOrders = Invoke-Preflight '/orders' 'POST' 'authorization,content-type'
Check 'OPTIONS /orders -> 2xx (dat hang goi duoc tu browser)' ($pfOrders.status -ge 200 -and $pfOrders.status -lt 300) ('status=' + $pfOrders.status + ' body=' + $pfOrders.text)
Check 'preflight /orders cho gui authorization + content-type' ($pfOrders.acah -match 'authorization' -and $pfOrders.acah -match 'content-type') ('ACAH=[' + $pfOrders.acah + ']')

$pfMine = Invoke-Preflight '/orders/mine' 'GET' 'authorization'
Check 'OPTIONS /orders/mine -> 2xx' ($pfMine.status -ge 200 -and $pfMine.status -lt 300) ('status=' + $pfMine.status)

$pfRecipients = Invoke-Preflight '/users/me/recipients' 'DELETE' 'authorization'
Check 'OPTIONS DELETE /users/me/recipients/{id} -> 2xx' ($pfRecipients.status -ge 200 -and $pfRecipients.status -lt 300) ('status=' + $pfRecipients.status)

$pfLogin = Invoke-Preflight '/users/login' 'POST' 'content-type'
Check 'OPTIONS /users/login -> 2xx (endpoint cong khai van phai dat chuan)' ($pfLogin.status -ge 200 -and $pfLogin.status -lt 300) ('status=' + $pfLogin.status)

# Chieu nguoc lai: origin khong thuoc danh sach phai bi tu choi VA khong duoc
# gia vao Access-Control-Allow-Origin, neu khong thi loi "khong goi duoc may chu"
# se den tu mot nguyen nhan khac ma khong ai doc ra.
$pfForeign = Invoke-Preflight '/users/me' 'GET' 'authorization' 'http://127.0.0.1:5174'
Check 'origin 127.0.0.1:5174 bi tu choi va khong duoc echo ACAO' ($pfForeign.status -ge 400 -and -not $pfForeign.acao) ('status=' + $pfForeign.status + ' acao=[' + $pfForeign.acao + ']')

Write-Output '== so nguoi nhan theo token =='

$r1 = Invoke-Api POST '/users/me/recipients' @{
    name = 'Nguyễn Văn Probe'; phone = $phoneA; address = '12 Lê Lợi, Quận 1, TP.HCM'; label = 'Nhà'
} $tokenA
Check 'nguoi nhan dau tien tu dong la mac dinh' ($r1.status -eq 201 -and $r1.json.isDefault -eq $true) $r1.text
$rid1 = $r1.json.id
Check 'giu nguyen tieng Viet co dau' ($r1.json.address -eq '12 Lê Lợi, Quận 1, TP.HCM') $r1.text

$r2 = Invoke-Api POST '/users/me/recipients' @{
    name = 'Bùi Đức Thắng'; phone = '0900000033'; address = '76 Nguyễn Trãi, Quận 5'
    label = 'Cơ quan'; isDefault = $true
} $tokenA
$rid2 = $r2.json.id
Check 'them mac dinh moi -> 201' ($r2.status -eq 201 -and $r2.json.isDefault -eq $true) $r2.text

$list = Invoke-Api GET '/users/me/recipients' $null $tokenA
$defaults = @($list.json | Where-Object { $_.isDefault })
Check 'chi mot nguoi nhan mac dinh' ($defaults.Count -eq 1 -and $defaults[0].id -eq $rid2) $list.text
Check 'xep hang: mac dinh len dau' ($list.json[0].id -eq $rid2) $list.text

$dup = Invoke-Api POST '/users/me/recipients' @{
    name = 'Bùi Đức Thắng'; phone = '0900000033'; address = '76 Nguyễn Trãi, Quận 5'
} $tokenA
Check 'nguoi nhan trung lap bi tu choi' ($dup.status -eq 400 -and $dup.json.message -match 'đã có') $dup.text

$badPhone = Invoke-Api POST '/users/me/recipients' @{
    name = 'Ai Do'; phone = 'abc'; address = 'dau do'
} $tokenA
Check 'so dien thoai sai bi chan tu @Valid' ($badPhone.status -eq 400 -and $badPhone.text -match 'điện thoại') $badPhone.text

$edit = Invoke-Api PUT ('/users/me/recipients/' + $rid1) @{
    name = 'Nguyễn Văn Probe'; phone = $phoneA; address = '12 Lê Lợi, Quận 1, TP.HCM'; label = 'Nhà cũ'
} $tokenA
Check 'sua khong mang isDefault thi gi co mac dinh cu' ($edit.status -eq 200 -and $edit.json.isDefault -eq $false -and $edit.json.label -eq 'Nhà cũ') $edit.text

$setDefault = Invoke-Api PUT ('/users/me/recipients/' + $rid1 + '/default') $null $tokenA
$sdDefaults = @($setDefault.json | Where-Object { $_.isDefault })
Check 'doi mac dinh -> tra ve danh sach moi, dung mot mac dinh' ($setDefault.status -eq 200 -and $sdDefaults.Count -eq 1 -and $sdDefaults[0].id -eq $rid1) $setDefault.text

Write-Output '== token cua nguoi khac khong mo duoc so cua nguoi minh =='

$bList = Invoke-Api GET '/users/me/recipients' $null $tokenB
Check 'B dang nhap thi so nguoi nhan trang rong' (@($bList.json).Count -eq 0) $bList.text

$crossPut = Invoke-Api PUT ('/users/me/recipients/' + $rid1) @{
    name = 'Đánh cắp'; phone = '0900000044'; address = 'dia chi khac'
} $tokenB
Check 'khong sua duoc nguoi nhan cua tai khoản khac' ($crossPut.status -eq 400 -and $crossPut.json.message -match 'không thuộc') $crossPut.text

$crossDelete = Invoke-Api DELETE ('/users/me/recipients/' + $rid1) $null $tokenB
Check 'khong xoa duoc nguoi nhan cua tai khoản khac' ($crossDelete.status -eq 400) ('status=' + $crossDelete.status)

$crossDefault = Invoke-Api PUT ('/users/me/recipients/' + $rid1 + '/default') $null $tokenB
Check 'khong dat duoc mac dinh cua nguoi khac' ($crossDefault.status -eq 400) ('status=' + $crossDefault.status)

$ghost = Invoke-Api PUT ('/users/me/recipients/99999999') @{
    name = 'Ai Do'; phone = '0900000066'; address = 'dia chi khac'
} $tokenA
Check 'recipientId khong ton tai -> 400' ($ghost.status -eq 400) ('status=' + $ghost.status)

$untouched = Invoke-Api GET '/users/me/recipients' $null $tokenA
$stillThere = @($untouched.json | Where-Object { $_.id -eq $rid1 })
Check 'nhung lan do cross khong thay doi duoc du lieu that' ($stillThere.Count -eq 1 -and $stillThere[0].address -eq '12 Lê Lợi, Quận 1, TP.HCM' -and $stillThere[0].label -eq 'Nhà cũ') $untouched.text

Write-Output '== don hang theo token =='

# Ton kho o day la du lieu THAT: hai don duoi day tieu hao 2 cai id=2 va 2 cai id=4,
# va don cua probe bi xoa cuc o cuoi script nhung KHONG hoan lai hang. Cu the chay
# nhieu lan thi kho ve 0, va toan bo nhanh test don hang fail day chuyen voi 400
# "con 0 cai" - nom nhu loi xac thuc trong khi chi la test khong tu duyet. Nen ghi
# gia tri goc lai, tam day len du dung, va tra lai khi ket thuc.
$stockSnap = DbScalar "select coalesce(string_agg(id || '=' || quantity, ',' order by id), '') from product where id in (2, 4)"
DbExec 'update product set quantity = quantity + 10 where id in (2, 4)'

$p2 = Invoke-Api GET '/products/2' $null
$p4 = Invoke-Api GET '/products/4' $null
$stockBefore = $p2.json.quantity
$expectedTotal = [decimal]$p2.json.price + 2 * [decimal]$p4.json.price

$guestOrder = Invoke-Api POST '/orders' @{
    customerName = 'Khách Vãng Lai'; phone = '0900000055'; address = '45 Hai Bà Trưng'
    items = @(@{ productId = 2; quantity = 1 })
}
Check 'khong dang nhap thi dat hang bi tu choi 401' ($guestOrder.status -eq 401) ('status=' + $guestOrder.status + ' ' + $guestOrder.text)
Check '401 dat hang cung co header CORS' ($guestOrder.acao -eq 'http://localhost:5173') ('acao=[' + $guestOrder.acao + ']')
$afterReject = Invoke-Api GET '/products/2' $null
Check 'don bi chan tu 401 khong lam mat hang trong kho' ($afterReject.json.quantity -eq $stockBefore) ($stockBefore.ToString() + ' -> ' + $afterReject.json.quantity.ToString())

$order = Invoke-Api POST '/orders' @{
    customerName = 'Bùi Đức Thắng'; phone = '0900000033'; address = '76 Nguyễn Trãi, Quận 5'
    note = 'gọi trước khi giao'
    items = @(@{ productId = 2; quantity = 1 }, @{ productId = 4; quantity = 2 })
} $tokenA
Check 'dat hang voi token -> 201' ($order.status -eq 201) ('status=' + $order.status + ' ' + $order.text)
Check 'tong tien do server tinh tu gia DB' ([decimal]$order.json.totalAmount -eq $expectedTotal) ('total=' + $order.json.totalAmount + ' expect=' + $expectedTotal)
$orderCode = $order.json.orderCode

# Body ghi de gia/tong ten va tu khai userId: khong thuoc truong hop nao duoc
# server tin. OrderCreateRequest khong con cot userId, nen Jackson bo qua;
# gia la do server tinh lai tu DB.
$forge = Invoke-Api POST '/orders' @{
    userId = $idB; totalAmount = 1; customerName = 'Kẻ Gian Lận'; phone = '0900000077'
    address = 'dia chi khac'
    items = @(@{ productId = 2; quantity = 1; price = 1; lineTotal = 1; productName = ' Gia ' })
} $tokenA
Check 'don co gia/userId gia van 201 (khach khong quyet duoc no)' ($forge.status -eq 201) ('status=' + $forge.status + ' ' + $forge.text)
Check 'tong tien do server tinh, khong phai khach khai' ([decimal]$forge.json.totalAmount -eq [decimal]$p2.json.price) ('total=' + $forge.json.totalAmount)
Check 'ten san pham trong don la ten that tu DB' ($forge.json.items[0].productName -eq $p2.json.name) ('item=' + ($forge.json.items[0] | ConvertTo-Json -Compress))

$mineA = Invoke-Api GET '/orders/mine' $null $tokenA
Check '/orders/mine voi token -> 200' ($mineA.status -eq 200) ('status=' + $mineA.status)
$mineCodesA = @($mineA.json | ForEach-Object { $_.orderCode })
Check '/orders/mine thay don minh vua dat' ($mineCodesA -contains $orderCode -and $mineCodesA -contains $forge.json.orderCode) ($mineCodesA -join ',')
Check '/orders/mine xep don moi nhat len dau' ($mineCodesA[0] -eq $forge.json.orderCode) ($mineCodesA -join ',')

$mineB = Invoke-Api GET '/orders/mine' $null $tokenB
$mineCodesB = @($mineB.json | ForEach-Object { $_.orderCode })
Check 'don cua A khong xuat hien trong /orders/mine cua B' (-not ($mineCodesB -contains $orderCode)) ($mineCodesB -join ',')

$mineNoToken = Invoke-Api GET '/orders/mine' $null
Check '/orders/mine khong token -> 401' ($mineNoToken.status -eq 401) ('status=' + $mineNoToken.status)

$dbOwner = DbScalar ("select coalesce(user_id::text,'NULL') from orders where order_code='" + $orderCode + "'")
Check 'orders.user_id lay tu token chu khong tu body' ($dbOwner -eq [string]$idA) ('db=' + $dbOwner + ' expect=' + $idA)
$forgeOwner = DbScalar ("select coalesce(user_id::text,'NULL') from orders where order_code='" + $forge.json.orderCode + "'")
Check 'userId gia trong body khong doi duoc chu don' ($forgeOwner -eq [string]$idA) ('db=' + $forgeOwner)
$snapshotName = DbScalar ("select customer_name from orders where order_code='" + $orderCode + "'")
Check 'ten nguoi nhan luu vao don khong phai ten tai khoản' ($snapshotName -eq 'Bùi Đức Thắng') ('db=' + $snapshotName)
$defaultRows = DbScalar 'select count(*) from (select user_id from recipient where is_default group by user_id having count(*) > 1) t'
Check 'DB khong the co hai mac dinh moi tai khoản' ($defaultRows -eq '0') ('rows=' + $defaultRows)

Write-Output '== xoa va dem mac dinh len =='

$del = Invoke-Api DELETE ('/users/me/recipients/' + $rid1) $null $tokenA
Check 'xoa nguoi nhan mac dinh -> 204' ($del.status -eq 204) ('status=' + $del.status)

$afterDelete = Invoke-Api GET '/users/me/recipients' $null $tokenA
$promoted = @($afterDelete.json | Where-Object { $_.isDefault })
Check 'nguoi con lai duoc dem len lam mac dinh' ($promoted.Count -eq 1 -and $promoted[0].id -eq $rid2) $afterDelete.text

$mineCount = DbScalar ("select count(*) from orders where user_id=" + $idA)
Check 'truy duoc don theo user_id' ($mineCount -eq '2') ('count=' + $mineCount)

Write-Output '== migration 04_user_session =='

$idxCount = DbScalar "select count(*) from pg_indexes where tablename='user_session'"
Check 'bang co index phuc vu tra token va don' ([int]$idxCount -ge 3) ('indexes=' + $idxCount)
$nullable = DbScalar "select count(*) from information_schema.columns where table_name='user_session' and column_name in ('token_hash','user_id','created_at','expires_at')"
Check 'du cot {token_hash, user_id, created_at, expires_at}' ($nullable -eq '4') ('cols=' + $nullable)
$ttlOk = DbScalar ("select count(*) from user_session where user_id=" + $idB + " and expires_at > now() + interval '29 days'")
Check 'expires_at dung TTL 30 ngay cua app.user.session-ttl-days' ($ttlOk -eq '1') ('rows=' + $ttlOk)

Write-Output '== logout va han session =='

$logout = Invoke-Api POST '/users/logout' $null $tokenA
Check 'logout -> 204' ($logout.status -eq 204) ('status=' + $logout.status + ' ' + $logout.text)

$afterLogout = Invoke-Api GET '/users/me' $null $tokenA
Check 'token da logout bi thu hoi ngay lap tuc' ($afterLogout.status -eq 401) ('status=' + $afterLogout.status)
$afterLogoutOrders = Invoke-Api GET '/orders/mine' $null $tokenA
Check 'logout xong khong con doc duoc don' ($afterLogoutOrders.status -eq 401) ('status=' + $afterLogoutOrders.status)
$afterLogoutPost = Invoke-Api POST '/orders' @{
    customerName = 'Ai Do'; phone = '0900000088'; address = 'dau do'
    items = @(@{ productId = 2; quantity = 1 })
} $tokenA
Check 'logout xong khong dat duoc nua' ($afterLogoutPost.status -eq 401) ('status=' + $afterLogoutPost.status)
$againLogout = Invoke-Api POST '/users/logout' $null $tokenA
Check 'logout lan hai voi token da chet -> 401 (client tu clear local)' ($againLogout.status -eq 401) ('status=' + $againLogout.status)

$otherSession = Invoke-Api GET '/users/me' $null $tokenA2
Check 'phien khac cung tai khoản van song sau logout' ($otherSession.status -eq 200 -and $otherSession.json.id -eq $idA) ('status=' + $otherSession.status)
# Ba phien duoc phat hanh o tren, logout thu hoi dung mot -> con hai.
$leftSessions = DbScalar ("select count(*) from user_session where user_id=" + $idA)
Check 'logout chi xoá dung mot dong session' ($leftSessions -eq '2') ('count=' + $leftSessions)

# Han session: khong can cho den 30 ngay - git expires_at trong DB lui lai la
# token phai chet ngay. Day la thu duy nhat chung minh expires_at duoc ton trong
# chu khong chi la cot de trang.
DbExec ("update user_session set expires_at = now() - interval '1 minute' where user_id=" + $idA)
$expired = Invoke-Api GET '/users/me' $null $tokenA2
Check 'token qua expires_at -> 401' ($expired.status -eq 401) ('status=' + $expired.status + ' ' + $expired.text)

# issue() don ran het han: dang nhap lai thi dong chet phai bi xoa.
$relogin = Invoke-Api POST '/users/login' @{ account = $userA; password = $pass }
Check 'dang nhap lai sau khi het han -> co token moi' ($relogin.status -eq 200 -and $relogin.json.token.Length -eq 43) $relogin.text
$purged = DbScalar ("select count(*) from user_session where user_id=" + $idA)
Check 'phat han token moi tien tay don session het han' ($purged -eq '1') ('count=' + $purged)

Write-Output '== don va session de lai sau khi xoa tai khoản =='

DbExec ("delete from user_account where id in (" + $idA + "," + $idB + ")")
$orphan = DbScalar ("select coalesce(user_id::text,'NULL') from orders where order_code='" + $orderCode + "'")
Check 'xoa tai khoản thi don van con, user_id thanh NULL' ($orphan -eq 'NULL') ('db=' + $orphan)
$deadSessions = DbScalar ("select count(*) from user_session where user_id in (" + $idA + "," + $idB + ")")
Check 'xoa tai khoản thi session cung bi xoá theo (FK cascade)' ($deadSessions -eq '0') ('count=' + $deadSessions)
$deadRecipients = DbScalar ("select count(*) from recipient where user_id in (" + $idA + "," + $idB + ")")
Check 'so nguoi nhan cung bi xoá theo tai khoản' ($deadRecipients -eq '0') ('count=' + $deadRecipients)

# Don cua script mang dien thoai probe: xoa de moi lan chay khong dem dan so
# don trong trang quan tri. order_item bi xoá theo theo FK cascade.
DbExec "delete from orders where phone in ('0900000033','0900000077','0900000055')"

# Tra lai dung ton kho ban dau. Test phai ty duyet: ket qua lan chay nay khong
# duoc phu thuoc vao viec da chay bao nhieu lan, va san pham demo khong bi mon
# dan den khi khong dat duoc nua.
foreach ($pair in ($stockSnap -split ',')) {
    if (-not $pair) { continue }
    $parts = $pair -split '='
    DbExec ("update product set quantity = " + [int]$parts[1] + " where id = " + [int]$parts[0])
}
$stockRestored = DbScalar "select coalesce(string_agg(id || '=' || quantity, ',' order by id), '') from product where id in (2, 4)"
Check 'ton kho demo duoc tra lai dung nhu truoc khi chay' ($stockRestored -eq $stockSnap) ('truoc=' + $stockSnap + '  sau=' + $stockRestored)

Write-Output ''
Write-Output ('PASS ' + $script:passCount + '  FAIL ' + $script:failCount)
if ($script:failCount -gt 0) { exit 1 }
