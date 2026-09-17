$ErrorActionPreference = 'Stop'

# Kiem tra LUONG DON HANG KEP KIN: PENDING -> CONFIRMED -> SHIPPING -> DELIVERED,
# nhanh CANCELLED hoan ton kho, lich su trang thai, may tran trang thai bi tu choi,
# va xac thuc THAT phia admin (token + preflight CORS).
#
# Vi sao phai la script rieng chu khong them vao user-check.ps1: user-check do
# quyen cua KHACH, con file nay do quyen cua ADMIN + nghiep vu don hang. Hai nhom
# assertion khong chung cai nao ngoai bo ham goi HTTP.
#
# HAI thu phai dung Encoding khi chay file nay (giong user-check.ps1):
#   - PS 5.1 doc file .ps1 khong co BOM theo ANSI -> moi chuoi Viet trong file bi
#     hieu sai va so sanh sai du server tra ve dung. File phai luu UTF-8 CO BOM.
#   - PS 5.1 giai ma stdout cua chuong trinh ngoai (psql) theo codepage console,
#     nen OutputEncoding + PGCLIENTENCODING phai la UTF-8.
#
# Bien moi truong: SMOKE_API_BASE (mac dinh http://localhost:8081/api), PSQL_EXE,
# PGPASSWORD. Can mot backend dang chay tren port do.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# May tien do "Reading web response" cua Invoke-WebRequest lam ban doan output khi
# chay ca thang test; im lang no di de file ket qua doc duoc.
$ProgressPreference = 'SilentlyContinue'
$env:PGCLIENTENCODING = 'UTF8'

$base = if ($env:SMOKE_API_BASE) { $env:SMOKE_API_BASE } else { 'http://localhost:8081/api' }
$psqlExe = if ($env:PSQL_EXE) { $env:PSQL_EXE } else { 'C:\Program Files\PostgreSQL\18\bin\psql.exe' }
if (-not $env:PGPASSWORD) { $env:PGPASSWORD = '123' }

$suffix = Get-Random -Maximum 99999
$adminUser = 'smokeadm{0:d5}' -f $suffix
$adminMail = $adminUser + '@example.com'
$adminPass = 'smoke-admin-pass-123'
$customerUser = 'smokecust{0:d5}' -f $suffix
$customerMail = $customerUser + '@example.com'
$smokePhone = '0900009911'
$cancelPhone = '0900009922'

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
        $value = $null
        if ($headers.TryGetValue($name, [ref]$value)) { return [string]$value }
    } catch { return '' }
    return ''
}

# Invoke-Api: goi HTTP va LUON tra ve { status, json, text, acao } thay vi nem loi,
# de mot test 401/400 duoc viet nhu mot lenh so binh thuong.
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
        # ConvertFrom-Json NEM NOI khi body khong phai JSON (dang ky admin tra
        # text/plain "Dang ky thanh cong"). Nam trong return cua try thi ngoai lai
        # cuoc thanh catch -> status = 0, nom nhu khong goi duoc may chu.
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

# Invoke-Preflight: mo phong dung buoc trinh duyet gui TRUOC khi goi that. Day la
# lop bao ve chong lai chinh cai loi "Khong goi duoc may chu": PowerShell khong tu
# gui preflight, nen chi co goi OPTIONS thang moi phat hien interceptor chan no.
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

function DbScalar($sql) {
    return (& $psqlExe -w -h localhost -U postgres -d shop_management_db `
        -P pager=off -At -c $sql) -join ''
}

function DbExec($sql) {
    & $psqlExe -w -h localhost -U postgres -d shop_management_db -P pager=off -q -c $sql | Out-Null
}

function Sha256Hex($value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($value)) |
        ForEach-Object { $_.ToString('x2') }) -join ''
}

# ---------------------------------------------------------------------------
# 1) PREFLIGHT. Day chinh la cho da gay ra banner "Khong goi duoc may chu" phia
#    khach hang: trinh duyet gui OPTIONS ma khong kem Authorization, nen interceptor
#    chan no bang 401 la browser bao "goi khong duoc", khong phai "chua dang nhap".
#    AdminSessionInterceptor phai cho OPTIONS di qua nhu UserSessionInterceptor.
# ---------------------------------------------------------------------------
$pfOrders = Invoke-Preflight '/admin/orders' 'GET' 'authorization'
Check 'OPTIONS /admin/orders -> 2xx (khong bi chan 401)' ($pfOrders.status -ge 200 -and $pfOrders.status -lt 300) ('status=' + $pfOrders.status + ' ' + $pfOrders.text)
Check 'preflight /admin/orders tra dung origin duoc phep' ($pfOrders.acao -eq 'http://localhost:5173') ('acao=[' + $pfOrders.acao + ']')
Check 'preflight cho phep header authorization' ($pfOrders.acah -match '(?i)authorization') ('acah=[' + $pfOrders.acah + ']')

$pfShip = Invoke-Preflight '/admin/orders/1/status' 'POST' 'authorization,content-type'
Check 'OPTIONS /admin/orders/{id}/status -> 2xx' ($pfShip.status -ge 200 -and $pfShip.status -lt 300) ('status=' + $pfShip.status + ' ' + $pfShip.text)

$pfForeign = Invoke-Preflight '/admin/orders' 'GET' 'authorization' 'http://evil.example'
Check 'origin la bi tu choi va khong echo header' ($pfForeign.acao -eq '') ('acao=[' + $pfForeign.acao + ']')

# ---------------------------------------------------------------------------
# 2) /api/admin/** that su phai co token.
# ---------------------------------------------------------------------------
$noToken = Invoke-Api GET '/admin/orders' $null
Check 'GET /admin/orders khong token -> 401' ($noToken.status -eq 401) ('status=' + $noToken.status + ' ' + $noToken.text)
Check '401 admin van mang header CORS (filter se mat dieu nay)' ($noToken.acao -eq 'http://localhost:5173') ('acao=[' + $noToken.acao + ']')
Check '401 kem thong bao ro rang chu khong rong' ($noToken.json.message -eq 'Chưa đăng nhập quản trị. Vui lòng đăng nhập lại.') ('text=' + $noToken.text)

$badToken = Invoke-Api GET '/admin/orders' $null 'day-khong-phai-token-that'
Check 'token la -> 401' ($badToken.status -eq 401) ('status=' + $badToken.status)

$productsNoToken = Invoke-Api GET '/admin/products' $null
Check '/admin/products cung da phai dang nhap (truoc day mo hoan toan)' ($productsNoToken.status -eq 401) ('status=' + $productsNoToken.status)

# ---------------------------------------------------------------------------
# 3) Dang ky / dang nhap admin.
# ---------------------------------------------------------------------------
$wrongBefore = Invoke-Api POST '/admin/login' @{ account = $adminUser; password = 'mat-khau-sai-123' }
Check 'tai khoan chua ton tai -> 400' ($wrongBefore.status -eq 400) ('status=' + $wrongBefore.status + ' ' + $wrongBefore.text)

$register = Invoke-Api POST '/admin/register' @{
    username = $adminUser; fullName = 'Smoke Admin'; email = $adminMail
    password = $adminPass; confirmPassword = $adminPass
}
Check 'dang ky admin smoke -> 200' ($register.status -eq 200) ('status=' + $register.status + ' ' + $register.text)

$wrongAfter = Invoke-Api POST '/admin/login' @{ account = $adminUser; password = 'mat-khau-sai-123' }
Check 'sai mat khau cua tai khoan CO that -> cung mot thong bao' ($wrongAfter.json.message -eq $wrongBefore.json.message) ('truoc=' + $wrongBefore.json.message + ' | sau=' + $wrongAfter.json.message)

$login = Invoke-Api POST '/admin/login' @{ account = $adminUser; password = $adminPass }
Check 'dang nhap admin -> 200' ($login.status -eq 200) ('status=' + $login.status + ' ' + $login.text)
$adminToken = $login.json.token
Check 'dang nhap tra ve token' ($adminToken -and $adminToken.Length -ge 40) ('token=' + $adminToken)
Check 'token nam trong phan admin cua response' ($login.json.admin.username -eq $adminUser) ($login.text)

$hashStored = DbScalar ("select count(*) from admin_session where token_hash = '" + (Sha256Hex $adminToken) + "'")
Check 'DB luu HASH cua token, khong luu token goc' ($hashStored -eq '1') ('rows=' + $hashStored)
$plainStored = DbScalar ("select count(*) from admin_session where token_hash = '" + $adminToken + "'")
Check 'khong dong nao trong admin_session chua token nguyen ven' ($plainStored -eq '0') ('rows=' + $plainStored)

$me = Invoke-Api GET '/admin/me' $null $adminToken
Check 'GET /admin/me voi token -> 200 dung nguoi' ($me.status -eq 200 -and $me.json.username -eq $adminUser) ('status=' + $me.status + ' ' + $me.text)

# ---------------------------------------------------------------------------
# 4) Khach dat hang, roi admin chay tung buoc.
# ---------------------------------------------------------------------------
$cust = Invoke-Api POST '/users/register' @{
    username = $customerUser; email = $customerMail; fullName = 'Khách Smoke'
    password = $adminPass; confirmPassword = $adminPass
}
Check 'dang ky khach smoke -> 201' ($cust.status -eq 201) ('status=' + $cust.status + ' ' + $cust.text)
$customerToken = $cust.json.token

# Ton kho la du lieu THAT chung voi moi lan chay: snapshot 2/4 va tra lai o cuoi
# script, neu khong chay lan thu 50 thi "con 0 cai" va ca nhanh test do that bai
# nom nhu loi xac thuc (bai hoc da tra gia voi user-check.ps1).
$stockSnap = DbScalar "select coalesce(string_agg(id || '=' || quantity, ',' order by id), '') from product where id in (2, 4)"
DbExec 'update product set quantity = quantity + 30 where id in (2, 4)'

$p2 = Invoke-Api GET '/products/2' $null
$stockBefore = [int]$p2.json.quantity

$order = Invoke-Api POST '/orders' @{
    customerName = 'Khách Smoke'; phone = $smokePhone; address = '09 Smoke Street'
    items = @(@{ productId = 2; quantity = 3 })
} $customerToken
Check 'khach dat hang -> 201' ($order.status -eq 201) ('status=' + $order.status + ' ' + $order.text)
$orderId = $order.json.id
$orderCode = $order.json.orderCode
Check 'don moi ro ra PENDING' ($order.json.status -eq 'PENDING') ($order.text)
Check 'server gui ca nhan tieng Viet cho trang thai' ($order.json.statusLabel -eq 'Chờ xác nhận') ($order.text)
$firstLine = $order.json.history[0]
Check 'dat hang tao dong lich su dau tien, actor USER' ($order.json.history.Count -eq 1 -and $firstLine.toStatus -eq 'PENDING' -and $firstLine.actorType -eq 'USER') ($order.text)
Check 'PENDING mo duong sang CONFIRMED va CANCELLED' (@($order.json.nextStatuses) -contains 'CONFIRMED' -and @($order.json.nextStatuses) -contains 'CANCELLED') ($order.json.nextStatuses -join ',')

$afterOrder = Invoke-Api GET '/products/2' $null
Check 'dat hang tru dung 3 khoi ton' ([int]$afterOrder.json.quantity -eq ($stockBefore - 3)) ($stockBefore.ToString() + ' -> ' + $afterOrder.json.quantity)

$asCustomer = Invoke-Api POST ('/admin/orders/' + $orderId + '/status') @{ toStatus = 'CONFIRMED' } $customerToken
Check 'token KHACH khong mo duoc endpoint admin' ($asCustomer.status -eq 401) ('status=' + $asCustomer.status + ' ' + $asCustomer.text)

$confirmUrl = '/admin/orders/' + $orderId + '/status'
$confirm = Invoke-Api POST $confirmUrl @{ toStatus = 'CONFIRMED'; note = 'Da goi dien thoai xac nhan' } $adminToken
Check 'PENDING -> CONFIRMED -> 200' ($confirm.status -eq 200 -and $confirm.json.status -eq 'CONFIRMED') ('status=' + $confirm.status + ' ' + $confirm.text)
Check 'lich su len 2 dong' (@($confirm.json.history).Count -eq 2) ($confirm.json.history.Count)
$adminLine = $confirm.json.history[1]
Check 'dong lich su ghi ro tu dau sang dau' ($adminLine.fromStatus -eq 'PENDING' -and $adminLine.toStatus -eq 'CONFIRMED') ($adminLine | ConvertTo-Json -Compress)
Check 'dong lich su ghi ten admin lam' ($adminLine.actorType -eq 'ADMIN' -and $adminLine.actorLabel -eq $adminUser) ($adminLine | ConvertTo-Json -Compress)
Check 'ghi chu cua admin duoc luu lai' ($adminLine.note -eq 'Da goi dien thoai xac nhan') ($adminLine.note)
Check 'CONFIRMED con mo SHIPPING va CANCELLED' (@($confirm.json.nextStatuses) -contains 'SHIPPING' -and @($confirm.json.nextStatuses) -contains 'CANCELLED') ($confirm.json.nextStatuses -join ',')

$midStock = Invoke-Api GET '/products/2' $null
Check 'xac nhan don khong cham vao ton kho' ([int]$midStock.json.quantity -eq ($stockBefore - 3)) ($midStock.json.quantity)

$sameAgain = Invoke-Api POST $confirmUrl @{ toStatus = 'CONFIRMED' } $adminToken
Check 'chuyen don dang o cung trang thai -> 400' ($sameAgain.status -eq 400) ('status=' + $sameAgain.status + ' ' + $sameAgain.text)
Check '400 do noi ro ly do trung trang thai' ($sameAgain.json.message -match 'không cần chuyển lại') ($sameAgain.json.message)

$backwards = Invoke-Api POST $confirmUrl @{ toStatus = 'PENDING' } $adminToken
Check 'quay lui CONFIRMED -> PENDING bi tu choi' ($backwards.status -eq 400) ('status=' + $backwards.status)
$jumpAhead = Invoke-Api POST $confirmUrl @{ toStatus = 'DELIVERED' } $adminToken
Check 'nhay qua buoc CONFIRMED -> DELIVERED bi tu choi' ($jumpAhead.status -eq 400) ('status=' + $jumpAhead.status)
Check '400 ke ten cac buoc duoc phep lam' ($jumpAhead.json.message -match 'Chờ lấy hàng') ($jumpAhead.json.message)

$shipNoUnit = Invoke-Api POST $confirmUrl @{ toStatus = 'SHIPPING' } $adminToken
Check 'ban giao khong ghi don vi van chuyen -> 400' ($shipNoUnit.status -eq 400) ('status=' + $shipNoUnit.status + ' ' + $shipNoUnit.text)
Check '400 do bao thieu don vi van chuyen' ($shipNoUnit.json.message -match 'đơn vị vận chuyển') ($shipNoUnit.json.message)

$ship = Invoke-Api POST $confirmUrl @{ toStatus = 'SHIPPING'; shippingUnit = 'Giao Hàng Nhanh'; trackingCode = 'GHN-123' } $adminToken
Check 'CONFIRMED -> SHIPPING -> 200' ($ship.status -eq 200 -and $ship.json.status -eq 'SHIPPING') ('status=' + $ship.status + ' ' + $ship.text)
Check 'luu don vi van chuyen len don' ($ship.json.shippingUnit -eq 'Giao Hàng Nhanh') ($ship.json.shippingUnit)
Check 'luu ma van don len don' ($ship.json.trackingCode -eq 'GHN-123') ($ship.json.trackingCode)
Check 'lich su len 3 dong' (@($ship.json.history).Count -eq 3) (@($ship.json.history).Count)
$shipDbUnit = DbScalar ('select shipping_unit from orders where id = ' + $orderId)
Check 'shipping_unit xuong den cot DB that' ($shipDbUnit -eq 'Giao Hàng Nhanh') ('db=' + $shipDbUnit)

$cancelAfterShip = Invoke-Api POST $confirmUrl @{ toStatus = 'CANCELLED' } $adminToken
Check 'don dang van chuyen thi KHONG huy duoc (hang da ra khoi cua hang)' ($cancelAfterShip.status -eq 400) ('status=' + $cancelAfterShip.status)

$deliver = Invoke-Api POST $confirmUrl @{ toStatus = 'DELIVERED'; note = 'Khach nhan du' } $adminToken
Check 'SHIPPING -> DELIVERED -> 200' ($deliver.status -eq 200 -and $deliver.json.status -eq 'DELIVERED') ('status=' + $deliver.status + ' ' + $deliver.text)
Check 'DELIVERED khong con buoc nao de di tiep' (@($deliver.json.nextStatuses).Count -eq 0) ($deliver.json.nextStatuses -join ',')
$reopen = Invoke-Api POST $confirmUrl @{ toStatus = 'SHIPPING' } $adminToken
Check 'mo nguoc mot don da giao bi tu choi' ($reopen.status -eq 400) ('status=' + $reopen.status)

$stockAfterDeliver = Invoke-Api GET '/products/2' $null
Check 'giao xong khong dong them vao ton kho' ([int]$stockAfterDeliver.json.quantity -eq ($stockBefore - 3)) ($stockAfterDeliver.json.quantity)
$historyRows = DbScalar ('select count(*) from order_status_history where order_id = ' + $orderId)
Check 'DB ghi dung 4 dong lich su cho duong luong hoan chinh' ($historyRows -eq '4') ('db=' + $historyRows)
$stampSet = DbScalar ('select (status_updated_at is not null)::int from orders where id = ' + $orderId)
Check 'don co moc thoi gian trang thai gan nhat' ($stampSet -eq '1') ('db=' + $stampSet)

Write-Output '== danh sach, loc, dem =='

$listAll = Invoke-Api GET '/admin/orders?size=50' $null $adminToken
Check 'GET /admin/orders -> 200' ($listAll.status -eq 200) ('status=' + $listAll.status + ' ' + $listAll.text)
$listCodes = @($listAll.json.content | ForEach-Object { $_.orderCode })
Check 'don vua chay het luong co trong danh sach' ($listCodes -contains $orderCode) ($listCodes -join ',')
Check 'danh sach moi nhat len dau' ($listCodes[0] -eq $orderCode) ($listCodes -join ',')
Check 'moi don trong danh sach ke duoc buoc tiep theo' (@($listAll.json.content | Where-Object { $null -eq $_.nextStatuses }).Count -eq 0) ($listAll.text)

$pendingCodes = @( @(Invoke-Api GET '/admin/orders?status=PENDING&size=50' $null $adminToken).json.content | ForEach-Object { $_.orderCode } )
Check 'loc status=PENDING loai don da giao' (-not ($pendingCodes -contains $orderCode)) ($pendingCodes -join ',')
$deliveredCodes = @( @(Invoke-Api GET '/admin/orders?status=DELIVERED&size=50' $null $adminToken).json.content | ForEach-Object { $_.orderCode } )
Check 'loc status=DELIVERED giu don da giao' ($deliveredCodes -contains $orderCode) ($deliveredCodes -join ',')

$byPhone = @( @(Invoke-Api GET ('/admin/orders?search=' + $smokePhone) $null $adminToken).json.content | ForEach-Object { $_.orderCode } )
Check 'tim theo so dien thoai ra dung don' ($byPhone -contains $orderCode) ($byPhone -join ',')
$byCode = @( @(Invoke-Api GET ('/admin/orders?search=' + $orderCode) $null $adminToken).json.content | ForEach-Object { $_.orderCode } )
Check 'tim theo ma don ra dung don' ($byCode -contains $orderCode) ($byCode -join ',')
$noneFound = @(Invoke-Api GET '/admin/orders?search=khong-ton-tai-nao-dau' $null $adminToken).json.content
Check 'tim khong ra thi rong, khong phai loi' (@($noneFound).Count -eq 0) (@($noneFound).Count)

$badStatus = Invoke-Api GET '/admin/orders?status=FOO' $null $adminToken
Check 'status khong ton tai -> 400 kem thong bao' ($badStatus.status -eq 400 -and $badStatus.json.message -match 'không hợp lệ') ('status=' + $badStatus.status + ' ' + $badStatus.text)
$ghostOrder = Invoke-Api GET '/admin/orders/999999' $null $adminToken
Check 'don khong ton tai -> 400' ($ghostOrder.status -eq 400) ('status=' + $ghostOrder.status)

$summary = Invoke-Api GET '/admin/orders/summary' $null $adminToken
Check 'summary -> 200' ($summary.status -eq 200) ('status=' + $summary.status)
Check 'summary du 5 trang thai, khong thieu khoa nao' (@($summary.json.byStatus.PSObject.Properties.Name).Count -eq 5) ($summary.json.byStatus | ConvertTo-Json -Compress)
Check 'dem PENDING khong am' ([int]$summary.json.byStatus.PENDING -ge 0 -and [int]$summary.json.byStatus.DELIVERED -ge 1) ($summary.json.byStatus | ConvertTo-Json -Compress)
Check 'summary gui ca nhan tieng Viet' (@($summary.json.labels.PSObject.Properties.Name).Count -eq 5) ($summary.json.labels | ConvertTo-Json -Compress)

$p4 = Invoke-Api GET '/products/4' $null
$stock4Before = [int]$p4.json.quantity

$second = Invoke-Api POST '/orders' @{
    customerName = 'Khách Hủy'; phone = $cancelPhone; address = '08 Cancel Street'
    items = @(@{ productId = 4; quantity = 2 })
} $customerToken
Check 'tao don thu hai de do nhanh huy -> 201' ($second.status -eq 201) ('status=' + $second.status + ' ' + $second.text)
$secondId = $second.json.id
$secondUrl = '/admin/orders/' + $secondId + '/status'

$stock4Mid = Invoke-Api GET '/products/4' $null
Check 'don thu hai tru kho 2 cai' ([int]$stock4Mid.json.quantity -eq ($stock4Before - 2)) ($stock4Before.ToString() + ' -> ' + $stock4Mid.json.quantity)

$cancelBadField = Invoke-Api POST $secondUrl @{ toStatus = 'CANCELLED'; shippingUnit = 'GHN' } $adminToken
Check 'nop don vi van chuyen o buoc khong phai ban giao -> 400' ($cancelBadField.status -eq 400) ('status=' + $cancelBadField.status + ' ' + $cancelBadField.text)

$cancelled = Invoke-Api POST $secondUrl @{ toStatus = 'CANCELLED'; note = 'Khach doi y' } $adminToken
Check 'PENDING -> CANCELLED -> 200' ($cancelled.status -eq 200 -and $cancelled.json.status -eq 'CANCELLED') ('status=' + $cancelled.status + ' ' + $cancelled.text)
Check 'huy don tra lai dung 2 cai vao kho' ([int](Invoke-Api GET '/products/4' $null).json.quantity -eq $stock4Before) ('truoc=' + $stock4Before + ' sau=' + (Invoke-Api GET '/products/4' $null).json.quantity)

$cancelAgain = Invoke-Api POST $secondUrl @{ toStatus = 'CANCELLED' } $adminToken
Check 'huy lan thu hai bi tu choi' ($cancelAgain.status -eq 400) ('status=' + $cancelAgain.status)
Check 'huy lan thu hai khong cong tro kho lan nua' ([int](Invoke-Api GET '/products/4' $null).json.quantity -eq $stock4Before) ((Invoke-Api GET '/products/4' $null).json.quantity)
$cancelLine = @($cancelled.json.history)[-1]
Check 'dong lich su huy ghi tu PENDING sang CANCELLED' ($cancelLine.fromStatus -eq 'PENDING' -and $cancelLine.toStatus -eq 'CANCELLED') ($cancelLine | ConvertTo-Json -Compress)
Check 'don da huy khong con buoc nao' (@($cancelled.json.nextStatuses).Count -eq 0) ($cancelled.json.nextStatuses -join ',')

Write-Output '== khach thay gi =='

$mine = Invoke-Api GET '/orders/mine' $null $customerToken
Check '/orders/mine -> 200' ($mine.status -eq 200) ('status=' + $mine.status)
$mineFirst = @($mine.json | Where-Object { $_.orderCode -eq $orderCode })
Check 'khach don duoc don cua minh kem lich su' (@($mineFirst).Count -eq 1 -and @($mineFirst[0].history).Count -eq 4) (@($mineFirst).Count)
Check 'khach thay nhan tieng Viet o danh sach' ($mineFirst[0].statusLabel -eq 'Giao hàng thành công') ($mineFirst[0].statusLabel)
$mineCancelled = @($mine.json | Where-Object { $_.status -eq 'CANCELLED' })
Check 'don huy cung thay duoc 2 dong lich su' (@($mineCancelled).Count -eq 1 -and @($mineCancelled[0].history).Count -eq 2) (@($mineCancelled).Count)
$mineNoAdmin = @($mine.json | Where-Object { $_.customerName -ne 'Khách Smoke' -and $_.customerName -ne 'Khách Hủy' })
Check '/orders/mine khong ro don cua nguoi khac' (@($mineNoAdmin).Count -eq 0) (@($mineNoAdmin).Count)

Write-Output '== logout =='

$logout = Invoke-Api POST '/admin/logout' $null $adminToken
Check 'POST /admin/logout -> 204' ($logout.status -eq 204) ('status=' + $logout.status + ' ' + $logout.text)
$afterLogout = Invoke-Api GET '/admin/me' $null $adminToken
Check 'token da logout bi tu choi ngay, khong cho den het han' ($afterLogout.status -eq 401) ('status=' + $afterLogout.status)
$sessionLeft = DbScalar ("select count(*) from admin_session where token_hash = '" + (Sha256Hex $adminToken) + "'")
Check 'dong session xoa khoi DB khi logout' ($sessionLeft -eq '0') ('rows=' + $sessionLeft)

Write-Output '== don dep =='

DbExec ("delete from orders where phone in ('" + $smokePhone + "','" + $cancelPhone + "')")
DbExec ("delete from user_account where username in ('" + $customerUser + "')")
$adminId = DbScalar ("select id from admin where username = '" + $adminUser + "'")
DbExec ("delete from admin where id = " + $adminId)

foreach ($pair in ($stockSnap -split ',')) {
    if ($pair) {
        $kv = $pair -split '='
        DbExec ('update product set quantity = ' + $kv[1] + ' where id = ' + $kv[0])
    }
}

$stockRestored = DbScalar "select coalesce(string_agg(id || '=' || quantity, ',' order by id), '') from product where id in (2, 4)"
Check 'ton kho demo duoc tra lai dung nhu truoc khi chay' ($stockRestored -eq $stockSnap) ('truoc=' + $stockSnap + '  sau=' + $stockRestored)
$orderLeft = DbScalar ("select count(*) from orders where phone in ('" + $smokePhone + "','" + $cancelPhone + "')")
Check 'don probe da xoa het' ($orderLeft -eq '0') ('rows=' + $orderLeft)
$adminLeft = DbScalar ("select count(*) from admin where username = '" + $adminUser + "'")
Check 'admin probe da xoa (kèm session theo FK cascade)' ($adminLeft -eq '0') ('rows=' + $adminLeft)
$orphanHistory = DbScalar 'select count(*) from order_status_history h left join orders o on o.id = h.order_id where o.id is null'
Check 'khong con dong lich su ve don da bi xoa' ($orphanHistory -eq '0') ('rows=' + $orphanHistory)

Write-Output ('PASS ' + $script:passCount + '  FAIL ' + $script:failCount)
if ($script:failCount -gt 0) { exit 1 }



