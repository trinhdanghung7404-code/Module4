$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21.0.11'
Set-Location 'E:\DemoBuildShop\backend\backend'

# 8081 chu khong phai 8080: de nguyen instance IntelliJ dang chay cua nguoi dung,
# hai instance cung tro mot DB du cho viec smoke test nay.
# Cloudinary gia vi application.properties doc thang tu bien moi truong
# (CLOUDINARY_*), ma IntelliJ moi la noi dien vao. Smoke test khong tai anh nao.
$arguments = @(
    '--server.port=8081',
    '--cloudinary.cloud-name=smoke',
    '--cloudinary.api-key=000000000000000',
    '--cloudinary.api-secret=000000000000000000000000000'
) -join ' '

& .\mvnw.cmd -o spring-boot:run "-Dspring-boot.run.arguments=$arguments" *>&1 |
    Out-File -FilePath 'E:\DemoBuildShop\backend\backend\target\boot.log' -Encoding utf8
