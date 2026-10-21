$ErrorActionPreference = 'Stop'
$burhanRoot = Split-Path $PSScriptRoot -Parent
$burhanTools = Join-Path $burhanRoot '.tools'
New-Item -ItemType Directory -Force "$burhanTools\downloads", "$burhanTools\sdk" | Out-Null
$burhanJava = Get-ChildItem "$burhanTools\jdk" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $burhanJava) {
    Invoke-WebRequest 'https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal/eclipse' -OutFile "$burhanTools\downloads\jdk21.zip"
    Expand-Archive -LiteralPath "$burhanTools\downloads\jdk21.zip" -DestinationPath "$burhanTools\jdk" -Force
    $burhanJava = Get-ChildItem "$burhanTools\jdk" -Directory | Select-Object -First 1
}
$env:JAVA_HOME = $burhanJava.FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$sdkManager = "$env:LOCALAPPDATA\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat"
if (-not (Test-Path -LiteralPath $sdkManager)) {
    Invoke-WebRequest 'https://dl.google.com/android/repository/commandlinetools-win-13114758_latest.zip' -OutFile "$burhanTools\downloads\android-tools.zip"
    Expand-Archive -LiteralPath "$burhanTools\downloads\android-tools.zip" -DestinationPath "$burhanTools\android-cli" -Force
    $sdkManager = "$burhanTools\android-cli\cmdline-tools\bin\sdkmanager.bat"
}
Write-Host 'Review and accept Android SDK licences to build Burhan.'
& $sdkManager --sdk_root="$burhanTools\sdk" --licenses
if ($LASTEXITCODE -ne 0) { throw 'SDK licence setup failed' }
& $sdkManager --sdk_root="$burhanTools\sdk" 'platforms;android-36' 'build-tools;36.0.0' 'platform-tools'
if ($LASTEXITCODE -ne 0) { throw 'SDK installation failed' }
Write-Host 'Burhan private Android toolchain is ready.'
