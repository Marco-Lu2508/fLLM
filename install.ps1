<#
fLLM installer for Windows: downloads the release binary for this machine, verifies it against
the published SHA256 file and installs it to .\.local\bin\fllm-rs.exe (the path the wrapper
scripts use).

    .\install.ps1                     # newest release, the variant that fits this machine
    .\install.ps1 -Version v0.1.0     # a specific tag
    .\install.ps1 -Variant cuda       # force a variant: default | cuda | hip (auto picks cuda when an NVIDIA driver is present)
    .\install.ps1 -NoVerify           # allow a release without <asset>.sha256
    .\install.ps1 -Build              # skip the download, build this checkout with cargo (scripts\build-windows.ps1)

Release assets, tried in this order (the first one the release has is installed):
    x86_64:  fllm-windows-x86_64-msvc-<variant>.zip   (variant cuda or hip only)
             fllm-windows-x86_64-msvc.zip             (native MSVC build: CPU, Vulkan, DirectX 12)
             fllm-windows-x86_64.zip                  (MinGW cross build of older releases)
    ARM64:   fllm-windows-aarch64-msvc.zip, then the x86_64 assets (run in Windows' x64 emulation, which is slow)
The MSVC assets are linked with the static C runtime: no Visual C++ redistributable is needed. The CUDA variant needs only the
NVIDIA driver (the CUDA runtime is inside), the HIP variant the AMD HIP runtime (amdhip64.dll, part of the Adrenalin driver / HIP SDK).

The download URL is derived from `git remote -v` of this checkout, e.g.
https://github.com/Marco-Lu2508/fLLM/releases/download/<tag>/fllm-windows-x86_64-msvc.zip, and
falls back to that URL when git or the remote is missing. The expected hash always comes from
the release's .sha256 file - nothing is hardcoded here. A checksum that exists is always
checked; a mismatch aborts the installation.

The fallback build (also used when the release has no matching asset) compiles the checkout
next to this script with scripts\build-windows.ps1 and needs cargo (Rust toolchain) plus the
Visual Studio C++ build tools. PowerShell 5 or newer; no external tools are required
(Invoke-WebRequest / Get-FileHash / Expand-Archive).
#>
param(
    [string]$Version = "latest",
    [ValidateSet("auto", "default", "cuda", "hip")]
    [string]$Variant = "auto",
    [switch]$NoVerify,
    [switch]$Build
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$DefaultRepo = "https://github.com/Marco-Lu2508/fLLM"
$InstallDir = Join-Path $PSScriptRoot ".local\bin"

function Fail([string]$Message) {
    Write-Error "install.ps1: $Message" -ErrorAction Continue
    exit 1
}

# ---------------------------------------------------------------- platform
# a 32-bit PowerShell on 64-bit Windows reports x86 in PROCESSOR_ARCHITECTURE; the real one is in PROCESSOR_ARCHITEW6432
$NativeArch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $NativeArch = $env:PROCESSOR_ARCHITEW6432 }
$IsArm = $NativeArch -eq "ARM64"
if ($NativeArch -ne "AMD64" -and -not $IsArm) {
    Fail "unsupported processor architecture '$NativeArch' (fLLM needs 64-bit Windows on x86_64 or ARM64)"
}

# Which build fits this machine. NVIDIA: nvcuda.dll is part of the driver; AMD: the HIP runtime ships with the Adrenalin driver / HIP SDK.
function Get-Variant {
    if ($Variant -ne "auto") { return $Variant }
    if ($IsArm) { return "default" }
    $system = Join-Path $env:SystemRoot "System32"
    if (Test-Path (Join-Path $system "nvcuda.dll")) { return "cuda" }
    if (Get-ChildItem -Path $system -Filter "amdhip64*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1) { return "hip" }
    return "default"
}
$ChosenVariant = Get-Variant
if ($IsArm -and $ChosenVariant -ne "default") {
    Write-Host "install.ps1: no $ChosenVariant build for Windows ARM64; using the default build"
    $ChosenVariant = "default"
}

# Asset names are part of the contract with .github/workflows/release.yml and scripts\build-windows.ps1.
function Get-AssetCandidates {
    $candidates = @()
    if ($IsArm) { $candidates += "fllm-windows-aarch64-msvc.zip" }
    if (-not $IsArm -and $ChosenVariant -ne "default") { $candidates += "fllm-windows-x86_64-msvc-$ChosenVariant.zip" }
    $candidates += "fllm-windows-x86_64-msvc.zip"
    $candidates += "fllm-windows-x86_64.zip"
    return $candidates
}

# --------------------------------------------------------------- repository URL
$RepoUrl = $DefaultRepo
try {
    $line = (& git -C $PSScriptRoot remote -v 2>$null | Select-Object -First 1)
    if ($line) {
        $url = ($line -split "\s+")[1]
        if ($url -match '^(https?://.+?)(\.git)?$') { $RepoUrl = $Matches[1] }
        elseif ($url -match '^git@([^:]+):(.+?)(\.git)?$') {
            $RepoUrl = "https://$($Matches[1])/$($Matches[2])"
        }
    }
} catch { }
$RepoUrl = $RepoUrl.TrimEnd("/")
$ReleaseBase = "$RepoUrl/releases"

function Show-NextSteps([string]$Binary) {
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  $Binary devices                       # what fLLM sees on this machine (GPUs, memory, capabilities)"
    Write-Host "  $Binary run <huggingface link>        # download a model and chat (--device auto picks the best GPU, else the CPU)"
    $onPath = ($env:PATH -split ";") -contains $InstallDir
    if (-not $onPath) {
        Write-Host "To call it as 'fllm-rs' from anywhere, add the folder to PATH:"
        Write-Host "  [Environment]::SetEnvironmentVariable('PATH', [Environment]::GetEnvironmentVariable('PATH', 'User') + ';$InstallDir', 'User')"
    }
}

# --------------------------------------------------------------- build from source
function Build-FromSource {
    Write-Host "Building fLLM from source in $PSScriptRoot (needs cargo and the Visual Studio C++ build tools)"
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Fail "cargo not found - install a Rust toolchain (https://rustup.rs), or fix the release download reported above"
    }
    if (-not (Test-Path (Join-Path $PSScriptRoot "Cargo.toml"))) {
        Fail "$PSScriptRoot has no Cargo.toml; the source build needs a checkout of the repository"
    }
    $script = Join-Path $PSScriptRoot "scripts\build-windows.ps1"
    $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "target" }
    if ((Test-Path $script) -and -not $env:RUST_FEATURES) {
        # the script loads the Visual Studio environment, picks the features that the tools on this machine allow and builds for MSVC
        $buildArgs = @("-SkipPackage", "-SkipTests")
        if ($ChosenVariant -eq "cuda") { $buildArgs += "-Cuda" }
        if ($ChosenVariant -eq "hip") { $buildArgs += "-Hip" }
        if ($IsArm) { $buildArgs += "-Arm64" }
        & $script @buildArgs
        if ($LASTEXITCODE -ne 0) { Fail "scripts\build-windows.ps1 failed" }
        $triple = if ($IsArm) { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
        $built = Join-Path $targetDir "$triple\release\fllm-rs.exe"
    } else {
        Push-Location $PSScriptRoot
        try {
            $features = if ($env:RUST_FEATURES) { $env:RUST_FEATURES } else { "rust-gpu" }
            & cargo build --release --locked -p fllm-engine --features $features --bin fllm-rs
            if ($LASTEXITCODE -ne 0) { Fail "cargo build failed" }
        } finally { Pop-Location }
        $built = Join-Path $targetDir "release\fllm-rs.exe"
    }
    if (-not (Test-Path $built)) { Fail "build finished but $built is missing" }
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item $built (Join-Path $InstallDir "fllm-rs.exe") -Force
    Write-Host "Installed: $(Join-Path $InstallDir 'fllm-rs.exe')"
    Show-NextSteps (Join-Path $InstallDir 'fllm-rs.exe')
}

if ($Build) { Build-FromSource; exit 0 }

# --------------------------------------------------------------- resolve the tag
if ($Version -eq "latest") {
    $api = ($RepoUrl -replace '^https?://', '') -replace '^([^/]+)/(.+)$', 'https://$1/api/v1/repos/$2'
    try {
        $release = Invoke-RestMethod -Uri "$api/releases/latest" -TimeoutSec 30
    } catch {
        Fail "cannot query $api/releases/latest - is $RepoUrl reachable? (use -Build to compile instead)"
    }
    $Version = $release.tag_name
    if (-not $Version) { Fail "no release found via $api/releases/latest" }
    Write-Host "Newest release: $Version"
}

# --------------------------------------------------------------- download + verify
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("fllm-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
    $Asset = $null
    $Zip = $null
    foreach ($candidate in Get-AssetCandidates) {
        $url = "$ReleaseBase/download/$Version/$candidate"
        $target = Join-Path $Tmp $candidate
        Write-Host "Fetching $url"
        try {
            Invoke-WebRequest -Uri $url -OutFile $target -TimeoutSec 600 | Out-Null
            $Asset = $candidate
            $Zip = $target
            break
        } catch {
            Write-Host "install.ps1: no release asset '$candidate' for tag '$Version'"
        }
    }
    if (-not $Asset) {
        Write-Host "install.ps1: the release '$Version' has no Windows asset for this machine"
        Build-FromSource
        exit 0
    }
    if ($IsArm -and $Asset -notlike "*aarch64*") {
        Write-Host "install.ps1: WARNING - installing the x86_64 build on ARM64; it runs in Windows' x64 emulation, which is slow"
    }
    Write-Host "Variant: $ChosenVariant, asset: $Asset"
    $AssetUrl = "$ReleaseBase/download/$Version/$Asset"
    $Sum = "$Zip.sha256"

    $haveSum = $true
    try { Invoke-WebRequest -Uri "$AssetUrl.sha256" -OutFile $Sum -TimeoutSec 60 | Out-Null }
    catch { $haveSum = $false }

    if ($haveSum) {
        $expected = ((Get-Content $Sum -Raw) -split '\s+')[0].ToLowerInvariant()
        if (-not $expected) { Fail "$Asset.sha256 is empty; refusing to install" }
        $actual = (Get-FileHash -Algorithm SHA256 -Path $Zip).Hash.ToLowerInvariant()
        if ($expected -ne $actual) {
            Write-Host "install.ps1: CHECKSUM MISMATCH for $Asset"
            Write-Host "  expected: $expected"
            Write-Host "  actual:   $actual"
            Fail "the download was NOT installed; the asset and its .sha256 file do not belong together"
        }
        Write-Host "SHA256 OK: $actual"
    } elseif ($NoVerify) {
        Write-Host "install.ps1: WARNING - -NoVerify: installing WITHOUT checksum verification"
    } else {
        Fail "the release for '$Version' publishes no $Asset.sha256, so the download cannot be verified. Re-run with -NoVerify to install it anyway"
    }

    Expand-Archive -Path $Zip -DestinationPath $Tmp -Force
    $Binary = Join-Path $Tmp "fllm-rs.exe"
    if (-not (Test-Path $Binary)) { Fail "$Asset does not contain fllm-rs.exe" }
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $Installed = Join-Path $InstallDir "fllm-rs.exe"
    Copy-Item $Binary $Installed -Force
    Write-Host "Installed ${Version}: $Installed"

    # does it start? (a missing DLL or an unsupported CPU shows up here and not at the first model download)
    & $Installed --help *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "install.ps1: WARNING - the installed binary did not start (exit code $LASTEXITCODE)."
        Write-Host "  The 'gnu' asset of older releases needs no extra files; the MSVC assets need none either (static C runtime)."
        Write-Host "  If it still fails, use WSL2 ('wsl ./install.ps1') or Docker Desktop ('docker compose up -d')."
    } else {
        Show-NextSteps $Installed
    }
} finally {
    if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force -ErrorAction SilentlyContinue }
}
