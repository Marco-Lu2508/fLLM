<#
Builds fllm-rs natively for Windows with the MSVC toolchain and packs the release asset that install.ps1 downloads.

    .\scripts\build-windows.ps1                 # CPU + Vulkan/DirectX 12 (wgpu) + native Vulkan: dist\fllm-windows-x86_64-msvc.zip
    .\scripts\build-windows.ps1 -Cuda           # plus the CUDA kernels:      dist\fllm-windows-x86_64-msvc-cuda.zip
    .\scripts\build-windows.ps1 -Hip            # plus the AMD HIP kernels:   dist\fllm-windows-x86_64-msvc-hip.zip
    .\scripts\build-windows.ps1 -Arm64          # aarch64-pc-windows-msvc (run on a Windows ARM64 machine)
    .\scripts\build-windows.ps1 -DryRun         # print the plan (tools found, features, command) and stop

What it needs
  * Rust (rustup) with the MSVC host toolchain, and the "Desktop development with C++" workload of Visual Studio 2022 / Build Tools
    (cl.exe, link.exe, lib.exe). From a normal PowerShell the script loads the Visual Studio developer environment itself.
  * -Cuda: the CUDA Toolkit (CUDA_PATH is set by its installer; nvcc needs the MSVC host compiler, which the step above provides).
    FLLM_CUDA_ARCHS selects the GPU architectures (-CudaArchs "89;120" for a quick build); a Visual Studio newer than the toolkit
    supports needs FLLM_CUDA_EXTRA_FLAGS="-allow-unsupported-compiler".
  * -Hip: a ROCm / HIP SDK for Windows (HIP_PATH, or the newest Program Files\AMD\ROCm\X.Y) with hipcc or lib\llvm\bin\clang++, and
    lib\amdhip64.lib. -HipArchs "gfx1100;gfx1151" selects the GPU architectures (default: RDNA2..RDNA4 and the RDNA3.5 APUs; the
    ones the SDK has no device libraries for are skipped). The package gets the runtime DLLs of the SDK (amdhip64_N.dll, amd_comgr,
    rocm_kpack, when present) next to fllm-rs.exe unless -NoHipRuntime is given; the binary finds a runtime in the driver
    (System32) or the SDK on its own, so a package without them also works where one is installed.
  * the native Vulkan backend: the Vulkan SDK (glslc, VULKAN_SDK); without it the script drops the "vulkan" feature and says so.

The binary is linked with the static C runtime (-C target-feature=+crt-static) so that no Visual C++ redistributable is needed;
-DynamicCrt links the shared one (smaller, but needs vcruntime140.dll on the target machine). The zip holds fllm-rs.exe and LICENSE, next to
it <asset>.sha256 ("<hash>  <asset>", the format install.ps1 verifies).

Nothing here has run on a Windows machine with these toolchains yet (status L0: the PowerShell parses, -DryRun runs).
#>
param(
    [switch]$Cuda,
    [switch]$Hip,
    [switch]$NoHipRuntime,
    [switch]$Arm64,
    [switch]$DynamicCrt,
    [switch]$NoVulkan,
    [switch]$SkipTests,
    [switch]$SkipPackage,
    [switch]$DryRun,
    [string]$Features = "",
    [string]$CudaArchs = "",
    [string]$HipArchs = "",
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Triple = if ($Arm64) { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
$ArchName = if ($Arm64) { "aarch64" } else { "x86_64" }

function Fail([string]$Message) {
    Write-Error "build-windows.ps1: $Message" -ErrorAction Continue
    exit 1
}

function Find-Tool([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

# ---------------------------------------------------------------- Visual Studio developer environment
function Enter-DeveloperEnvironment {
    if (Find-Tool "cl.exe") { return "cl.exe already on PATH" }
    $programFiles = ${env:ProgramFiles(x86)}
    if (-not $programFiles) { $programFiles = $env:ProgramFiles }
    $vswhere = Join-Path $programFiles "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Fail "Visual Studio (C++ build tools) not found: install 'Build Tools for Visual Studio' with the 'Desktop development with C++' workload, or run this from a Developer PowerShell"
    }
    $installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    if (-not $installation) { Fail "vswhere found no Visual Studio with the C++ x64 build tools" }
    $arch = if ($Arm64) { "arm64" } else { "x64" }
    $hostArch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
    $devShell = Join-Path $installation "Common7\Tools\Launch-VsDevShell.ps1"
    if (Test-Path $devShell) {
        # the official way: the script changes the environment of this process
        # Launch-VsDevShell.ps1 names the 64-bit x86 architecture amd64 (ValidateSet x86, amd64, arm, arm64); VsDevCmd.bat below takes x64
        $shellArch = if ($arch -eq "x64") { "amd64" } else { $arch }
        $shellHost = if ($hostArch -eq "x64") { "amd64" } else { $hostArch }
        & $devShell -Arch $shellArch -HostArch $shellHost -SkipAutomaticLocation | Out-Null
        if (Find-Tool "cl.exe") { return "loaded $devShell" }
    }
    $vsDevCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) { Fail "neither Launch-VsDevShell.ps1 nor VsDevCmd.bat found in $installation" }
    # run VsDevCmd in cmd.exe (through a batch file: no quoting trouble with spaces) and take over the environment it leaves behind
    $batch = Join-Path ([System.IO.Path]::GetTempPath()) ("fllm-vsenv-" + [System.Guid]::NewGuid().ToString("N") + ".cmd")
    Set-Content -Path $batch -Encoding ASCII -Value @("@echo off", "call `"$vsDevCmd`" -no_logo -arch=$arch -host_arch=$hostArch >nul", "set")
    try {
        $lines = & $batch
    } finally {
        Remove-Item $batch -Force -ErrorAction SilentlyContinue
    }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
    if (-not (Find-Tool "cl.exe")) { Fail "cl.exe is still not on PATH after loading $vsDevCmd" }
    return "loaded $vsDevCmd"
}

# ---------------------------------------------------------------- features
$base = @("rust-gpu")
$vulkanSdkNote = ""
if (-not $NoVulkan) {
    $glslc = Find-Tool "glslc.exe"
    if (-not $glslc -and $env:VULKAN_SDK) {
        $candidate = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
        if (Test-Path $candidate) { $glslc = $candidate }
    }
    if ($glslc) { $base += "vulkan" } else { $vulkanSdkNote = "glslc not found (Vulkan SDK): building without the native Vulkan backend" }
}
if ($Cuda) { $base += "cuda" }
if ($Hip) { $base += "hip" }
if ($Cuda -and $Hip) { Fail "-Cuda and -Hip build the same kernel library with different compilers; choose one" }
$featureList = if ($Features) { $Features } else { $base -join "," }

# ---------------------------------------------------------------- asset name
$suffix = ""
if ($Cuda) { $suffix = "-cuda" }
if ($Hip) { $suffix = "-hip" }
$Asset = "fllm-windows-$ArchName-msvc$suffix.zip"

$crt = if ($DynamicCrt) { "dynamic C runtime (/MD)" } else { "static C runtime (/MT, no redistributable needed)" }
Write-Host "target:    $Triple"
Write-Host "features:  $featureList"
Write-Host "runtime:   $crt"
Write-Host "asset:     $(Join-Path $OutDir $Asset)"
if ($vulkanSdkNote) { Write-Host "note:      $vulkanSdkNote" }
if ($Cuda) {
    $cudaHome = if ($env:CUDA_HOME) { $env:CUDA_HOME } elseif ($env:CUDA_PATH) { $env:CUDA_PATH } else { "(build.rs picks the newest toolkit under Program Files)" }
    Write-Host "cuda:      $cudaHome   archs: $(if ($CudaArchs) { $CudaArchs } elseif ($env:FLLM_CUDA_ARCHS) { $env:FLLM_CUDA_ARCHS } else { 'default' })"
}
if ($Hip) {
    $hipHome = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "(build.rs picks the newest Program Files\AMD\ROCm)" }
    Write-Host "hip:       $hipHome   archs: $(if ($HipArchs) { $HipArchs } elseif ($env:FLLM_HIP_ARCHS) { $env:FLLM_HIP_ARCHS } else { 'default' })"
}

if ($DryRun) {
    Write-Host ""
    Write-Host "dry run: would load the Visual Studio environment, run"
    Write-Host "  cargo build --release --locked -p fllm-engine --features `"$featureList`" --bin fllm-rs --target $Triple"
    Write-Host "and pack $Asset with its .sha256 into $OutDir"
    exit 0
}

# ---------------------------------------------------------------- build
Write-Host ""
Write-Host "Visual Studio: $(Enter-DeveloperEnvironment)"
if (-not (Find-Tool "cargo.exe")) { Fail "cargo not found - install Rust from https://rustup.rs" }
& rustup target add $Triple
if ($LASTEXITCODE -ne 0) { Fail "rustup target add $Triple failed" }

if ($CudaArchs) { $env:FLLM_CUDA_ARCHS = $CudaArchs }
if ($HipArchs) { $env:FLLM_HIP_ARCHS = $HipArchs }
$HipRoot = $null
if ($Hip) {
    $env:FLLM_HIP = "1"
    # the SDK root, the way build.rs finds it: ROCM_PATH, HIP_PATH, else the newest Program Files\AMD\ROCm\X.Y
    $HipRoot = if ($env:ROCM_PATH) { $env:ROCM_PATH } elseif ($env:HIP_PATH) { $env:HIP_PATH } else {
        $rocmDir = Join-Path $env:ProgramFiles "AMD\ROCm"
        $newest = if (Test-Path $rocmDir) { Get-ChildItem $rocmDir -Directory | Sort-Object { try { [version]$_.Name } catch { [version]"0.0" } } -Descending | Select-Object -First 1 } else { $null }
        if ($newest) { $newest.FullName } else { $null }
    }
    if (-not $HipRoot -or -not (Test-Path $HipRoot)) { Fail "-Hip: no HIP SDK found; install the AMD HIP SDK for Windows or set HIP_PATH" }
    $hipcc = Join-Path $HipRoot "bin\hipcc.exe"
    $hipClang = Join-Path $HipRoot "lib\llvm\bin\clang++.exe"
    $hipLib = Join-Path $HipRoot "lib\amdhip64.lib"
    Write-Host "hip root:  $HipRoot"
    Write-Host "compiler:  $(if (Test-Path $hipcc) { $hipcc } elseif (Test-Path $hipClang) { "$hipClang (no hipcc.exe in the SDK)" } else { 'NOT FOUND' })"
    if (-not (Test-Path $hipcc) -and -not (Test-Path $hipClang)) { Fail "-Hip: neither bin\hipcc.exe nor lib\llvm\bin\clang++.exe in $HipRoot" }
    if (-not (Test-Path $hipLib)) { Fail "-Hip: $hipLib (import library of the HIP runtime) is missing" }
    # build.rs reads ROCM_PATH / HIP_PATH; make sure it sees the root found here
    if (-not $env:ROCM_PATH -and -not $env:HIP_PATH) { $env:HIP_PATH = $HipRoot }
}
if (-not $DynamicCrt) {
    $flag = "-C target-feature=+crt-static"
    if ($env:RUSTFLAGS -and $env:RUSTFLAGS -notmatch "crt-static") { $env:RUSTFLAGS = "$($env:RUSTFLAGS) $flag" }
    elseif (-not $env:RUSTFLAGS) { $env:RUSTFLAGS = $flag }
}

Push-Location $Root
try {
    & cargo build --release --locked -p fllm-engine --features $featureList --bin fllm-rs --target $Triple
    if ($LASTEXITCODE -ne 0) { Fail "cargo build failed" }
    if (-not $SkipTests -and -not $Arm64 -and $env:PROCESSOR_ARCHITECTURE -ne "ARM64") {
        & cargo test --release --locked -p fllm-engine --lib --target $Triple
        if ($LASTEXITCODE -ne 0) { Fail "cargo test failed" }
    }
} finally { Pop-Location }

$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root "target" }
$Binary = Join-Path $targetDir "$Triple\release\fllm-rs.exe"
if (-not (Test-Path $Binary)) { Fail "build finished but $Binary is missing" }

# smoke test: the binary starts and lists its devices (the CPU at least)
if ($Hip) {
    # show how the HIP runtime is imported: it must be a delay-load import, or the exe does not start without the DLL
    $dumpbin = Find-Tool "dumpbin.exe"
    if ($dumpbin) {
        Write-Host "---- dumpbin /dependents"
        & $dumpbin /nologo /dependents $Binary
        Write-Host "---- dumpbin /imports (delay-load section)"
        & $dumpbin /nologo /imports $Binary | Select-String -Pattern "amdhip64|delay" -Context 0,1
        Write-Host "----"
    }
}
& $Binary --help *> $null
& $Binary devices
if ($LASTEXITCODE -ne 0) { Fail "$Binary devices failed" }

if ($SkipPackage) { Write-Host "Built: $Binary"; exit 0 }

# ---------------------------------------------------------------- package
$dist = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $Root $OutDir }
New-Item -ItemType Directory -Force -Path $dist | Out-Null
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("fllm-pack-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
    Copy-Item $Binary (Join-Path $stage "fllm-rs.exe")
    # the test script (devices, self tests, speed per device -> fllm-report.txt) travels with the binary
    if (Test-Path (Join-Path $Root "scripts\windows-check.ps1")) { Copy-Item (Join-Path $Root "scripts\windows-check.ps1") (Join-Path $stage "windows-check.ps1") }
    Copy-Item (Join-Path $Root "LICENSE") (Join-Path $stage "LICENSE")
    if (Test-Path (Join-Path $Root "NOTICE")) { Copy-Item (Join-Path $Root "NOTICE") (Join-Path $stage "NOTICE") }
    if (Test-Path (Join-Path $Root "THIRD_PARTY.md")) { Copy-Item (Join-Path $Root "THIRD_PARTY.md") (Join-Path $stage "THIRD_PARTY.md") }
    if (Test-Path (Join-Path $Root "licenses")) { Copy-Item -Recurse (Join-Path $Root "licenses") (Join-Path $stage "licenses") }
    if ($Hip) {
        # What may be redistributed: the HIP runtime DLLs of the SDK (open-source ROCm components, MIT / Apache-2.0 with LLVM exception;
        # their license texts go to licenses\rocm). Not bundled: the graphics driver, the Visual C++ runtime (the exe uses the static one),
        # and the SDK's closed or EULA-bound parts. The user needs an AMD driver that supports the GPU; the binary uses the DLL next to
        # it first, then the one in System32 (driver), then HIP_PATH / Program Files\AMD\ROCm.
        $bundled = @()
        if (-not $NoHipRuntime) {
            foreach ($pattern in @("amdhip64*.dll", "amd_comgr*.dll", "rocm_kpack*.dll")) {
                foreach ($f in Get-ChildItem -Path (Join-Path $HipRoot "bin") -Filter $pattern -File -ErrorAction SilentlyContinue) {
                    Copy-Item $f.FullName (Join-Path $stage $f.Name)
                    $bundled += $f.Name
                }
            }
            $licDir = Join-Path $stage "licenses\rocm"
            New-Item -ItemType Directory -Force -Path $licDir | Out-Null
            foreach ($f in Get-ChildItem -Path $HipRoot -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(LICENSE|NOTICE|COPYING)' }) { Copy-Item $f.FullName $licDir }
            foreach ($f in Get-ChildItem -Path (Join-Path $HipRoot "share\doc") -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(LICENSE|NOTICE|COPYING)' } | Select-Object -First 40) {
                Copy-Item $f.FullName (Join-Path $licDir ("{0}-{1}" -f $f.Directory.Name, $f.Name)) -ErrorAction SilentlyContinue
            }
        }
        Write-Host "bundled HIP runtime files: $(if ($bundled) { $bundled -join ', ' } else { '(none)' })"
        $readme = @(
            "fLLM with AMD HIP (ROCm) kernels for Windows",
            "",
            "Needs an AMD GPU that the Windows HIP runtime supports (Radeon RX 7000 / 9000, some RX 6000, Ryzen AI Max / Strix Halo)",
            "and a current AMD Radeon (Adrenalin) driver. Without a usable GPU or runtime fllm-rs.exe still starts and uses Vulkan or the CPU.",
            "",
            "The HIP runtime DLL (amdhip64_N.dll) is looked up in this order: FLLM_HIP_DLL, this folder, System32 (the driver),",
            "HIP_PATH\bin, ROCM_PATH\bin, Program Files\AMD\ROCm\<newest>\bin. Delete the bundled amdhip64*.dll to use the driver's copy.",
            "",
            "Bundled runtime files: $(if ($bundled) { $bundled -join ', ' } else { 'none' }). They come from the AMD ROCm / HIP SDK and keep their own licenses (licenses\rocm).",
            "Check with: fllm-rs devices"
        )
        Set-Content -Path (Join-Path $stage "README-HIP.txt") -Value $readme -Encoding ASCII
    }
    $zip = Join-Path $dist $Asset
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
    $hash = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    # "<hash>  <asset>": two spaces, LF, no byte order mark
    [System.IO.File]::WriteAllText("$zip.sha256", "$hash  $Asset`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Packed: $zip"
    Write-Host "SHA256: $hash"
} finally {
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
}
