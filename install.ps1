<#
.SYNOPSIS
fLLM installer for Windows: downloads the release binary for this machine, verifies its checksum and installs it.

.DESCRIPTION
Works from a clone and without one (Windows PowerShell 5.1 and PowerShell 7):

    irm https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.ps1 | iex

    # with parameters
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.ps1))) -Variant cuda -AddToPath

    .\install.ps1 -Version v0.1.2          # a specific tag (default: the newest release)
    .\install.ps1 -Variant cuda            # auto (default) | default (alias cpu) | cuda | hip
    .\install.ps1 -AddToPath               # add the install folder to the USER PATH without asking
    .\install.ps1 -NoAddToPath             # never touch PATH and do not ask
    .\install.ps1 -Uninstall               # remove fllm-rs.exe and the PATH entry this script added
    .\install.ps1 -InstallDir D:\tools\fllm
    .\install.ps1 -NoVerify                # allow a release without a checksum
    .\install.ps1 -Build                   # build this checkout with cargo (scripts\build-windows.ps1) instead of downloading

The binary goes to %LOCALAPPDATA%\fllm\bin\fllm-rs.exe (FLLM_INSTALL_DIR or -InstallDir change that). The USER PATH (never the machine
PATH, no administrator rights) is changed only with -AddToPath, or after you answer the prompt when the host is interactive.

Release assets, tried in this order (the first one the release has is installed):
    x86_64:  fllm-windows-x86_64-msvc-<variant>.zip   (variant cuda or hip only)
             fllm-windows-x86_64-msvc.zip             (native MSVC build: CPU, Vulkan, DirectX 12)
             fllm-windows-x86_64.zip                  (MinGW cross build of older releases)
    ARM64:   fllm-windows-aarch64-msvc.zip, then the x86_64 assets (run in Windows' x64 emulation, which is slow)
The MSVC assets are linked with the static C runtime: no Visual C++ redistributable is needed. The CUDA variant needs only the
NVIDIA driver (the CUDA runtime is inside), the HIP variant the AMD HIP runtime (amdhip64.dll, part of the Adrenalin driver / HIP SDK).
With -Variant auto an NVIDIA GPU (nvcuda.dll, nvidia-smi or the display adapter list in the registry) selects cuda when the release
has that asset; otherwise the default build is installed and the script says so.

What the checksum check protects against: a damaged or truncated download, a proxy that serves the wrong file and an asset that does not
belong to its .sha256 file. The expected hash comes from the release's <asset>.sha256 file and, when the release notes carry a
"<sha256>  <asset>" list, from there too; both must agree. It is NOT a signature: whoever can replace the asset in the release can replace
the .sha256 file and the notes as well. HTTPS to github.com authenticates the source.

The binaries are NOT code-signed. Windows SmartScreen or Defender may warn about an unsigned program that was downloaded from the internet;
a file fetched by this script does not carry the "downloaded from the internet" mark (and the script removes it if present), a zip saved by a
browser does (use Unblock-File on the zip, or "Properties > Unblock"). The script never changes Defender or SmartScreen settings.

Environment overrides: FLLM_INSTALL_DIR, FLLM_REPO_URL (repository URL instead of the default: a mirror, a fork, a test server),
FLLM_FORGE (github | gitea), FLLM_API_URL (GitHub API base, default https://api.github.com), RUST_FEATURES (-Build).
"latest" on github.com follows the /releases/latest redirect; while the repository has only pre-releases (v0.x) that redirect has no
target and the newest release is read from api.github.com instead. Other hosts use the Gitea API (/api/v1).

The -Build path compiles the checkout next to this script with scripts\build-windows.ps1 and needs cargo (Rust toolchain) plus the Visual
Studio C++ build tools; it only works from a clone.
Test hook: FLLM_TEST_USERPATH names a file that stands in for the user PATH (used by the tests on non-Windows hosts).
#>
param(
    [string]$Version = "latest",
    [ValidateSet("auto", "default", "cpu", "cuda", "hip")]
    [string]$Variant = "auto",
    [string]$InstallDir = "",
    [switch]$AddToPath,
    [switch]$NoAddToPath,
    [switch]$Uninstall,
    [switch]$NoVerify,
    [switch]$Build
)

function Install-Fllm {
    [CmdletBinding()]
    param(
        [string]$Version = "latest",
        [string]$Variant = "auto",
        [string]$InstallDir = "",
        [switch]$AddToPath,
        [switch]$NoAddToPath,
        [switch]$Uninstall,
        [switch]$NoVerify,
        [switch]$Build
    )

    # Set in this function only: nothing leaks into the session that ran `irm ... | iex`.
    $ErrorActionPreference = "Stop"
    Set-StrictMode -Version Latest
    $ProgressPreference = "SilentlyContinue"   # Windows PowerShell 5.1 downloads are very slow with the progress bar
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch { }

    $DefaultRepo = "https://github.com/Marco-Lu2508/fLLM"
    $Seps = [char[]]@([char]92, [char]47)   # backslash and slash
    $Here = $PSScriptRoot   # empty when the script comes through `iex` or a script block
    if ($Variant -eq "cpu") { $Variant = "default" }
    if ($Version -match '^[0-9]') { $Version = "v$Version" }

    # ---------------------------------------------------------------- install folder
    if (-not $InstallDir) { $InstallDir = $env:FLLM_INSTALL_DIR }
    if (-not $InstallDir) {
        $base = $env:LOCALAPPDATA
        if (-not $base) { throw "LOCALAPPDATA is not set; give a folder with -InstallDir" }
        $InstallDir = Join-Path (Join-Path $base "fllm") "bin"
    }
    $InstallDir = $InstallDir.TrimEnd($Seps)
    $Exe = Join-Path $InstallDir "fllm-rs.exe"

    # ---------------------------------------------------------------- user PATH
    function Get-UserPath {
        if ($env:FLLM_TEST_USERPATH) {
            if (Test-Path $env:FLLM_TEST_USERPATH) {
                $raw = Get-Content $env:FLLM_TEST_USERPATH -Raw
                if ($null -ne $raw) { return $raw.Trim() }
            }
            return ""
        }
        $value = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($null -eq $value) { return "" }
        return $value
    }
    function Set-UserPath([string]$Value) {
        if ($env:FLLM_TEST_USERPATH) { Set-Content -Path $env:FLLM_TEST_USERPATH -Value $Value -NoNewline; return }
        [Environment]::SetEnvironmentVariable("PATH", $Value, "User")
    }
    function Split-PathList([string]$List) {
        return @($List -split ";" | Where-Object { $_ -ne "" })
    }
    function Test-InList([string[]]$Entries, [string]$Dir) {
        $want = $Dir.TrimEnd($Seps)
        foreach ($entry in $Entries) {
            if ($entry.TrimEnd($Seps) -ieq $want) { return $true }
        }
        return $false
    }
    function Test-Interactive {
        try { return ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and $Host.Name -ne "Default Host") }
        catch { return $false }
    }

    if ($Uninstall) {
        if (Test-Path $Exe) { Remove-Item $Exe -Force; Write-Host "Removed $Exe" } else { Write-Host "No $Exe to remove." }
        if ((Test-Path $InstallDir) -and -not (Get-ChildItem -Path $InstallDir -Force | Select-Object -First 1)) {
            Remove-Item $InstallDir -Force -ErrorAction SilentlyContinue
        }
        $entries = Split-PathList (Get-UserPath)
        if (Test-InList $entries $InstallDir) {
            $kept = @($entries | Where-Object { $_.TrimEnd($Seps) -ine $InstallDir.TrimEnd($Seps) })
            Set-UserPath ($kept -join ";")
            Write-Host "Removed $InstallDir from the user PATH (open a new terminal for it to take effect)."
        }
        Write-Host "Models and caches are not touched (see docs/getting-started.md for their location)."
        return
    }

    # ---------------------------------------------------------------- platform
    # a 32-bit PowerShell on 64-bit Windows reports x86 in PROCESSOR_ARCHITECTURE; the real one is in PROCESSOR_ARCHITEW6432
    $NativeArch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $NativeArch = $env:PROCESSOR_ARCHITEW6432 }
    $IsArm = ($NativeArch -eq "ARM64")
    if ($NativeArch -ne "AMD64" -and -not $IsArm) {
        throw "unsupported processor architecture '$NativeArch' (fLLM needs 64-bit Windows on x86_64 or ARM64)"
    }

    function Test-NvidiaGpu {
        if ($env:SystemRoot -and (Test-Path (Join-Path (Join-Path $env:SystemRoot "System32") "nvcuda.dll"))) { return $true }
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
        try {
            # display adapters: HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-...}\0000, 0001, ... with DriverDesc
            $class = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            if (Test-Path $class) {
                foreach ($key in Get-ChildItem -Path $class -ErrorAction SilentlyContinue) {
                    $desc = (Get-ItemProperty -Path $key.PSPath -Name DriverDesc -ErrorAction SilentlyContinue)
                    if ($desc -and $desc.DriverDesc -match "NVIDIA") { return $true }
                }
            }
        } catch { }
        return $false
    }
    function Test-AmdHip {
        if (-not $env:SystemRoot) { return $false }
        $system = Join-Path $env:SystemRoot "System32"
        if (-not (Test-Path $system)) { return $false }
        return [bool](Get-ChildItem -Path $system -Filter "amdhip64*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1)
    }

    $Chosen = $Variant
    $GpuNote = ""
    if ($Variant -eq "auto") {
        $Chosen = "default"
        if ($IsArm) { $Chosen = "default" }
        elseif (Test-NvidiaGpu) { $Chosen = "cuda"; $GpuNote = "NVIDIA GPU detected" }
        elseif (Test-AmdHip) { $Chosen = "hip"; $GpuNote = "AMD HIP runtime detected" }
    }
    if ($IsArm -and $Chosen -ne "default") {
        Write-Host "install.ps1: no $Chosen build for Windows ARM64; using the default build"
        $Chosen = "default"
    }

    # Asset names are part of the contract with the release workflow (.github/workflows/release.yml) and scripts\build-windows.ps1.
    function Get-AssetCandidates {
        $candidates = @()
        if ($IsArm) { $candidates += "fllm-windows-aarch64-msvc.zip" }
        if (-not $IsArm -and $Chosen -ne "default") { $candidates += "fllm-windows-x86_64-msvc-$Chosen.zip" }
        $candidates += "fllm-windows-x86_64-msvc.zip"
        $candidates += "fllm-windows-x86_64.zip"
        return $candidates
    }

    # ---------------------------------------------------------------- repository URL
    $RepoUrl = $DefaultRepo
    if ($env:FLLM_REPO_URL) {
        $RepoUrl = $env:FLLM_REPO_URL
    } elseif ($Here -and (Test-Path (Join-Path $Here ".git"))) {
        try {
            $line = (& git -C $Here remote -v 2>$null | Select-Object -First 1)
            if ($line) {
                $url = ($line -split "\s+")[1]
                if ($url -match '^(https?://.+?)(\.git)?$') { $RepoUrl = $Matches[1] }
                elseif ($url -match '^git@([^:]+):(.+?)(\.git)?$') { $RepoUrl = "https://$($Matches[1])/$($Matches[2])" }
            }
        } catch { }
    }
    $RepoUrl = $RepoUrl.TrimEnd("/")
    $ReleaseBase = "$RepoUrl/releases"

    $RepoHost = ($RepoUrl -replace '^https?://', '') -replace '/.*$', ''
    $Forge = $env:FLLM_FORGE
    if (-not $Forge) { $Forge = if ($RepoHost -in @("github.com", "www.github.com")) { "github" } else { "gitea" } }
    if ($Forge -notin @("github", "gitea")) { throw "FLLM_FORGE must be 'github' or 'gitea', not '$Forge'" }
    $Slug = ($RepoUrl -replace '^https?://[^/]+/', '')
    if ($Forge -eq "github") {
        $apiRoot = if ($env:FLLM_API_URL) { $env:FLLM_API_URL.TrimEnd("/") } else { "https://api.github.com" }
        $ApiRepo = "$apiRoot/repos/$Slug"
    } else {
        $ApiRepo = ($RepoUrl -replace '^(https?)://([^/]+)/(.+)$', '$1://$2/api/v1/repos/$3')
    }

    function Get-HttpStatus($ErrorRecord) {
        try { return [int]$ErrorRecord.Exception.Response.StatusCode } catch { return 0 }
    }

    function Show-NextSteps {
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  fllm-rs devices                       # what fLLM sees on this machine (GPUs, memory, capabilities)"
        Write-Host "  fllm-rs run <huggingface link>        # download a model and chat (--device auto picks the best GPU, else the CPU)"
    }

    # PATH: consent first. -AddToPath, or a yes at the prompt of an interactive host. Never the machine PATH.
    function Update-PathForInstall {
        $entries = Split-PathList (Get-UserPath)
        if (Test-InList $entries $InstallDir) {
            Write-Host "$InstallDir is already on your user PATH."
            return
        }
        $consent = $false
        if ($AddToPath) { $consent = $true }
        elseif (-not $NoAddToPath -and (Test-Interactive)) {
            $answer = Read-Host "Add $InstallDir to your user PATH so that 'fllm-rs' works in any terminal? [Y/n]"
            $consent = ($answer -eq "" -or $answer -match '^[yY]')
        }
        if ($consent) {
            $new = (@($entries) + $InstallDir) -join ";"
            Set-UserPath $new
            $env:PATH = "$env:PATH;$InstallDir"
            Write-Host "Added $InstallDir to the user PATH. Terminals opened from now on find 'fllm-rs'; this one does already."
        } else {
            Write-Host "$InstallDir is not on your PATH. To call 'fllm-rs' from anywhere, run this (user PATH, no administrator needed):"
            Write-Host "  [Environment]::SetEnvironmentVariable('PATH', [Environment]::GetEnvironmentVariable('PATH', 'User') + ';$InstallDir', 'User')"
            Write-Host "or run the installer again with -AddToPath."
        }
    }

    function Complete-Install([string]$Installed) {
        # does it start? (a missing DLL or an unsupported CPU shows up here and not at the first model download)
        $started = $false
        try {
            $text = & $Installed --version 2>&1
            if ($LASTEXITCODE -eq 0) { Write-Host ($text | Out-String).Trim(); $started = $true }
        } catch { }
        if (-not $started) {
            Write-Host "install.ps1: WARNING - the installed binary did not start."
            Write-Host "  The MSVC assets need no extra files (static C runtime). If Windows blocked it (SmartScreen/Defender: the binary is unsigned),"
            Write-Host "  see the notes in the header of this script. WSL2 ('wsl') or Docker Desktop ('docker compose up -d') are alternatives."
        }
        Update-PathForInstall
        Show-NextSteps
    }

    # ---------------------------------------------------------------- build from source
    function Build-FromSource {
        if (-not $Here -or -not (Test-Path (Join-Path $Here "Cargo.toml"))) {
            throw "the source build needs a checkout of the repository: git clone $DefaultRepo; cd fLLM; .\install.ps1 -Build"
        }
        Write-Host "Building fLLM from source in $Here (needs cargo and the Visual Studio C++ build tools)"
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            throw "cargo not found - install a Rust toolchain (https://rustup.rs)"
        }
        $script = Join-Path (Join-Path $Here "scripts") "build-windows.ps1"
        $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Here "target" }
        if ((Test-Path $script) -and -not $env:RUST_FEATURES) {
            $buildArgs = @("-SkipPackage", "-SkipTests")
            if ($Chosen -eq "cuda") { $buildArgs += "-Cuda" }
            if ($Chosen -eq "hip") { $buildArgs += "-Hip" }
            if ($IsArm) { $buildArgs += "-Arm64" }
            & $script @buildArgs
            if ($LASTEXITCODE -ne 0) { throw "scripts\build-windows.ps1 failed" }
            $triple = if ($IsArm) { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
            $built = Join-Path (Join-Path (Join-Path $targetDir $triple) "release") "fllm-rs.exe"
        } else {
            Push-Location $Here
            try {
                $features = if ($env:RUST_FEATURES) { $env:RUST_FEATURES } else { "rust-gpu" }
                & cargo build --release --locked -p fllm-engine --features $features --bin fllm-rs
                if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
            } finally { Pop-Location }
            $built = Join-Path (Join-Path $targetDir "release") "fllm-rs.exe"
        }
        if (-not (Test-Path $built)) { throw "build finished but $built is missing" }
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        Copy-Item $built $Exe -Force
        Write-Host "Installed: $Exe"
        Complete-Install $Exe
    }

    if ($Build) { Build-FromSource; return }

    # ---------------------------------------------------------------- resolve the tag
    if ($Version -eq "latest") {
        if ($Forge -eq "github") {
            # /releases/latest redirects to /releases/tag/<tag>; it ignores pre-releases (v0.x), then there is no target
            $Version = ""
            try {
                $head = Invoke-WebRequest -Uri "$RepoUrl/releases/latest" -Method Head -UseBasicParsing -TimeoutSec 30
                $final = ""
                if ($head.BaseResponse.PSObject.Properties["ResponseUri"]) { $final = $head.BaseResponse.ResponseUri.AbsoluteUri }       # Windows PowerShell 5
                elseif ($head.BaseResponse.PSObject.Properties["RequestMessage"]) { $final = $head.BaseResponse.RequestMessage.RequestUri.AbsoluteUri }  # PowerShell 7
                if ($final -match '/releases/tag/(.+)$') { $Version = [System.Uri]::UnescapeDataString($Matches[1]) }
            } catch { }
            if (-not $Version) {
                $api = "$ApiRepo/releases?per_page=1"
                try { $releases = @(Invoke-RestMethod -Uri $api -TimeoutSec 30) }
                catch { throw "cannot query $api - is $RepoUrl reachable? (give a tag with -Version, or use -Build from a clone)" }
                if ($releases.Count -gt 0) { $Version = $releases[0].tag_name }
                if (-not $Version) { throw "no release found via $api" }
            }
        } else {
            try { $release = Invoke-RestMethod -Uri "$ApiRepo/releases/latest" -TimeoutSec 30 }
            catch { throw "cannot query $ApiRepo/releases/latest - is $RepoUrl reachable?" }
            $Version = $release.tag_name
            if (-not $Version) { throw "no release found via $ApiRepo/releases/latest" }
        }
        Write-Host "Newest release: $Version"
    }

    # ---------------------------------------------------------------- download + verify
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
                Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 600 | Out-Null
                $Asset = $candidate
                $Zip = $target
                break
            } catch {
                $status = Get-HttpStatus $_
                if ($status -eq 404 -or $status -eq 410) {
                    Write-Host "install.ps1: no release asset '$candidate' for tag '$Version'"
                } else {
                    throw "download of $url failed ($($_.Exception.Message)); check the network, proxy and TLS settings"
                }
            }
        }
        if (-not $Asset) {
            Write-Host "install.ps1: the release '$Version' has no Windows asset for this machine ($ReleaseBase)"
            if ($Here -and (Test-Path (Join-Path $Here "Cargo.toml"))) { Build-FromSource; return }
            throw "nothing to install. Check the tag (-Version) and $ReleaseBase, or build from a clone: git clone $DefaultRepo; cd fLLM; .\install.ps1 -Build"
        }
        if ($IsArm -and $Asset -notlike "*aarch64*") {
            Write-Host "install.ps1: WARNING - installing the x86_64 build on ARM64; it runs in Windows' x64 emulation, which is slow"
        }
        if ($Chosen -ne "default" -and $Asset -notlike "*-$Chosen.zip") {
            Write-Host "install.ps1: the release has no $Chosen build; installing the default build instead (CPU, Vulkan, DirectX 12)"
        }
        $note = if ($GpuNote) { " ($GpuNote)" } else { "" }
        Write-Host "Variant: $Chosen$note, asset: $Asset"
        $AssetUrl = "$ReleaseBase/download/$Version/$Asset"

        # expected hash 1: the <asset>.sha256 file of the release
        $wantFile = ""
        $Sum = "$Zip.sha256"
        try {
            Invoke-WebRequest -Uri "$AssetUrl.sha256" -OutFile $Sum -UseBasicParsing -TimeoutSec 60 | Out-Null
            $wantFile = ((Get-Content $Sum -Raw) -split '\s+')[0].ToLowerInvariant()
            if (-not $wantFile) { throw "$Asset.sha256 is empty; refusing to install" }
        } catch {
            if ($_.Exception.Message -like "*refusing*") { throw }
            $wantFile = ""
        }
        # expected hash 2: a "<sha256>  <asset>" list in the release notes, when there is one
        $wantNotes = ""
        try {
            $rel = Invoke-RestMethod -Uri "$ApiRepo/releases/tags/$Version" -TimeoutSec 30
            if ($rel.body -match ("([0-9a-fA-F]{64})\W{1,16}" + [regex]::Escape($Asset))) { $wantNotes = $Matches[1].ToLowerInvariant() }
        } catch { }
        $actual = (Get-FileHash -Algorithm SHA256 -Path $Zip).Hash.ToLowerInvariant()
        if ($wantFile -and $wantNotes -and $wantFile -ne $wantNotes) {
            throw "the .sha256 file ($wantFile) and the release notes ($wantNotes) disagree for $Asset; refusing to install"
        }
        $expected = if ($wantFile) { $wantFile } else { $wantNotes }
        if ($expected) {
            if ($expected -ne $actual) {
                Write-Host "install.ps1: CHECKSUM MISMATCH for $Asset"
                Write-Host "  expected: $expected"
                Write-Host "  actual:   $actual"
                throw "the download was NOT installed; the asset and its checksum do not belong together"
            }
            if ($wantFile -and $wantNotes) { Write-Host "SHA256 OK: $actual (matches the .sha256 file and the release notes)" }
            elseif ($wantFile) { Write-Host "SHA256 OK: $actual (matches the release's .sha256 file; the release notes carry no checksum list)" }
            else { Write-Host "SHA256 OK: $actual (matches the release notes; the release has no .sha256 file)" }
        } elseif ($NoVerify) {
            Write-Host "install.ps1: WARNING - -NoVerify: installing WITHOUT checksum verification (sha256 of the file: $actual)"
        } else {
            throw "the release for '$Version' publishes no checksum for $Asset, so the download cannot be verified. Re-run with -NoVerify to install it anyway"
        }

        $Unpack = Join-Path $Tmp "unpacked"
        Expand-Archive -Path $Zip -DestinationPath $Unpack -Force
        $Binary = Join-Path $Unpack "fllm-rs.exe"
        if (-not (Test-Path $Binary)) { throw "$Asset does not contain fllm-rs.exe" }
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        Copy-Item $Binary $Exe -Force
        try { Unblock-File -Path $Exe -ErrorAction Stop } catch { }   # removes a "downloaded from the internet" mark when there is one
        Write-Host "Installed ${Version}: $Exe"
        Write-Host "Note: the binary is not code-signed; Windows SmartScreen or Defender may warn about it (see the header of install.ps1)."
        Complete-Install $Exe
    } finally {
        if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

try {
    Install-Fllm @PSBoundParameters
} catch {
    Write-Host "install.ps1: $($_.Exception.Message)" -ForegroundColor Red
    $global:LASTEXITCODE = 1
    # a script file ends the process with the error code; through `iex` or a script block the session stays open
    if ($PSCommandPath) { exit 1 }
}
