param(
    [string]$LlamaDir = $env:HERMES_LLAMA_CPP_DIR,
    [string]$ModelPath = $env:HERMES_BONSAI_MODEL,
    [string]$RuntimeDir = $env:HERMES_RUNTIME_DIR,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8801,
    [int]$Context = 12288,
    [int]$GpuLayers = 99,
    [int]$Threads = 6,
    [int]$ThreadsBatch = 4,
    [int]$Batch = 256,
    [int]$UBatch = 128,
    [int]$CacheRamMb = 2048,
    [int]$WaitSeconds = 120,
    [switch]$Warmup,
    [switch]$Foreground,
    [switch]$Status,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = Join-Path $RepoRoot ".hermes\runtime"
}

function Test-TcpPort {
    param([string]$HostName, [int]$TcpPort, [int]$TimeoutMs = 800)
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect($HostName, $TcpPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Read-Pid {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    try {
        $raw = Get-Content -LiteralPath $Path -Raw
        return [int]($raw.Trim())
    } catch {
        return $null
    }
}

function Get-OwnedProcess {
    param([string]$Path)
    $pidValue = Read-Pid -Path $Path
    if ($null -eq $pidValue) {
        return $null
    }
    return Get-Process -Id $pidValue -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

$pidFile = Join-Path $RuntimeDir "bonsai_llamacpp_8801.pid"
$manifestPath = Join-Path $RuntimeDir "bonsai_llamacpp_8801.manifest.json"
$stdoutLog = Join-Path $RuntimeDir "bonsai_llamacpp_8801.out.log"
$stderrLog = Join-Path $RuntimeDir "bonsai_llamacpp_8801.err.log"
$serverExe = if ([string]::IsNullOrWhiteSpace($LlamaDir)) {
    $null
} else {
    Join-Path $LlamaDir "llama-server.exe"
}

if ($Stop) {
    $proc = Get-OwnedProcess -Path $pidFile
    if ($null -eq $proc) {
        Write-Host "No recorded Bonsai llama.cpp process is running."
        Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
        exit 0
    }
    Write-Host "Stopping recorded Bonsai llama.cpp process PID $($proc.Id)..."
    Stop-Process -Id $proc.Id -Force
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    exit 0
}

if ($Status) {
    $proc = Get-OwnedProcess -Path $pidFile
    $alive = Test-TcpPort -HostName $HostAddress -TcpPort $Port
    if ($proc) {
        Write-Host "Recorded process: running PID $($proc.Id)"
    } else {
        Write-Host "Recorded process: not running"
    }
    Write-Host "TCP $HostAddress`:$Port`: $alive"
    if (Test-Path -LiteralPath $manifestPath) {
        Write-Host "Manifest: $manifestPath"
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($LlamaDir)) {
    throw "llama.cpp directory not configured. Pass -LlamaDir or set HERMES_LLAMA_CPP_DIR."
}
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    throw "Bonsai model not configured. Pass -ModelPath or set HERMES_BONSAI_MODEL."
}
if (-not (Test-Path -LiteralPath $serverExe)) {
    throw "llama-server.exe not found at $serverExe"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Bonsai GGUF not found at $ModelPath"
}

$existingProc = Get-OwnedProcess -Path $pidFile
if ($existingProc) {
    if (Test-TcpPort -HostName $HostAddress -TcpPort $Port) {
        Write-Host "Bonsai llama.cpp already running: PID $($existingProc.Id), http://$HostAddress`:$Port"
        exit 0
    }
    Write-Host "Removing stale PID file for PID $($existingProc.Id); port is not responding."
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
}

if (Test-TcpPort -HostName $HostAddress -TcpPort $Port) {
    throw "Port $Port is already in use, but not by the recorded Bonsai process. Refusing to take over."
}

$serverArgs = @(
    "--host", $HostAddress,
    "--port", "$Port",
    "-m", $ModelPath,
    "-c", "$Context",
    "-ngl", "$GpuLayers",
    "-t", "$Threads",
    "-tb", "$ThreadsBatch",
    "-b", "$Batch",
    "-ub", "$UBatch",
    "--cache-ram", "$CacheRamMb",
    "--no-webui"
)
if (-not $Warmup) {
    $serverArgs += "--no-warmup"
}

$manifest = [ordered]@{
    started_at = (Get-Date).ToString("o")
    server_exe = $serverExe
    model_path = $ModelPath
    endpoint = "http://$HostAddress`:$Port/v1"
    host = $HostAddress
    port = $Port
    context = $Context
    gpu_layers = $GpuLayers
    threads = $Threads
    threads_batch = $ThreadsBatch
    batch = $Batch
    ubatch = $UBatch
    cache_ram_mb = $CacheRamMb
    warmup = [bool]$Warmup
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    args = $serverArgs
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Starting Bonsai llama.cpp server..."
Write-Host "  Model:    $ModelPath"
Write-Host "  Endpoint: http://$HostAddress`:$Port/v1"
Write-Host "  Logs:     $stderrLog"

if ($Foreground) {
    & $serverExe @serverArgs
    exit $LASTEXITCODE
}

$proc = Start-Process `
    -FilePath $serverExe `
    -ArgumentList $serverArgs `
    -WorkingDirectory $LlamaDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -LiteralPath $pidFile -Encoding ASCII

$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    $running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $running) {
        Write-Host "Bonsai llama.cpp process exited before the port became ready."
        if (Test-Path -LiteralPath $stderrLog) {
            Write-Host "Last stderr lines:"
            Get-Content -LiteralPath $stderrLog -Tail 40
        }
        exit 1
    }
    if (Test-TcpPort -HostName $HostAddress -TcpPort $Port) {
        Write-Host "Bonsai llama.cpp is listening at http://$HostAddress`:$Port/v1 (PID $($proc.Id))"
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

Write-Host "Timed out waiting for http://$HostAddress`:$Port/v1"
Write-Host "PID $($proc.Id) is still running; use -Stop to stop the recorded process."
if (Test-Path -LiteralPath $stderrLog) {
    Write-Host "Last stderr lines:"
    Get-Content -LiteralPath $stderrLog -Tail 40
}
exit 2
