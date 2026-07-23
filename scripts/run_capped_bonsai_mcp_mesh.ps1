param(
  [string]$RegistrationDir = "evals\registered\bonsai_mcp_skill_mesh_v0",
  [string]$RegistrationId = "fdaa6dce3853b05e8dba52cf0b1c935b32123e25e746bdce317708616767679f",
  [string]$LlamaDir = "D:\models\Tesseract\runtime\llama.cpp\b10064\payload",
  [string]$ModelPath = "D:\models\Tesseract\bonsai-8b-q1\Bonsai-8B-Q1_0.gguf",
  [ValidateSet("screening", "confirmation")]
  [string]$Stage = "screening",
  [int]$RamMb = 2048,
  [int]$CpuPct = 50,
  [int]$IoMbS = 50,
  [int]$WallSeconds = 1800,
  [int]$Port = 8801,
  [int]$Context = 12288,
  [ValidatePattern("^(auto|all|[0-9]+)$")]
  [string]$GpuLayers = "auto",
  [int]$Threads = 6,
  [int]$ThreadsBatch = 4,
  [int]$Batch = 128,
  [int]$UBatch = 64,
  [int]$ParallelSlots = 1,
  [int]$CacheRamMb = 0,
  [ValidateSet("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")]
  [string]$CacheTypeK = "f16",
  [ValidateSet("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")]
  [string]$CacheTypeV = "f16",
  [int]$FitTargetMb = 1024,
  [int]$MinFreeVramMb = 512,
  [int]$MaxTempC = 86,
  [int]$StartupSeconds = 120,
  [int]$RequestTimeoutSeconds = 180,
  [switch]$ValidateOnly,
  [switch]$JobObjectProbe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ServerExe = Join-Path $LlamaDir "llama-server.exe"
$RegistrationDir = if ([IO.Path]::IsPathRooted($RegistrationDir)) {
  [IO.Path]::GetFullPath($RegistrationDir)
} else {
  [IO.Path]::GetFullPath((Join-Path $RepoRoot $RegistrationDir))
}
$RunId = "bonsai-mcp-mesh-$Stage-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$LiveRoot = Join-Path $RepoRoot "experiments\mcp_skill_mesh_v0\live"
$RunDir = Join-Path $LiveRoot "wrappers\$RunId"
$EventsPath = Join-Path $RunDir "resource_events.jsonl"
$ManifestPath = Join-Path $RunDir "wrapper_manifest.json"
$ResourceReceiptPath = Join-Path $RunDir "resource_receipt.json"
$ServerStdout = Join-Path $RunDir "llama.stdout.log"
$ServerStderr = Join-Path $RunDir "llama.stderr.log"
$EvalStdout = Join-Path $RunDir "eval.stdout.log"
$EvalStderr = Join-Path $RunDir "eval.stderr.log"

function Write-JsonFile {
  param([string]$Path, $Value)
  $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-Event {
  param($Value)
  ($Value | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $EventsPath -Encoding UTF8
}

function Test-TcpPort {
  param([string]$HostName, [int]$TcpPort, [int]$TimeoutMs = 500)
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $pending = $client.BeginConnect($HostName, $TcpPort, $null, $null)
    if (-not $pending.AsyncWaitHandle.WaitOne($TimeoutMs)) {
      return $false
    }
    $client.EndConnect($pending)
    return $true
  } catch {
    return $false
  } finally {
    $client.Close()
  }
}

function Test-ServerReady {
  param([int]$TcpPort)
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$TcpPort/health" -TimeoutSec 2
    return $health.status -eq "ok"
  } catch {
    return $false
  }
}

function Stop-OwnedProcess {
  param([int]$OwnedPid)
  if ($OwnedPid -le 0) {
    return $false
  }
  $process = Get-Process -Id $OwnedPid -ErrorAction SilentlyContinue
  if ($null -eq $process) {
    return $false
  }
  Stop-Process -Id $OwnedPid -Force -ErrorAction SilentlyContinue
  try {
    $process.WaitForExit(10000)
  } catch {
  }
  return $true
}

function Get-ProcessSample {
  param([int[]]$OwnedPids)
  $workingSetBytes = 0.0
  $privateBytes = 0.0
  $ioBytes = 0.0
  $alive = @()
  foreach ($ownedPid in $OwnedPids) {
    if ($ownedPid -le 0) {
      continue
    }
    $process = Get-Process -Id $ownedPid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
      continue
    }
    $alive += $ownedPid
    $workingSetBytes += [double]$process.WorkingSet64
    $privateBytes += [double]$process.PrivateMemorySize64
    $readBytes = if ($null -ne $process.PSObject.Properties["IOReadBytes"]) {
      [double]$process.IOReadBytes
    } else {
      0.0
    }
    $writeBytes = if ($null -ne $process.PSObject.Properties["IOWriteBytes"]) {
      [double]$process.IOWriteBytes
    } else {
      0.0
    }
    $ioBytes += $readBytes + $writeBytes
  }
  return @{
    alive_pids = $alive
    private_mb = [math]::Round($privateBytes / 1MB, 3)
    working_set_mb = [math]::Round($workingSetBytes / 1MB, 3)
    io_bytes = $ioBytes
  }
}

function Get-GpuSnapshot {
  param([int[]]$OwnedPids)
  $result = @{
    available = $false
    temperature_c = 0
    memory_used_mb = 0
    memory_free_mb = 0
    owned_gpu_memory_mb = 0
  }
  try {
    $gpu = & nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
      $gpuLine = @($gpu)[0]
      $parts = @($gpuLine.Split(",") | ForEach-Object { $_.Trim() })
      if ($parts.Count -ge 3) {
        $result.available = $true
        $result.temperature_c = [int]$parts[0]
        $result.memory_used_mb = [int]$parts[1]
        $result.memory_free_mb = [int]$parts[2]
      }
    }
    $apps = & nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
    foreach ($line in @($apps)) {
      $parts = @($line.Split(",") | ForEach-Object { $_.Trim() })
      if ($parts.Count -ge 2 -and $OwnedPids -contains [int]$parts[0]) {
        $result.owned_gpu_memory_mb += [int]$parts[1]
      }
    }
  } catch {
  }
  return $result
}

foreach ($requiredPath in @($Python, $ServerExe, $ModelPath, $RegistrationDir)) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required path not found: $requiredPath"
  }
}
if ($RamMb -lt 1024 -or $CpuPct -lt 1 -or $CpuPct -gt 100 -or $IoMbS -lt 1 -or $WallSeconds -lt 60) {
  throw "Invalid resource cap."
}
if ($Context -gt 12288) {
  throw "Context exceeds the registered 12k implementation ceiling."
}
if (Test-TcpPort -HostName "127.0.0.1" -TcpPort $Port) {
  throw "Port $Port is already occupied. Refusing to take over an unowned endpoint."
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$verifyArgs = @(
  "-m", "agent.mcp_skill_mesh_gym", "verify",
  "--registration-dir", $RegistrationDir,
  "--registration-id", $RegistrationId
)
& $Python @verifyArgs *> (Join-Path $RunDir "registration_verify.log")
if ($LASTEXITCODE -ne 0) {
  throw "Registration verification failed."
}

$freeRamMb = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
if ($freeRamMb -lt ($RamMb + 512)) {
  throw "Run requires at least $($RamMb + 512) MB free RAM; found $freeRamMb MB."
}

$runtimeEvidence = @(
  "llama-server.exe",
  "ggml.dll",
  "llama.dll",
  "ggml-cuda.dll",
  "cudart64_12.dll",
  "cublas64_12.dll",
  "cublasLt64_12.dll"
) | ForEach-Object {
  $runtimePath = Join-Path $LlamaDir $_
  if (Test-Path -LiteralPath $runtimePath) {
    @{
      name = $_
      bytes = (Get-Item -LiteralPath $runtimePath).Length
      sha256 = (Get-FileHash -LiteralPath $runtimePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
  }
}

$manifest = [ordered]@{
  run_id = $RunId
  study_id = "bonsai_mcp_skill_mesh_complexity_v0"
  registration_id = $RegistrationId
  stage = $Stage
  registration_dir = $RegistrationDir
  live_output_dir = $LiveRoot
  model_path = $ModelPath
  server_exe = $ServerExe
  server_exe_sha256 = (Get-FileHash -LiteralPath $ServerExe -Algorithm SHA256).Hash.ToLowerInvariant()
  runtime_files = $runtimeEvidence
  endpoint = "http://127.0.0.1:$Port/v1"
  caps = @{
    ram_mb = $RamMb
    cpu_pct = $CpuPct
    io_mb_s = $IoMbS
    wall_seconds = $WallSeconds
    min_free_vram_mb = $MinFreeVramMb
    max_temp_c = $MaxTempC
  }
  inference = @{
    context = $Context
    gpu_layers = $GpuLayers
    threads = $Threads
    threads_batch = $ThreadsBatch
    batch = $Batch
    ubatch = $UBatch
    parallel_slots = $ParallelSlots
    cache_ram_mb = $CacheRamMb
    cache_type_k = $CacheTypeK
    cache_type_v = $CacheTypeV
    fit = "on"
    fit_target_mb = $FitTargetMb
    request_timeout_seconds = $RequestTimeoutSeconds
  }
  checkpoint_cadence = "after every live cell"
  chunk_strategy = "one case-arm response per request"
  pressure_monitor = @{
    authority = "external_wrapper"
    cadence_seconds = 1
    evaluator_subprocess_probes = $false
  }
  owned_pids = @()
  created_at = (Get-Date).ToUniversalTime().ToString("o")
  validate_only = [bool]$ValidateOnly
  job_object_probe = [bool]$JobObjectProbe
}
Write-JsonFile -Path $ManifestPath -Value $manifest

if ($ValidateOnly) {
  $result = @{
    status = "validated_no_processes_started"
    run_id = $RunId
    run_dir = $RunDir
    registration_id = $RegistrationId
    caps = $manifest.caps
    free_ram_mb = $freeRamMb
  }
  Write-JsonFile -Path $ResourceReceiptPath -Value $result
  $result | ConvertTo-Json -Depth 6
  exit 0
}

if (-not ("BonsaiMeshJob" -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class BonsaiMeshJob {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int t, IntPtr i, uint s);
  [DllImport("kernel32.dll", SetLastError = true)] public static extern bool QueryInformationJobObject(IntPtr j, int t, IntPtr i, uint s, out uint r);
  public const int Extended = 9; public const int Cpu = 15;
  public const uint ProcessMemory = 0x100; public const uint JobMemory = 0x200;
  public const uint CpuEnable = 0x1; public const uint CpuHardCap = 0x4;
  [StructLayout(LayoutKind.Sequential)] public struct IO_COUNTERS { public ulong a,b,c,d,e,f; }
  [StructLayout(LayoutKind.Sequential)] public struct BASIC { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public long f; public uint g,h; }
  [StructLayout(LayoutKind.Sequential)] public struct EXTENDED { public BASIC basic; public IO_COUNTERS io; public UIntPtr processMemory; public UIntPtr jobMemory; public UIntPtr peakProcess; public UIntPtr peakJob; }
  [StructLayout(LayoutKind.Sequential)] public struct CPU_RATE { public uint flags; public uint rate; }
  public static bool QueryExtended(IntPtr job, out EXTENDED value) {
    uint returned;
    IntPtr pointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(EXTENDED)));
    try {
      if (!QueryInformationJobObject(job, Extended, pointer, (uint)Marshal.SizeOf(typeof(EXTENDED)), out returned)) {
        value = new EXTENDED();
        return false;
      }
      value = (EXTENDED)Marshal.PtrToStructure(pointer, typeof(EXTENDED));
      return true;
    } finally {
      Marshal.FreeHGlobal(pointer);
    }
  }
}
'@
}

function Set-JobStruct {
  param($Job, [int]$Type, $Value)
  $size = [System.Runtime.InteropServices.Marshal]::SizeOf($Value)
  $pointer = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
  try {
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($Value, $pointer, $false)
    if (-not [BonsaiMeshJob]::SetInformationJobObject($Job, $Type, $pointer, $size)) {
      throw "SetInformationJobObject failed for type $Type."
    }
  } finally {
    [System.Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
  }
}

function Get-JobMemoryAccounting {
  param($Job)
  $accounting = New-Object BonsaiMeshJob+EXTENDED
  if (-not [BonsaiMeshJob]::QueryExtended($Job, [ref]$accounting)) {
    return @{
      available = $false
      error_code = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
      limit_flags = 0
      configured_process_memory_limit_mb = 0
      configured_job_memory_limit_mb = 0
      peak_process_memory_mb = 0
      peak_job_memory_mb = 0
    }
  }
  return @{
    available = $true
    error_code = 0
    limit_flags = [uint32]$accounting.basic.flags
    configured_process_memory_limit_mb = [math]::Round($accounting.processMemory.ToUInt64() / 1MB, 3)
    configured_job_memory_limit_mb = [math]::Round($accounting.jobMemory.ToUInt64() / 1MB, 3)
    peak_process_memory_mb = [math]::Round($accounting.peakProcess.ToUInt64() / 1MB, 3)
    peak_job_memory_mb = [math]::Round($accounting.peakJob.ToUInt64() / 1MB, 3)
  }
}

$job = [BonsaiMeshJob]::CreateJobObject([IntPtr]::Zero, $RunId)
if ($job -eq [IntPtr]::Zero) {
  throw "CreateJobObject failed."
}
$memory = New-Object BonsaiMeshJob+EXTENDED
$basicLimits = $memory.basic
$basicLimits.flags = [BonsaiMeshJob]::ProcessMemory -bor [BonsaiMeshJob]::JobMemory
$memory.basic = $basicLimits
$memory.processMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
$memory.jobMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
Set-JobStruct -Job $job -Type ([BonsaiMeshJob]::Extended) -Value $memory
$cpu = New-Object BonsaiMeshJob+CPU_RATE
$cpu.flags = [BonsaiMeshJob]::CpuEnable -bor [BonsaiMeshJob]::CpuHardCap
$cpu.rate = [uint32]($CpuPct * 100)
Set-JobStruct -Job $job -Type ([BonsaiMeshJob]::Cpu) -Value $cpu

if ($JobObjectProbe) {
  $probeProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-Command", "Start-Sleep -Milliseconds 500") `
    -PassThru `
    -WindowStyle Hidden
  try {
    if (-not [BonsaiMeshJob]::AssignProcessToJobObject($job, $probeProcess.Handle)) {
      throw "AssignProcessToJobObject failed for probe PID $($probeProcess.Id)."
    }
    $probeProcess.WaitForExit()
    $probeAccounting = Get-JobMemoryAccounting -Job $job
  } finally {
    Stop-OwnedProcess -OwnedPid $probeProcess.Id | Out-Null
  }
  $probeResult = @{
    status = if ($probeAccounting.available) { "job_object_probe_passed" } else { "job_object_probe_failed" }
    run_id = $RunId
    pid = $probeProcess.Id
    accounting = $probeAccounting
  }
  Write-JsonFile -Path $ResourceReceiptPath -Value $probeResult
  $probeResult | ConvertTo-Json -Depth 6
  if (-not $probeAccounting.available) {
    exit 1
  }
  exit 0
}

$serverArgs = @(
  "--host", "127.0.0.1",
  "--port", "$Port",
  "-m", $ModelPath,
  "-c", "$Context",
  "-ngl", "$GpuLayers",
  "-t", "$Threads",
  "-tb", "$ThreadsBatch",
  "-b", "$Batch",
  "-ub", "$UBatch",
  "--parallel", "$ParallelSlots",
  "--cache-ram", "$CacheRamMb",
  "--cache-type-k", $CacheTypeK,
  "--cache-type-v", $CacheTypeV,
  "--fit", "on",
  "--fit-target", "$FitTargetMb",
  "--no-webui",
  "--no-warmup"
)
$evalArgs = @(
  "-m", "agent.mcp_skill_mesh_gym", "live",
  "--registration-dir", $RegistrationDir,
  "--output-dir", $LiveRoot,
  "--confirm-registration-id", $RegistrationId,
  "--base-url", "http://127.0.0.1:$Port/v1",
  "--model", "local/bonsai-8b",
  "--stage", $Stage,
  "--timeout-s", "$RequestTimeoutSeconds",
  "--max-tokens", "128",
  "--min-free-mb", "$MinFreeVramMb",
  "--max-temp-c", "$MaxTempC",
  "--external-pressure-monitor"
)

$server = $null
$evaluator = $null
$abortReason = ""
$peakRamMb = 0.0
$peakWorkingSetMb = 0.0
$sumRamMb = 0.0
$sampleCount = 0
$peakIoMbS = 0.0
$ioExcessStreak = 0
$lastIoBytes = 0.0
$lastSampleAt = Get-Date
$startAt = Get-Date
$cleanupStopped = @()

try {
  $server = Start-Process `
    -FilePath $ServerExe `
    -ArgumentList $serverArgs `
    -WorkingDirectory $LlamaDir `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $ServerStdout `
    -RedirectStandardError $ServerStderr
  if (-not [BonsaiMeshJob]::AssignProcessToJobObject($job, $server.Handle)) {
    throw "AssignProcessToJobObject failed for server PID $($server.Id)."
  }
  $manifest.owned_pids = @($server.Id)
  Write-JsonFile -Path $ManifestPath -Value $manifest
  Write-Event @{
    ts = (Get-Date).ToUniversalTime().ToString("o")
    event = "server_started"
    pid = $server.Id
  }

  $startupDeadline = (Get-Date).AddSeconds($StartupSeconds)
  while ((Get-Date) -lt $startupDeadline) {
    if ($server.HasExited) {
      $abortReason = "server_startup_failure"
      break
    }
    if (Test-ServerReady -TcpPort $Port) {
      break
    }
    Start-Sleep -Milliseconds 500
  }
  if (-not $abortReason -and -not (Test-ServerReady -TcpPort $Port)) {
    $abortReason = "server_startup_timeout"
  }
  if ($abortReason) {
    throw $abortReason
  }

  $evaluator = Start-Process `
    -FilePath $Python `
    -ArgumentList $evalArgs `
    -WorkingDirectory $RepoRoot `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $EvalStdout `
    -RedirectStandardError $EvalStderr
  if (-not [BonsaiMeshJob]::AssignProcessToJobObject($job, $evaluator.Handle)) {
    throw "AssignProcessToJobObject failed for evaluator PID $($evaluator.Id)."
  }
  $manifest.owned_pids = @($server.Id, $evaluator.Id)
  Write-JsonFile -Path $ManifestPath -Value $manifest
  Write-Event @{
    ts = (Get-Date).ToUniversalTime().ToString("o")
    event = "evaluation_started"
    pid = $evaluator.Id
  }

  while (-not $evaluator.HasExited) {
    Start-Sleep -Seconds 1
    $now = Get-Date
    $sample = Get-ProcessSample -OwnedPids @($server.Id, $evaluator.Id)
    $elapsed = [math]::Max(0.001, ($now - $lastSampleAt).TotalSeconds)
    $ioMbS = [math]::Max(0.0, ($sample.io_bytes - $lastIoBytes) / 1MB / $elapsed)
    $lastIoBytes = $sample.io_bytes
    $lastSampleAt = $now
    $peakRamMb = [math]::Max($peakRamMb, [double]$sample.private_mb)
    $peakWorkingSetMb = [math]::Max($peakWorkingSetMb, [double]$sample.working_set_mb)
    $peakIoMbS = [math]::Max($peakIoMbS, $ioMbS)
    $sumRamMb += [double]$sample.private_mb
    $sampleCount += 1
    $ioExcessStreak = if ($ioMbS -gt $IoMbS) { $ioExcessStreak + 1 } else { 0 }
    $gpu = Get-GpuSnapshot -OwnedPids @($server.Id, $evaluator.Id)
    $kernelMemory = Get-JobMemoryAccounting -Job $job
    Write-Event @{
      ts = $now.ToUniversalTime().ToString("o")
      event = "resource_sample"
      private_mb = $sample.private_mb
      working_set_mb = $sample.working_set_mb
      io_mb_s = [math]::Round($ioMbS, 3)
      kernel_job_memory = $kernelMemory
      gpu = $gpu
      alive_pids = $sample.alive_pids
    }
    if ($ioExcessStreak -ge 3) {
      $abortReason = "sustained_io_over_cap"
      break
    }
    if ([double]$sample.private_mb -gt [double]$RamMb) {
      $abortReason = "ram_cap_exceeded"
      break
    }
    if ($gpu.available -and ($gpu.temperature_c -gt $MaxTempC -or $gpu.memory_free_mb -lt $MinFreeVramMb)) {
      $abortReason = "gpu_pressure_gate"
      break
    }
    if (($now - $startAt).TotalSeconds -gt $WallSeconds) {
      $abortReason = "wall_clock_cap"
      break
    }
  }
  if ($abortReason) {
    Stop-OwnedProcess -OwnedPid $evaluator.Id | Out-Null
  } else {
    $evaluator.WaitForExit()
    if ($evaluator.ExitCode -ne 0) {
      $innerSummaryPath = Join-Path $LiveRoot "live_$Stage`_summary.json"
      if (Test-Path -LiteralPath $innerSummaryPath) {
        $innerSummary = Get-Content -LiteralPath $innerSummaryPath -Raw | ConvertFrom-Json
        $abortReason = if ($innerSummary.abort_reason) { [string]$innerSummary.abort_reason } else { "evaluation_failed" }
      } else {
        $abortReason = "evaluation_failed"
      }
    }
  }
} catch {
  if (-not $abortReason) {
    $abortReason = "wrapper_exception"
  }
  Write-Event @{
    ts = (Get-Date).ToUniversalTime().ToString("o")
    event = "exception"
    reason = $abortReason
    message = $_.Exception.Message
  }
} finally {
  if ($null -ne $evaluator -and (Stop-OwnedProcess -OwnedPid $evaluator.Id)) {
    $cleanupStopped += $evaluator.Id
  }
  if ($null -ne $server -and (Stop-OwnedProcess -OwnedPid $server.Id)) {
    $cleanupStopped += $server.Id
  }
}

$liveSummaryPath = Join-Path $LiveRoot "live_$Stage`_summary.json"
$liveCellsPath = Join-Path $LiveRoot "live_$Stage`_cells.jsonl"
$liveSummary = if (Test-Path -LiteralPath $liveSummaryPath) {
  Get-Content -LiteralPath $liveSummaryPath -Raw | ConvertFrom-Json
} else {
  $null
}
$ownedPids = @()
if ($null -ne $server) {
  $ownedPids += $server.Id
}
if ($null -ne $evaluator) {
  $ownedPids += $evaluator.Id
}
$gpuAfter = Get-GpuSnapshot -OwnedPids $ownedPids
$jobMemory = Get-JobMemoryAccounting -Job $job
$capEnforcementPassed = (
  $jobMemory.available -and
  [double]$jobMemory.configured_job_memory_limit_mb -eq [double]$RamMb -and
  [double]$jobMemory.peak_job_memory_mb -le ([double]$RamMb + 1.0)
)
$lingeringPids = @(
  $ownedPids | Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) }
)
$cleanupPassed = $lingeringPids.Count -eq 0 -and $gpuAfter.owned_gpu_memory_mb -eq 0
if (-not $cleanupPassed -and -not $abortReason) {
  $abortReason = "cleanup_failed"
}
if (-not $capEnforcementPassed) {
  $abortReason = if ($abortReason) {
    "$abortReason+cap_enforcement_failed"
  } else {
    "cap_enforcement_failed"
  }
}
$liveRows = @()
if (Test-Path -LiteralPath $liveCellsPath) {
  try {
    $liveRows = @(Get-Content -LiteralPath $liveCellsPath | ForEach-Object { $_ | ConvertFrom-Json })
  } catch {
    $abortReason = if ($abortReason) {
      "$abortReason+checkpoint_integrity_failure"
    } else {
      "checkpoint_integrity_failure"
    }
    Write-Event @{
      ts = (Get-Date).ToUniversalTime().ToString("o")
      event = "checkpoint_integrity_failure"
      message = $_.Exception.Message
    }
  }
}
$status = if ($abortReason) {
  "aborted"
} elseif ($null -ne $liveSummary -and $liveSummary.status -eq "completed") {
  "completed"
} else {
  "failed"
}
$completedKeys = @(
  $liveRows |
    Where-Object status -eq "completed" |
    ForEach-Object { "$($_.task_id)|$($_.arm)|$($_.seed)" } |
    Sort-Object -Unique
)
$resourceReceipt = [ordered]@{
  schema = "hermes.bonsai_mcp_mesh_resource_receipt.v0"
  run_id = $RunId
  registration_id = $RegistrationId
  stage = $Stage
  status = $status
  abort_reason = $abortReason
  caps = $manifest.caps
  peak_ram_mb = $jobMemory.peak_job_memory_mb
  avg_ram_mb = if ($sampleCount) { [math]::Round($sumRamMb / $sampleCount, 3) } else { 0 }
  sampled_peak_private_mb = [math]::Round($peakRamMb, 3)
  peak_working_set_mb = [math]::Round($peakWorkingSetMb, 3)
  peak_io_mb_s = [math]::Round($peakIoMbS, 3)
  cpu_pct = $CpuPct
  samples = $sampleCount
  steps_completed = $completedKeys.Count
  expected_steps = 144
  owned_pids = $ownedPids
  cap_enforcement = @{
    job_memory_limit_mb = $RamMb
    kernel_accounting = $jobMemory
    passed = $capEnforcementPassed
  }
  cleanup = @{
    stopped_pids = $cleanupStopped
    lingering_pids = $lingeringPids
    gpu_after = $gpuAfter
    passed = $cleanupPassed
  }
  live_summary_path = $liveSummaryPath
  completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-JsonFile -Path $ResourceReceiptPath -Value $resourceReceipt
Write-Event @{
  ts = (Get-Date).ToUniversalTime().ToString("o")
  event = "wrapper_complete"
  receipt = $resourceReceipt
}
$resourceReceipt | ConvertTo-Json -Depth 8
if ($status -ne "completed") {
  exit 1
}
