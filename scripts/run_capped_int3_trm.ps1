param(
  [string]$TrainingTaskId = "int3-campsite-policy-trm-v1",
  [string]$Source = $env:HERMES_INTELLECT3_SOURCE,
  [int]$MaxSteps = 5000,
  [int]$BatchSize = 32,
  [int]$CheckpointSteps = 250,
  [int]$CheckpointSeconds = 60,
  [int]$RamMb = 2048,
  [int]$CpuPct = 50,
  [int]$IoMbS = 50,
  [int]$WallSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Source)) {
  $Source = Join-Path $RepoRoot "data\intellect_3_logic.jsonl"
}
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
  throw "Intellect-3 source not found. Pass -Source or set HERMES_INTELLECT3_SOURCE."
}
$RunId = "$TrainingTaskId-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$RunDir = Join-Path $RepoRoot "experiments\intellect3-campsite\training\$RunId"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$freeRamMb = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
if ($freeRamMb -lt ($RamMb + 512)) {
  @{ status = "preflight_failed"; reason = "insufficient_free_ram"; free_ram_mb = $freeRamMb; required_ram_mb = ($RamMb + 512) } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $RunDir "preflight.json")
  throw "Training requires at least $($RamMb + 512) MB free RAM."
}

if (-not ("Int3JobObject" -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Int3JobObject {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int t, IntPtr i, uint s);
  public const int Extended = 9;
  public const int Cpu = 15;
  public const uint ProcessMemory = 0x100;
  public const uint JobMemory = 0x200;
  public const uint CpuEnable = 0x1;
  public const uint CpuHardCap = 0x4;
  [StructLayout(LayoutKind.Sequential)] public struct IO_COUNTERS { public ulong a,b,c,d,e,f; }
  [StructLayout(LayoutKind.Sequential)] public struct BASIC { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public long f; public uint g,h; }
  [StructLayout(LayoutKind.Sequential)] public struct EXTENDED { public BASIC basic; public IO_COUNTERS io; public UIntPtr processMemory; public UIntPtr jobMemory; public UIntPtr peakProcess; public UIntPtr peakJob; }
  [StructLayout(LayoutKind.Sequential)] public struct CPU_RATE { public uint flags; public uint rate; }
}
'@
}

function Set-Struct($Job, [int]$Type, $Value) {
  $size = [System.Runtime.InteropServices.Marshal]::SizeOf($Value)
  $ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
  try {
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($Value, $ptr, $false)
    if (-not [Int3JobObject]::SetInformationJobObject($Job, $Type, $ptr, $size)) { throw "SetInformationJobObject failed" }
  } finally {
    [System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
  }
}

$job = [Int3JobObject]::CreateJobObject([IntPtr]::Zero, $RunId)
if ($job -eq [IntPtr]::Zero) { throw "CreateJobObject failed" }
$memory = New-Object Int3JobObject+EXTENDED
$memory.basic.flags = [Int3JobObject]::ProcessMemory -bor [Int3JobObject]::JobMemory
$memory.processMemory = [UIntPtr]([UInt64]$RamMb * 1024 * 1024)
$memory.jobMemory = [UIntPtr]([UInt64]$RamMb * 1024 * 1024)
Set-Struct $job ([Int3JobObject]::Extended) $memory
$cpu = New-Object Int3JobObject+CPU_RATE
$cpu.flags = [Int3JobObject]::CpuEnable -bor [Int3JobObject]::CpuHardCap
$cpu.rate = [uint32]($CpuPct * 100)
Set-Struct $job ([Int3JobObject]::Cpu) $cpu

$env:INT3_CAP_WRAPPER_ACTIVE = "1"
$env:INT3_TRAINING_TASK_ID = $TrainingTaskId
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$stdout = Join-Path $RunDir "trainer.stdout.log"
$stderr = Join-Path $RunDir "trainer.stderr.log"
$arguments = @(
  "-m", "agent.intellect3_policy_trm",
  "--source", $Source,
  "--run-dir", $RunDir,
  "--max-steps", "$MaxSteps",
  "--batch-size", "$BatchSize",
  "--checkpoint-steps", "$CheckpointSteps",
  "--checkpoint-seconds", "$CheckpointSeconds",
  "--io-cap-mb-s", "$IoMbS"
)
$manifest = @{
  training_task_id = $TrainingTaskId
  run_id = $RunId
  caps = @{ ram_mb = $RamMb; cpu_pct = $CpuPct; io_mb_s = $IoMbS; wall_seconds = $WallSeconds }
  checkpoint = @{ steps = $CheckpointSteps; seconds = $CheckpointSeconds }
  chunk_strategy = "256-frame shards; minibatch $BatchSize"
  started_at = (Get-Date).ToUniversalTime().ToString("o")
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_manifest.json")

$process = Start-Process -FilePath "python" -ArgumentList $arguments -WorkingDirectory $RepoRoot -PassThru `
  -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
if (-not [Int3JobObject]::AssignProcessToJobObject($job, $process.Handle)) {
  Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  throw "AssignProcessToJobObject failed for PID $($process.Id)"
}
$manifest.pid = $process.Id
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_manifest.json")

$timedOut = $false
try {
  if (-not $process.WaitForExit($WallSeconds * 1000)) {
    $timedOut = $true
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
} finally {
  & (Join-Path $PSScriptRoot "post_run_int3_cleanup.ps1") -RunId $RunId -OwnedPid $process.Id -SummaryPath (Join-Path $RunDir "cleanup_summary.json")
}

$summary = @{
  training_task_id = $TrainingTaskId
  run_id = $RunId
  status = if ($timedOut) { "aborted" } elseif ($process.ExitCode -eq 0) { "completed" } else { "failed" }
  abort_reason = if ($timedOut) { "wall_clock_cap" } else { "" }
  exit_code = $process.ExitCode
  completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_summary.json")
Write-Output ($summary | ConvertTo-Json -Depth 4)
if (-not $timedOut -and $process.ExitCode -ne 0) { throw "Trainer failed; inspect $RunDir" }
