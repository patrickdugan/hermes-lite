param(
  [string]$RegistrationDir = "evals\registered\hermes_lite_12k_control_mesh_v1",
  [string]$TrainingTaskId = "hermes-lite-12k-control-mesh-v1-1",
  [string]$PublishedModelName = "control-mesh-v1-1",
  [int]$Steps = 400,
  [int]$RamEpochs = 40,
  [int]$Seed = 73011,
  [int]$RamMb = 2048,
  [int]$CpuPct = 50,
  [int]$IoMbS = 50,
  [int]$WallSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction Stop).Source
$RegistrationDir = if ([IO.Path]::IsPathRooted($RegistrationDir)) {
  [IO.Path]::GetFullPath($RegistrationDir)
} else {
  [IO.Path]::GetFullPath((Join-Path $RepoRoot $RegistrationDir))
}
$RunId = "$TrainingTaskId-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$RunDir = Join-Path $RepoRoot "experiments\lean-control-mesh-v1\training\$RunId"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$freeRamMb = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
if ($freeRamMb -lt ($RamMb + 512)) {
  throw "Control-mesh training requires at least $($RamMb + 512) MB free RAM."
}
& $Python -c "import psutil, torch" 2>$null
if ($LASTEXITCODE -ne 0) {
  throw "Control-mesh training requires the optional neural environment with torch and psutil."
}

if (-not ("LeanControlMeshJob" -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LeanControlMeshJob {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int t, IntPtr i, uint s);
  public const int Extended = 9; public const int Cpu = 15;
  public const uint ProcessMemory = 0x100; public const uint JobMemory = 0x200;
  public const uint CpuEnable = 0x1; public const uint CpuHardCap = 0x4;
  [StructLayout(LayoutKind.Sequential)] public struct IO_COUNTERS { public ulong a,b,c,d,e,f; }
  [StructLayout(LayoutKind.Sequential)] public struct BASIC { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public long f; public uint g,h; }
  [StructLayout(LayoutKind.Sequential)] public struct EXTENDED { public BASIC basic; public IO_COUNTERS io; public UIntPtr processMemory; public UIntPtr jobMemory; public UIntPtr peakProcess; public UIntPtr peakJob; }
  [StructLayout(LayoutKind.Sequential)] public struct CPU_RATE { public uint flags; public uint rate; }
}
'@
}

function Set-Struct($Job, [int]$Type, $Value) {
  $size = [System.Runtime.InteropServices.Marshal]::SizeOf($Value)
  $pointer = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
  try {
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($Value, $pointer, $false)
    if (-not [LeanControlMeshJob]::SetInformationJobObject($Job, $Type, $pointer, $size)) {
      throw "SetInformationJobObject failed"
    }
  } finally {
    [System.Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
  }
}

$job = [LeanControlMeshJob]::CreateJobObject([IntPtr]::Zero, $RunId)
if ($job -eq [IntPtr]::Zero) {
  throw "CreateJobObject failed"
}
$memory = New-Object LeanControlMeshJob+EXTENDED
$memory.basic.flags = [LeanControlMeshJob]::ProcessMemory -bor [LeanControlMeshJob]::JobMemory
$memory.processMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
$memory.jobMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
Set-Struct $job ([LeanControlMeshJob]::Extended) $memory
$cpu = New-Object LeanControlMeshJob+CPU_RATE
$cpu.flags = [LeanControlMeshJob]::CpuEnable -bor [LeanControlMeshJob]::CpuHardCap
$cpu.rate = [uint32]($CpuPct * 100)
Set-Struct $job ([LeanControlMeshJob]::Cpu) $cpu

$env:LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$stdout = Join-Path $RunDir "trainer.stdout.log"
$stderr = Join-Path $RunDir "trainer.stderr.log"
$arguments = @(
  "-m", "agent.lean_control_mesh_v1", "train",
  "--registration-dir", $RegistrationDir,
  "--output-dir", $RunDir,
  "--steps", "$Steps",
  "--ram-epochs", "$RamEpochs",
  "--seed", "$Seed",
  "--ram-cap-mb", "$RamMb",
  "--io-cap-mb-s", "$IoMbS"
)
$manifest = @{
  schema = "hermes.lean_control_mesh_wrapper_manifest.v1"
  training_task_id = $TrainingTaskId
  run_id = $RunId
  pid = 0
  registration_dir = $RegistrationDir
  caps = @{
    ram_mb = $RamMb
    cpu_pct = $CpuPct
    io_mb_s = $IoMbS
    wall_seconds = $WallSeconds
  }
  chunk_strategy = "sparse RAM updates plus 16-row CPU TRM minibatches"
  checkpoint_cadence = "final artifact only; bounded run under 15 minutes"
  started_at = (Get-Date).ToUniversalTime().ToString("o")
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunDir "wrapper_manifest.json") -Encoding UTF8

$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
if (-not [LeanControlMeshJob]::AssignProcessToJobObject($job, $process.Handle)) {
  Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  throw "AssignProcessToJobObject failed"
}
$manifest.pid = $process.Id
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunDir "wrapper_manifest.json") -Encoding UTF8

$timedOut = $false
try {
  if (-not $process.WaitForExit($WallSeconds * 1000)) {
    $timedOut = $true
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
} finally {
  & (Join-Path $PSScriptRoot "post_run_int3_cleanup.ps1") -RunId $RunId -OwnedPid $process.Id -SummaryPath (Join-Path $RunDir "cleanup_summary.json") | Out-Null
}

$trainingReceiptPath = Join-Path $RunDir "training_receipt.json"
$cleanupPath = Join-Path $RunDir "cleanup_summary.json"
$trainingReceipt = if (Test-Path -LiteralPath $trainingReceiptPath) {
  Get-Content -Raw -LiteralPath $trainingReceiptPath | ConvertFrom-Json
} else {
  $null
}
$cleanup = if (Test-Path -LiteralPath $cleanupPath) {
  Get-Content -Raw -LiteralPath $cleanupPath | ConvertFrom-Json
} else {
  $null
}
$cleanupPassed = $null -ne $cleanup -and $cleanup.cleanup_passed -eq $true
$completed = (
  -not $timedOut -and
  $process.ExitCode -eq 0 -and
  $null -ne $trainingReceipt -and
  $trainingReceipt.status -eq "completed" -and
  $cleanupPassed
)
$publishedDir = ""
if ($completed) {
  $publishedDir = Join-Path $env:USERPROFILE ".hermes-lite\models\$PublishedModelName"
  New-Item -ItemType Directory -Force -Path $publishedDir | Out-Null
  Copy-Item -LiteralPath (Join-Path $RunDir "ram_policy.json") -Destination (Join-Path $publishedDir "ram_policy.json") -Force
  Copy-Item -LiteralPath (Join-Path $RunDir "trm_router.pt") -Destination (Join-Path $publishedDir "trm_router.pt") -Force
  Copy-Item -LiteralPath $trainingReceiptPath -Destination (Join-Path $publishedDir "training_receipt.json") -Force
}
$summary = @{
  schema = "hermes.lean_control_mesh_wrapper_summary.v1"
  training_task_id = $TrainingTaskId
  run_id = $RunId
  run_dir = $RunDir
  published_dir = $publishedDir
  status = if ($completed) { "completed" } elseif ($timedOut) { "aborted" } else { "failed" }
  abort_reason = if ($timedOut) { "wall_clock_cap" } elseif (-not $cleanupPassed) { "cleanup_failed" } else { "" }
  exit_code = $process.ExitCode
  cleanup_passed = $cleanupPassed
  completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RunDir "wrapper_summary.json") -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
if (-not $completed) {
  throw "Control-mesh trainer did not produce a publishable run; inspect $RunDir"
}
