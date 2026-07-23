param(
  [string]$TrainingTaskId = "hermes-lean-skill-router-v1",
  [int]$Steps = 800,
  [int]$CheckpointSteps = 100,
  [int]$RamMb = 2048,
  [int]$CpuPct = 50,
  [int]$IoMbS = 50,
  [int]$WallSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunId = "$TrainingTaskId-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$RunDir = Join-Path $RepoRoot "experiments\ultra-lean-skills\training\$RunId"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$freeRamMb = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
if ($freeRamMb -lt ($RamMb + 512)) { throw "Router training requires at least $($RamMb + 512) MB free RAM." }

if (-not ("LeanRouterJob" -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LeanRouterJob {
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
  $ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
  try {
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($Value, $ptr, $false)
    if (-not [LeanRouterJob]::SetInformationJobObject($Job, $Type, $ptr, $size)) { throw "SetInformationJobObject failed" }
  } finally { [System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr) }
}

$job = [LeanRouterJob]::CreateJobObject([IntPtr]::Zero, $RunId)
if ($job -eq [IntPtr]::Zero) { throw "CreateJobObject failed" }
$memory = New-Object LeanRouterJob+EXTENDED
$memory.basic.flags = [LeanRouterJob]::ProcessMemory -bor [LeanRouterJob]::JobMemory
$memory.processMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
$memory.jobMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
Set-Struct $job ([LeanRouterJob]::Extended) $memory
$cpu = New-Object LeanRouterJob+CPU_RATE
$cpu.flags = [LeanRouterJob]::CpuEnable -bor [LeanRouterJob]::CpuHardCap
$cpu.rate = [uint32]($CpuPct * 100)
Set-Struct $job ([LeanRouterJob]::Cpu) $cpu

$env:LEAN_ROUTER_CAP_WRAPPER_ACTIVE = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$stdout = Join-Path $RunDir "trainer.stdout.log"
$stderr = Join-Path $RunDir "trainer.stderr.log"
$arguments = @("-m", "agent.lean_router_trm", "--output-dir", $RunDir, "--steps", "$Steps", "--checkpoint-steps", "$CheckpointSteps", "--ram-cap-mb", "$RamMb", "--io-cap-mb-s", "$IoMbS")
$manifest = @{ training_task_id=$TrainingTaskId; run_id=$RunId; pid=0; caps=@{ram_mb=$RamMb;cpu_pct=$CpuPct;io_mb_s=$IoMbS;wall_seconds=$WallSeconds}; checkpoint_steps=$CheckpointSteps; chunk_strategy="five candidates per row; minibatch 16"; started_at=(Get-Date).ToUniversalTime().ToString("o") }
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_manifest.json")

$process = Start-Process -FilePath "python" -ArgumentList $arguments -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
if (-not [LeanRouterJob]::AssignProcessToJobObject($job, $process.Handle)) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue; throw "AssignProcessToJobObject failed" }
$manifest.pid = $process.Id
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_manifest.json")
$timedOut = $false
try {
  if (-not $process.WaitForExit($WallSeconds * 1000)) { $timedOut = $true; Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
} finally {
  & (Join-Path $PSScriptRoot "post_run_int3_cleanup.ps1") -RunId $RunId -OwnedPid $process.Id -SummaryPath (Join-Path $RunDir "cleanup_summary.json") | Out-Null
}
$PublishedCheckpoint = ""
if (-not $timedOut -and $process.ExitCode -eq 0) {
  $ModelDir = Join-Path $env:USERPROFILE ".hermes-lite\models"
  New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
  $PublishedCheckpoint = Join-Path $ModelDir "skill-router.pt"
  Copy-Item -LiteralPath (Join-Path $RunDir "router.pt") -Destination $PublishedCheckpoint -Force
}
$summary = @{ training_task_id=$TrainingTaskId; run_id=$RunId; run_dir=$RunDir; published_checkpoint=$PublishedCheckpoint; status=if($timedOut){"aborted"}elseif($process.ExitCode -eq 0){"completed"}else{"failed"}; abort_reason=if($timedOut){"wall_clock_cap"}else{""}; exit_code=$process.ExitCode; completed_at=(Get-Date).ToUniversalTime().ToString("o") }
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $RunDir "wrapper_summary.json")
$summary | ConvertTo-Json -Depth 4
if (-not $timedOut -and $process.ExitCode -ne 0) { throw "Router trainer failed; inspect $RunDir" }
