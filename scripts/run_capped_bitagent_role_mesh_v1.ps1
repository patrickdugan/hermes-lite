param(
  [string]$RegistrationDir = "evals\registered\bitagent_hermes_cross_domain_role_mesh_v1",
  [string]$RegistrationId = "bf3973f14198efb0778821b38e1ba94414a03ac2d2e3f77dd3eece7f702b7692",
  [Parameter(Mandatory = $true)]
  [string]$EvaluatorAddendumId,
  [string]$OutputDir = "experiments\bitagent-cross-domain-role-mesh-v1",
  [int]$RamMb = 2048,
  [int]$CpuPct = 50,
  [int]$IoMbS = 50,
  [int]$WallSeconds = 300
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction Stop).Source
$RegistrationDir = if ([IO.Path]::IsPathRooted($RegistrationDir)) {
  [IO.Path]::GetFullPath($RegistrationDir)
} else {
  [IO.Path]::GetFullPath((Join-Path $RepoRoot $RegistrationDir))
}
$OutputDir = if ([IO.Path]::IsPathRooted($OutputDir)) {
  [IO.Path]::GetFullPath($OutputDir)
} else {
  [IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
}
$WrapperDir = Join-Path $OutputDir "wrapper"
New-Item -ItemType Directory -Force -Path $WrapperDir | Out-Null

if (($RamMb -ne 2048) -or ($CpuPct -ne 50) -or ($IoMbS -ne 50) -or ($WallSeconds -ne 300)) {
  throw "Resource arguments differ from the frozen evaluator addendum."
}

if (-not ("BitAgentRoleMeshJob" -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class BitAgentRoleMeshJob {
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

function Set-JobStruct($Job, [int]$Type, $Value) {
  $size = [System.Runtime.InteropServices.Marshal]::SizeOf($Value)
  $pointer = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
  try {
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($Value, $pointer, $false)
    if (-not [BitAgentRoleMeshJob]::SetInformationJobObject($Job, $Type, $pointer, $size)) {
      throw "SetInformationJobObject failed"
    }
  } finally {
    [System.Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
  }
}

$runId = "bitagent-role-mesh-v1-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$job = [BitAgentRoleMeshJob]::CreateJobObject([IntPtr]::Zero, $runId)
if ($job -eq [IntPtr]::Zero) {
  throw "CreateJobObject failed"
}
$memory = New-Object BitAgentRoleMeshJob+EXTENDED
$memory.basic.flags = [BitAgentRoleMeshJob]::ProcessMemory -bor [BitAgentRoleMeshJob]::JobMemory
$memory.processMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
$memory.jobMemory = [UIntPtr]([UInt64]$RamMb * 1MB)
Set-JobStruct $job ([BitAgentRoleMeshJob]::Extended) $memory
$cpu = New-Object BitAgentRoleMeshJob+CPU_RATE
$cpu.flags = [BitAgentRoleMeshJob]::CpuEnable -bor [BitAgentRoleMeshJob]::CpuHardCap
$cpu.rate = [uint32]($CpuPct * 100)
Set-JobStruct $job ([BitAgentRoleMeshJob]::Cpu) $cpu

$env:BITAGENT_ROLE_MESH_CAP_WRAPPER_ACTIVE = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$stdout = Join-Path $WrapperDir "evaluator.stdout.log"
$stderr = Join-Path $WrapperDir "evaluator.stderr.log"
$arguments = @(
  "-m", "agent.bitagent_mcp_role_mesh_v1", "evaluate",
  "--registration-dir", $RegistrationDir,
  "--output-dir", $OutputDir,
  "--confirm-registration-id", $RegistrationId,
  "--confirm-addendum-id", $EvaluatorAddendumId,
  "--ram-cap-mb", "$RamMb",
  "--io-cap-mb-s", "$IoMbS",
  "--wall-seconds", "$WallSeconds"
)
$manifest = @{
  schema = "hermes.bitagent_cross_domain_role_mesh_wrapper_manifest.v1"
  run_id = $runId
  registration_id = $RegistrationId
  evaluator_addendum_id = $EvaluatorAddendumId
  evaluator_pid = 0
  caps = @{
    ram_mb = $RamMb
    cpu_pct = $CpuPct
    io_mb_s_telemetry = $IoMbS
    wall_seconds = $WallSeconds
    gpu = "disabled"
  }
  enforcement = @{
    ram = "Windows Job Object hard cap"
    cpu = "Windows Job Object hard cap"
    wall = "wrapper hard timeout"
    io = "evaluator telemetry abort"
  }
  started_at = (Get-Date).ToUniversalTime().ToString("o")
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $WrapperDir "manifest.json") -Encoding UTF8

$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
if (-not [BitAgentRoleMeshJob]::AssignProcessToJobObject($job, $process.Handle)) {
  Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  throw "AssignProcessToJobObject failed"
}
$manifest.evaluator_pid = $process.Id
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $WrapperDir "manifest.json") -Encoding UTF8

$peakRamMb = 0.0
$peakIoMbS = 0.0
$previousIo = 0.0
$previousSample = Get-Date
$timedOut = $false
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
while (-not $process.HasExited) {
  if ($stopwatch.Elapsed.TotalSeconds -gt $WallSeconds) {
    $timedOut = $true
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    break
  }
  $sample = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
  if ($null -ne $sample) {
    $ram = [math]::Round($sample.WorkingSet64 / 1MB, 3)
    $peakRamMb = [math]::Max($peakRamMb, $ram)
    $read = if ($null -ne $sample.PSObject.Properties["IOReadBytes"]) { [double]$sample.IOReadBytes } else { 0.0 }
    $write = if ($null -ne $sample.PSObject.Properties["IOWriteBytes"]) { [double]$sample.IOWriteBytes } else { 0.0 }
    $now = Get-Date
    $seconds = [math]::Max(0.001, ($now - $previousSample).TotalSeconds)
    $rate = [math]::Max(0.0, (($read + $write) - $previousIo) / 1MB / $seconds)
    $peakIoMbS = [math]::Max($peakIoMbS, $rate)
    $previousIo = $read + $write
    $previousSample = $now
  }
  Start-Sleep -Milliseconds 100
  $process.Refresh()
}
$process.WaitForExit()
$stopwatch.Stop()

& (Join-Path $PSScriptRoot "post_run_int3_cleanup.ps1") -RunId $runId -OwnedPid $process.Id -SummaryPath (Join-Path $WrapperDir "cleanup_summary.json") | Out-Null
$cleanup = Get-Content -Raw -LiteralPath (Join-Path $WrapperDir "cleanup_summary.json") | ConvertFrom-Json
$summaryPath = Join-Path $OutputDir "summary.json"
$completed = (
  -not $timedOut -and
  $process.ExitCode -eq 0 -and
  (Test-Path -LiteralPath $summaryPath) -and
  $cleanup.cleanup_passed -eq $true
)
$receipt = @{
  schema = "hermes.bitagent_cross_domain_role_mesh_resource_receipt.v1"
  run_id = $runId
  registration_id = $RegistrationId
  evaluator_addendum_id = $EvaluatorAddendumId
  status = if ($completed) { "completed" } elseif ($timedOut) { "aborted" } else { "failed" }
  exit_code = $process.ExitCode
  caps = $manifest.caps
  enforcement = $manifest.enforcement
  peak_ram_mb = [math]::Round($peakRamMb, 3)
  peak_io_mb_s_sampled = [math]::Round($peakIoMbS, 3)
  elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
  cleanup_passed = $cleanup.cleanup_passed
  lingering_owned_pids = $cleanup.lingering_owned_pids
  completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $WrapperDir "resource_receipt.json") -Encoding UTF8
$receipt | ConvertTo-Json -Depth 6
if (-not $completed) {
  throw "Cross-domain role-mesh evaluation failed; inspect $WrapperDir"
}
