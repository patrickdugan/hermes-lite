param(
  [string]$RunId,
  [int]$OwnedPid,
  [string]$SummaryPath
)

$ErrorActionPreference = "Continue"
function Memory-Snapshot {
  $os = Get-CimInstance Win32_OperatingSystem
  $perf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
  return @{
    available_mb = [int]($os.FreePhysicalMemory / 1024)
    committed_mb = if ($perf) { [math]::Round($perf.CommittedBytes / 1MB, 2) } else { $null }
    cache_mb = if ($perf) { [math]::Round($perf.CacheBytes / 1MB, 2) } else { $null }
    standby_mb = if ($perf) { [math]::Round(($perf.StandbyCacheReserveBytes + $perf.StandbyCacheNormalPriorityBytes + $perf.StandbyCacheCoreBytes) / 1MB, 2) } else { $null }
    page_reads_per_sec = if ($perf) { [int]$perf.PageReadsPersec } else { $null }
  }
}
function Gpu-Snapshot {
  if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return @{ available = $false } }
  return @{
    available = $true
    gpu = @(& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null)
    compute_apps = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null)
  }
}
$before = Memory-Snapshot
$gpu = Gpu-Snapshot
$ownedStopped = @()
$process = Get-Process -Id $OwnedPid -ErrorAction SilentlyContinue
if ($process) {
  Stop-Process -Id $OwnedPid -Force -ErrorAction SilentlyContinue
  $ownedStopped += $OwnedPid
  Start-Sleep -Milliseconds 500
}
$lingering = [bool](Get-Process -Id $OwnedPid -ErrorAction SilentlyContinue)
$top = @(Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 10 Id,ProcessName,
  @{n="private_mb";e={[math]::Round($_.PrivateMemorySize64 / 1MB, 2)}},
  @{n="working_set_mb";e={[math]::Round($_.WorkingSet64 / 1MB, 2)}})
$summary = @{
  run_id = $RunId
  owned_pids = @($OwnedPid)
  stopped_owned_pids = $ownedStopped
  lingering_owned_pids = if ($lingering) { @($OwnedPid) } else { @() }
  memory_before = $before
  memory_after = Memory-Snapshot
  gpu = $gpu
  top_processes = $top
  cleanup_passed = -not $lingering
  global_cache_purge = "not_requested"
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$summary | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 $SummaryPath
$summary | ConvertTo-Json -Depth 7
