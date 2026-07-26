#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_capped_bitagent_qlora.sh \
    --training-task-id ID --role ROLE --config PATH --examples PATH \
    --base-model PATH_OR_PINNED_REPO --output-dir LINUX_EXT4_PATH \
    --ram-mb N --cpu-pct N --io-mb-s N --io-device /dev/DEVICE \
    --wall-seconds N --min-free-vram-mb N \
    --checkpoint-steps N|--checkpoint-seconds N \
    --chunk-strategy TEXT [--python PYTHON] [--allow-download] [--validate-only]

The output directory and any local training base must be on a Linux block
filesystem controlled by --io-device. Validation-only never loads weights.
EOF
}

training_task_id=""
role=""
config=""
examples=""
base_model=""
output_dir=""
ram_mb=""
cpu_pct=""
io_mb_s=""
io_device=""
wall_seconds=""
min_free_vram_mb=""
checkpoint_steps="0"
checkpoint_seconds="0"
chunk_strategy=""
python_bin="python3"
allow_download="0"
validate_only="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --training-task-id) training_task_id="$2"; shift 2 ;;
    --role) role="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --examples) examples="$2"; shift 2 ;;
    --base-model) base_model="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --ram-mb) ram_mb="$2"; shift 2 ;;
    --cpu-pct) cpu_pct="$2"; shift 2 ;;
    --io-mb-s) io_mb_s="$2"; shift 2 ;;
    --io-device) io_device="$2"; shift 2 ;;
    --wall-seconds) wall_seconds="$2"; shift 2 ;;
    --min-free-vram-mb) min_free_vram_mb="$2"; shift 2 ;;
    --checkpoint-steps) checkpoint_steps="$2"; shift 2 ;;
    --checkpoint-seconds) checkpoint_seconds="$2"; shift 2 ;;
    --chunk-strategy) chunk_strategy="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --allow-download) allow_download="1"; shift ;;
    --validate-only) validate_only="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in training_task_id role config examples base_model output_dir ram_mb cpu_pct io_mb_s io_device wall_seconds min_free_vram_mb chunk_strategy; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required --${value_name//_/-}" >&2
    exit 2
  fi
done
if [[ "$checkpoint_steps" -le 0 && "$checkpoint_seconds" -le 0 ]]; then
  echo "A positive checkpoint interval is required." >&2
  exit 2
fi
if [[ "$io_device" != /dev/* || ! -b "$io_device" ]]; then
  echo "--io-device must name an existing block device." >&2
  exit 2
fi
if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run is required; uncapped training is refused." >&2
  exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
output_fstype="$(findmnt -no FSTYPE --target "$output_dir" || true)"
output_source="$(findmnt -no SOURCE --target "$output_dir" || true)"
if [[ "$validate_only" != "1" && "$output_fstype" =~ ^(9p|drvfs|fuseblk)$ ]]; then
  echo "Training output must be on WSL ext4 or another block filesystem so the I/O cap is enforceable." >&2
  exit 2
fi
if [[ "$validate_only" != "1" && "$output_source" != "$io_device" ]]; then
  echo "Training output is on $output_source, but --io-device is $io_device." >&2
  exit 2
fi
if [[ "$validate_only" != "1" && -e "$base_model" ]]; then
  base_fstype="$(findmnt -no FSTYPE --target "$base_model" || true)"
  base_source="$(findmnt -no SOURCE --target "$base_model" || true)"
  if [[ "$base_fstype" =~ ^(9p|drvfs|fuseblk)$ ]]; then
    echo "Local training base must be on a capped Linux block filesystem." >&2
    exit 2
  fi
  if [[ "$base_source" != "$io_device" ]]; then
    echo "Local training base is on $base_source, but --io-device is $io_device." >&2
    exit 2
  fi
fi

version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11+ is required; found $version." >&2
  exit 2
fi

common_args=(
  -m agent.bitagent_qlora_train
  --config "$config"
  --examples "$examples"
  --output-dir "$output_dir"
  --training-task-id "$training_task_id"
  --role "$role"
  --base-model "$base_model"
  --ram-cap-mb "$ram_mb"
  --cpu-cap-pct "$cpu_pct"
  --io-cap-mb-s "$io_mb_s"
  --wall-seconds "$wall_seconds"
  --min-free-vram-mb "$min_free_vram_mb"
  --checkpoint-steps "$checkpoint_steps"
  --checkpoint-seconds "$checkpoint_seconds"
  --chunk-strategy "$chunk_strategy"
)
if [[ "$allow_download" == "1" ]]; then common_args+=(--allow-download); fi
if [[ "$validate_only" == "1" ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" "${common_args[@]}" --validate-only
  exit $?
fi

if ! systemd-run --user --wait --quiet \
  --property "MemoryMax=64M" \
  --property "CPUQuota=10%" \
  /bin/true >/dev/null 2>&1; then
  echo "The WSL systemd user bus is unavailable; refusing uncapped training." >&2
  echo "Start or repair the WSL user systemd session, then rerun validation." >&2
  exit 2
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "WSL CUDA is unavailable; refusing the registered 8B QLoRA run." >&2
  exit 2
fi
"$python_bin" -c 'import accelerate, bitsandbytes, peft, torch, transformers' || {
  echo "Adapter dependencies are missing. Install with: pip install -e '.[adapters]'" >&2
  exit 2
}

logical_cpus="$(nproc)"
cpu_quota="$((logical_cpus * cpu_pct))%"
memory_high_mb="$((ram_mb * 9 / 10))"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_id="$(printf '%s' "$training_task_id-$role-$run_stamp" | tr -cs 'A-Za-z0-9_.-' '-')"
unit="bitagent-qlora-${safe_id:0:180}"
stdout_log="$output_dir/wrapper.stdout.log"

set +e
systemd-run --user --wait --pipe --quiet \
  --unit "$unit" \
  --working-directory "$repo_root" \
  --property "MemoryHigh=${memory_high_mb}M" \
  --property "MemoryMax=${ram_mb}M" \
  --property "MemorySwapMax=0" \
  --property "CPUAccounting=yes" \
  --property "CPUQuota=$cpu_quota" \
  --property "IOAccounting=yes" \
  --property "IOReadBandwidthMax=$io_device ${io_mb_s}M" \
  --property "IOWriteBandwidthMax=$io_device ${io_mb_s}M" \
  --property "RuntimeMaxSec=$wall_seconds" \
  --property "TasksMax=128" \
  --property "OOMPolicy=stop" \
  --setenv "BITAGENT_QLORA_CAP_WRAPPER_ACTIVE=1" \
  --setenv "PYTHONPATH=$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  --setenv "HF_HOME=$output_dir/hf-cache" \
  --setenv "TOKENIZERS_PARALLELISM=false" \
  "$python_bin" "${common_args[@]}" 2>&1 | tee "$stdout_log"
exit_code="${PIPESTATUS[0]}"
set -e

systemctl --user show "$unit" \
  --property Id,Result,ExecMainStatus,MemoryPeak,CPUUsageNSec,IOReadBytes,IOWriteBytes \
  > "$output_dir/cgroup_receipt.txt" 2>/dev/null || true
systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
exit "$exit_code"
