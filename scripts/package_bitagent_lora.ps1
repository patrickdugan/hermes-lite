param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("intent_planner", "utxo_tradelayer_specialist", "risk_approval_guard", "recovery_operator")]
    [string]$Role,
    [Parameter(Mandatory = $true)]
    [string]$AdapterDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$LlamaDir = $env:HERMES_LLAMA_CPP_DIR,
    [string]$PythonExe = "python",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$AdapterDir = (Resolve-Path -LiteralPath $AdapterDir).Path
$manifestPath = Join-Path $AdapterDir "bitagent_adapter_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Missing BitAgent adapter manifest: $manifestPath"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.schema -ne "hermes.bitagent_role_adapter_manifest.v1") {
    throw "Unsupported adapter manifest schema."
}
if ($manifest.role -ne $Role) {
    throw "Adapter role mismatch: manifest=$($manifest.role), requested=$Role"
}
if ($manifest.runtime_status -ne "candidate_pending_gguf_conversion_and_heldout_promotion") {
    throw "Adapter is not in the expected pre-conversion state."
}
if ([string]::IsNullOrWhiteSpace($LlamaDir)) {
    throw "Set HERMES_LLAMA_CPP_DIR or pass -LlamaDir."
}
$converter = Join-Path $LlamaDir "convert_lora_to_gguf.py"
if (-not (Test-Path -LiteralPath $converter)) {
    throw "llama.cpp LoRA converter not found: $converter"
}
$outputDirectory = Split-Path -Parent $OutputPath
if ([string]::IsNullOrWhiteSpace($outputDirectory)) {
    $outputDirectory = (Get-Location).Path
    $OutputPath = Join-Path $outputDirectory $OutputPath
}
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$arguments = @(
    $converter,
    "--base-model-id", "prism-ml/Bonsai-8B-unpacked",
    "--outtype", "f16",
    "--outfile", $OutputPath,
    $AdapterDir
)
if ($DryRun) {
    $arguments = @($converter, "--dry-run") + $arguments[1..($arguments.Count - 1)]
}
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "LoRA GGUF conversion failed with exit code $LASTEXITCODE"
}
if ($DryRun) {
    [pscustomobject]@{
        schema = "hermes.bitagent_lora_package_receipt.v1"
        status = "validated"
        role = $Role
        output_path = $OutputPath
        wrote_artifact = $false
    } | ConvertTo-Json -Depth 4
    exit 0
}
if (-not (Test-Path -LiteralPath $OutputPath)) {
    throw "Converter reported success but did not create $OutputPath"
}
$receipt = [ordered]@{
    schema = "hermes.bitagent_lora_package_receipt.v1"
    status = "packaged_pending_compatibility_test"
    role = $Role
    source_adapter_manifest = $manifestPath
    source_adapter_tree_sha256 = $manifest.adapter_tree_sha256
    base_model_revision = $manifest.base_model.revision
    output_path = (Resolve-Path -LiteralPath $OutputPath).Path
    output_sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
}
$receiptPath = "$OutputPath.receipt.json"
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
$receipt | ConvertTo-Json -Depth 6
