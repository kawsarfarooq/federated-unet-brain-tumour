param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$Cache = './cache',
    [string]$RunDir = './runs/smoke'
)
$ErrorActionPreference = 'Stop'
python -m brainseg_fl --dataset $Dataset --cache $Cache --run-dir $RunDir --mode smoke --device cpu
if ($LASTEXITCODE -ne 0) { throw "Verification failed with exit code $LASTEXITCODE" }
