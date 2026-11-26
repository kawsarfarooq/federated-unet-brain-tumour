$ErrorActionPreference = 'Stop'
$taskRoot = Get-Item -LiteralPath $PSScriptRoot
while ($null -ne $taskRoot -and -not (Test-Path -LiteralPath (Join-Path $taskRoot.FullName 'work/repro-env/Scripts/python.exe'))) {
    $taskRoot = $taskRoot.Parent
}
if ($null -eq $taskRoot) {
    throw 'Existing workspace environment not found. Follow README.md to create an environment and run the CLI.'
}
$projectPython = Join-Path $taskRoot.FullName 'work/repro-env/Scripts/python.exe'
$projectDataset = Join-Path $taskRoot.FullName 'work/data/Task01_BrainTumour'
$projectCache = Join-Path $taskRoot.FullName 'work/course_cache'
$projectRun = Join-Path $taskRoot.FullName 'outputs/course_project_smoke'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython -m brainseg_fl --dataset $projectDataset --cache $projectCache --run-dir $projectRun --mode smoke --device cpu
    if ($LASTEXITCODE -ne 0) { throw "Smoke run failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
