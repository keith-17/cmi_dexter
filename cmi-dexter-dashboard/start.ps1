$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$dashboardPython = 'C:\Users\maran\OneDrive\Documents\Git Profile\cmi_dexter\.venv-kaggle\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $dashboardPython)) {
    $dashboardPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $dashboardPython)) {
        py -3 -m venv --system-site-packages (Join-Path $PSScriptRoot '.venv')
    }
}
$packagesPath = Join-Path $PSScriptRoot 'packages-ready'
if (-not (Test-Path -LiteralPath (Join-Path $packagesPath 'streamlit'))) {
    & $dashboardPython -m pip install --no-cache-dir --disable-pip-version-check --no-input --target $packagesPath -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dashboard dependencies could not be installed.' }
}
& $dashboardPython (Join-Path $PSScriptRoot 'run_dashboard.py')

