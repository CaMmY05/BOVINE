Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  CATTLE BREED DETECTION MVP - Environment Activation" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

& "..\cattle_mvp_env\Scripts\Activate.ps1"

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  Environment activated!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor White
Write-Host "  - Verify setup: " -NoNewline -ForegroundColor White
Write-Host "python scripts/verify_setup.py" -ForegroundColor Yellow
Write-Host "  - Prepare data: " -NoNewline -ForegroundColor White
Write-Host "python scripts/prepare_data.py" -ForegroundColor Yellow
Write-Host "  - Train model:  " -NoNewline -ForegroundColor White
Write-Host "python scripts/train_classifier.py" -ForegroundColor Yellow
Write-Host "  - Launch app:   " -NoNewline -ForegroundColor White
Write-Host "streamlit run app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "For detailed instructions, see SETUP_INSTRUCTIONS.txt" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
