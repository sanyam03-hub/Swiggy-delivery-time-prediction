# VS Code Configuration Refresh Script
# Run this in PowerShell to refresh VS Code Python configuration

Write-Host "🔄 Refreshing VS Code Python Configuration..." -ForegroundColor Cyan

# Kill VS Code processes to ensure clean restart
Write-Host "1. Stopping VS Code processes..." -ForegroundColor Yellow
Get-Process "Code" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Clear VS Code workspace cache
Write-Host "2. Clearing VS Code workspace cache..." -ForegroundColor Yellow
$workspaceStorage = "$env:APPDATA\Code\User\workspaceStorage"
if (Test-Path $workspaceStorage) {
    Get-ChildItem $workspaceStorage -Recurse -Filter "*swiggy*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Clear Python extension cache
Write-Host "3. Clearing Python extension cache..." -ForegroundColor Yellow
$pythonCache = "$env:APPDATA\Code\User\globalStorage\ms-python.python"
if (Test-Path $pythonCache) {
    Remove-Item "$pythonCache\*" -Recurse -Force -ErrorAction SilentlyContinue
}

# Verify Python environment
Write-Host "4. Verifying Python environment..." -ForegroundColor Yellow
& "C:\Users\sanyam jain\AppData\Local\Programs\Python\Python313\python.exe" -c "import pandas; print(f'✅ Pandas {pandas.__version__} available')"

# Open VS Code with workspace file
Write-Host "5. Opening VS Code with workspace configuration..." -ForegroundColor Yellow
$workspaceFile = "C:\Users\sanyam jain\Desktop\academic\Projects\swiggy_project\swiggy_project.code-workspace"

if (Test-Path $workspaceFile) {
    Start-Process "code" -ArgumentList $workspaceFile
    Write-Host "✅ VS Code opened with workspace configuration" -ForegroundColor Green
} else {
    Write-Host "❌ Workspace file not found, opening project folder" -ForegroundColor Red
    Start-Process "code" -ArgumentList "C:\Users\sanyam jain\Desktop\academic\Projects\swiggy_project"
}

Write-Host "`n🎯 Configuration refresh complete!" -ForegroundColor Green
Write-Host "📋 Next steps in VS Code:" -ForegroundColor Cyan
Write-Host "   1. Press Ctrl+Shift+P" -ForegroundColor White
Write-Host "   2. Type 'Python: Select Interpreter'" -ForegroundColor White
Write-Host "   3. Choose: C:\Users\sanyam jain\AppData\Local\Programs\Python\Python313\python.exe" -ForegroundColor White
Write-Host "   4. Press Ctrl+Shift+P again" -ForegroundColor White
Write-Host "   5. Type 'Developer: Reload Window'" -ForegroundColor White