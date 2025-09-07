@echo off
echo 🔄 Refreshing VS Code Python Configuration...

echo 1. Verifying Python environment...
"C:\Users\sanyam jain\AppData\Local\Programs\Python\Python313\python.exe" -c "import pandas; print('✅ Pandas available')"

echo 2. Opening VS Code with workspace...
if exist "swiggy_project.code-workspace" (
    start "" "code" "swiggy_project.code-workspace"
    echo ✅ VS Code opened with workspace configuration
) else (
    start "" "code" "."
    echo ✅ VS Code opened with project folder
)

echo.
echo 🎯 Configuration refresh complete!
echo 📋 Next steps in VS Code:
echo    1. Press Ctrl+Shift+P
echo    2. Type "Python: Select Interpreter"
echo    3. Choose: C:\Users\sanyam jain\AppData\Local\Programs\Python\Python313\python.exe
echo    4. Press Ctrl+Shift+P again
echo    5. Type "Developer: Reload Window"
echo.
pause