"""
Comprehensive VS Code Python configuration checker and fixer.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

def check_python_executable():
    """Check if Python executable is accessible."""
    python_path = r"C:\Users\sanyam jain\AppData\Local\Programs\Python\Python313\python.exe"
    
    print("🔍 Checking Python executable...")
    if Path(python_path).exists():
        print(f"✅ Python executable found: {python_path}")
        
        # Test import
        try:
            result = subprocess.run([python_path, "-c", "import pandas; print('Pandas OK')"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Python environment working correctly")
                return True
            else:
                print(f"❌ Python environment error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error testing Python: {e}")
            return False
    else:
        print(f"❌ Python executable not found: {python_path}")
        return False

def check_vscode_settings():
    """Check VS Code settings configuration."""
    settings_path = Path(".vscode/settings.json")
    
    print("\n🔍 Checking VS Code settings...")
    
    if not settings_path.exists():
        print("❌ VS Code settings.json not found")
        return False
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        required_settings = [
            "python.defaultInterpreterPath",
            "python.languageServer",
            "python.analysis.extraPaths"
        ]
        
        for setting in required_settings:
            if setting in settings:
                print(f"✅ {setting}: {settings[setting]}")
            else:
                print(f"❌ Missing setting: {setting}")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in settings.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading settings: {e}")
        return False

def check_workspace_file():
    """Check workspace file configuration."""
    workspace_path = Path("swiggy_project.code-workspace")
    
    print("\n🔍 Checking workspace file...")
    
    if not workspace_path.exists():
        print("⚠️  Workspace file not found (optional)")
        return True
    
    try:
        with open(workspace_path, 'r') as f:
            workspace = json.load(f)
        
        if "settings" in workspace and "python.defaultInterpreterPath" in workspace["settings"]:
            print("✅ Workspace file configured correctly")
            return True
        else:
            print("⚠️  Workspace file missing Python settings")
            return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in workspace file: {e}")
        return False

def check_project_structure():
    """Check project structure."""
    print("\n🔍 Checking project structure...")
    
    required_dirs = ["src", "src/data", "src/models", "src/api", "src/dashboard", "src/utils"]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing: {dir_path}/")
            all_good = False
    
    return all_good

def provide_fix_instructions():
    """Provide step-by-step fix instructions."""
    print("\n" + "="*60)
    print("🛠️  FIX INSTRUCTIONS FOR VS CODE IMPORT ISSUES")
    print("="*60)
    
    print("\n📋 Step 1: Restart VS Code completely")
    print("   - Close all VS Code windows")
    print("   - Run: refresh_vscode.bat (double-click the file)")
    
    print("\n📋 Step 2: In VS Code, set Python interpreter")
    print("   - Press Ctrl+Shift+P")
    print("   - Type: Python: Select Interpreter")
    print("   - Choose: C:\\Users\\sanyam jain\\AppData\\Local\\Programs\\Python\\Python313\\python.exe")
    
    print("\n📋 Step 3: Reload VS Code window")
    print("   - Press Ctrl+Shift+P")
    print("   - Type: Developer: Reload Window")
    
    print("\n📋 Step 4: Clear Python cache (if still not working)")
    print("   - Press Ctrl+Shift+P")
    print("   - Type: Python: Clear Cache and Reload Window")
    
    print("\n📋 Step 5: Restart Python Language Server")
    print("   - Press Ctrl+Shift+P")
    print("   - Type: Python: Restart Language Server")
    
    print("\n🎯 Alternative: Open workspace file directly")
    print("   - Double-click: swiggy_project.code-workspace")
    print("   - This will open VS Code with proper configuration")

def main():
    """Run all checks and provide guidance."""
    print("🔧 VS Code Python Configuration Checker")
    print("="*50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    checks = [
        ("Python Executable", check_python_executable),
        ("VS Code Settings", check_vscode_settings),
        ("Workspace File", check_workspace_file),
        ("Project Structure", check_project_structure)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All configuration checks PASSED!")
        print("If you're still seeing import errors in VS Code, follow the fix instructions below.")
    else:
        print("❌ Some configuration issues found!")
        print("Please follow the fix instructions below.")
    
    provide_fix_instructions()

if __name__ == "__main__":
    main()