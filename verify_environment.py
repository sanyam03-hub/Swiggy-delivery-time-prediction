#!/usr/bin/env python3
"""
Python environment verification script.
Checks all required packages and their versions.
"""

def verify_environment():
    """Verify the Python environment and all required packages."""
    print("🔍 Verifying Python Environment...")
    print("=" * 50)
    
    # Check Python version
    import sys
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Python Executable: {sys.executable}")
    print()
    
    # Required packages for the project
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'streamlit',
        'flask',
        'fastapi',
        'joblib',
        'requests'
    ]
    
    print("📦 Package Verification:")
    print("-" * 30)
    
    all_good = True
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
                print(f"✅ {package}: {version}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: NOT FOUND")
            all_good = False
    
    # Optional packages
    optional_packages = ['xgboost', 'lightgbm']
    
    print("\n🔧 Optional ML Packages:")
    print("-" * 30)
    
    for package in optional_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"⚠️  {package}: NOT FOUND (optional)")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 Environment verification PASSED!")
        print("All required packages are properly installed.")
    else:
        print("❌ Environment verification FAILED!")
        print("Some required packages are missing.")
        return False
    
    # Test import of project modules
    print("\n🧪 Testing Project Module Imports:")
    print("-" * 40)
    
    try:
        import os
        import sys
        
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Test key project imports
        test_imports = [
            ('src.data.preprocessor', 'DataPreprocessor'),
            ('src.models.base_model', 'BaseModel'),
            ('src.utils.config', 'MODELS_DIR'),
            ('src.utils.logger', 'setup_logger')
        ]
        
        for module_name, class_name in test_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"✅ {module_name}.{class_name}")
            except ImportError as e:
                print(f"❌ {module_name}.{class_name}: {e}")
                all_good = False
            except AttributeError as e:
                print(f"⚠️  {module_name}.{class_name}: {e}")
        
    except Exception as e:
        print(f"❌ Project module testing failed: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🚀 COMPLETE VERIFICATION PASSED!")
        print("Your environment is ready for the Swiggy ML project!")
    else:
        print("🛠️  Some issues found. Please check the errors above.")
    
    return all_good

if __name__ == "__main__":
    verify_environment()