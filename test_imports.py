"""
Test script to verify all project dependencies are importable.
"""
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# VS Code Python Environment Test
# This file tests import resolution in VS Code

import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test project imports
try:
    from src.data.preprocessor import DataPreprocessor
    from src.models.base_model import BaseModel
    from src.utils.config import MODELS_DIR
    from src.utils.logger import setup_logger
    print("✅ All imports successful!")
    print(f"✅ Pandas version: {pd.__version__}")
    print(f"✅ NumPy version: {np.__version__}")
    print(f"✅ Python path includes: {src_path}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current Python path: {sys.path}")

# Test core dependencies
try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__} - OK")
except ImportError as e:
    print(f"❌ pandas - FAILED: {e}")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__} - OK")
except ImportError as e:
    print(f"❌ numpy - FAILED: {e}")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__} - OK")
except ImportError as e:
    print(f"❌ scikit-learn - FAILED: {e}")

try:
    import streamlit as st
    print(f"✅ streamlit {st.__version__} - OK")
except ImportError as e:
    print(f"❌ streamlit - FAILED: {e}")

try:
    import flask
    # Use importlib.metadata to avoid deprecation warning
    try:
        from importlib import metadata
        flask_version = metadata.version('flask')
    except Exception:
        flask_version = "Unknown"
    print(f"✅ flask {flask_version} - OK")
except ImportError as e:
    print(f"❌ flask - FAILED: {e}")

try:
    import fastapi
    print(f"✅ fastapi {fastapi.__version__} - OK")
except ImportError as e:
    print(f"❌ fastapi - FAILED: {e}")

print("\n🎯 If all packages show OK above, then your Python environment is correctly configured!")
print("The issue is likely with your IDE's Python interpreter setting.")