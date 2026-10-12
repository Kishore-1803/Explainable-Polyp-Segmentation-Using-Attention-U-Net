"""
Kaggle Setup Helper — Run this FIRST to configure Kaggle API credentials.
Then run cvc_evaluate.py to perform the external test.
"""
import os
import json
import sys
from pathlib import Path

print("="*65)
print("  Kaggle API Setup for CVC-ClinicDB Download")
print("="*65)
print()
print("  You need a Kaggle API token. Here's how to get one:")
print()
print("  STEP 1: Go to https://www.kaggle.com/settings/api")
print("  STEP 2: Click 'Create New API Token'")
print("  STEP 3: A file called 'kaggle.json' will be downloaded")
print("  STEP 4: It looks like this:")
print('           {"username":"your_username","key":"your_api_key"}')
print()

username = input("  Enter your Kaggle username: ").strip()
api_key  = input("  Enter your Kaggle API key:  ").strip()

if not username or not api_key:
    print("[ERROR] Username and API key cannot be empty.")
    sys.exit(1)

# Create ~/.kaggle/kaggle.json
kaggle_dir  = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"
kaggle_dir.mkdir(exist_ok=True)

credentials = {"username": username, "key": api_key}
with open(kaggle_json, "w") as f:
    json.dump(credentials, f)

# Set file permissions (important on Linux/Mac; no-op on Windows)
try:
    kaggle_json.chmod(0o600)
except Exception:
    pass

print(f"\n[OK] Credentials saved to {kaggle_json}")
print("\n  Testing authentication...")

try:
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "list", "--search", "cvcclinicdb"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print("[OK] Kaggle authentication successful!")
        print(f"     {result.stdout.strip()[:200]}")
    else:
        print(f"[WARN] Test returned: {result.stderr.strip()[:300]}")
except Exception as e:
    print(f"[WARN] Could not test: {e}")

print()
print("  Now run: venv\\Scripts\\python.exe experiments\\cvc_external_test\\cvc_evaluate.py")
print()
