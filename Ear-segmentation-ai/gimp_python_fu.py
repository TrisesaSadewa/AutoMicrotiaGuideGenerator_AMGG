# manual_gimp_test.py
import os
import subprocess

def manual_gimp_check():
    """Manually check for GIMP in common Microsoft Store locations"""
    
    print("🔍 Manually checking for GIMP...")
    
    # Common Microsoft Store app locations
    search_paths = [
        r"C:\Program Files\WindowsApps",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps"),
        os.path.expandvars(r"%ProgramFiles%\WindowsApps"),
    ]
    
    gimp_found = False
    
    for base_path in search_paths:
        if os.path.exists(base_path):
            print(f"📁 Checking: {base_path}")
            for item in os.listdir(base_path):
                if "GIMP" in item.upper():
                    full_path = os.path.join(base_path, item)
                    print(f"🎯 Found GIMP folder: {full_path}")
                    
                    # Look for executable
                    possible_exes = [
                        os.path.join(full_path, "bin", "gimp-2.10.exe"),
                        os.path.join(full_path, "gimp-2.10.exe"),
                        os.path.join(full_path, "bin", "gimp-console-2.10.exe"),
                    ]
                    
                    for exe_path in possible_exes:
                        if os.path.exists(exe_path):
                            print(f"✅ Found GIMP executable: {exe_path}")
                            gimp_found = True
                            # Test it
                            try:
                                result = subprocess.run([exe_path, "--version"], 
                                                      capture_output=True, text=True, timeout=10)
                                print(f"✅ GIMP works! Version: {result.stdout.strip()}")
                                return exe_path
                            except Exception as e:
                                print(f"❌ Couldn't run GIMP: {e}")
    
    if not gimp_found:
        print("❌ GIMP not found in standard Microsoft Store locations")
        print("💡 Try running GIMP manually first, then check:")
        print("   1. Open Start Menu, type 'GIMP'")
        print("   2. Right-click on GIMP and select 'Open file location'")
        print("   3. This will show you where GIMP is installed")
    
    return None

# Run the manual check
gimp_path = manual_gimp_check()
if gimp_path:
    print(f"🎉 Use this path in your script: {gimp_path}")
else:
    print("😞 GIMP not found automatically. Please find it manually.")