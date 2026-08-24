import os
import sys
import re

def fix_virtual_env(venv_dir=None, force=False):
    # Determine the project directory (parent of this script)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if venv_dir is None:
        venv_dir = os.path.join(project_dir, 'venv')
        
    if not os.path.isdir(venv_dir):
        print(f"Error: Virtual environment directory not found at '{venv_dir}'")
        return False

    print(f"Current project directory: {project_dir}")
    print(f"Virtual environment directory: {venv_dir}")

    # 1. Determine the OLD path from the existing pyvenv.cfg or activate.bat
    old_path = None
    pyvenv_cfg_path = os.path.join(venv_dir, 'pyvenv.cfg')
    if os.path.isfile(pyvenv_cfg_path):
        with open(pyvenv_cfg_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('command ='):
                    # Command usually looks like: command = python -m venv E:\old-path\venv
                    match = re.search(r'-m venv\s+(.*)', line)
                    if match:
                        potential_path = match.group(1).strip()
                        # Extract directory parent of 'venv'
                        if potential_path.lower().endswith(r'\venv'):
                            old_path = potential_path[:-5]
                        elif potential_path.lower().endswith('/venv'):
                            old_path = potential_path[:-5]
                        else:
                            old_path = os.path.dirname(potential_path)
                        break

    if not old_path:
        # Try from activate.bat
        activate_bat_path = os.path.join(venv_dir, 'Scripts', 'activate.bat')
        if os.path.isfile(activate_bat_path):
            with open(activate_bat_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                match = re.search(r'set\s+"VIRTUAL_ENV=(.*?)"', content, re.IGNORECASE)
                if match:
                    potential_path = match.group(1).strip()
                    if potential_path.lower().endswith(r'\venv'):
                        old_path = potential_path[:-5]
                    elif potential_path.lower().endswith('/venv'):
                        old_path = potential_path[:-5]
                    else:
                        old_path = os.path.dirname(potential_path)

    if not old_path:
        print("Error: Could not automatically detect the old virtual environment path.")
        print("Please run this script with python and pass the old path as an argument if needed.")
        return False

    # Normalize paths (replace backward slashes with forward/backward appropriately)
    old_path = os.path.abspath(old_path)
    new_path = os.path.abspath(project_dir)

    print(f"Detected OLD path: {old_path}")
    print(f"Detected NEW path: {new_path}")

    if old_path.lower() == new_path.lower() and not force:
        print("Success: Virtual environment path is already correct!")
        return True

    # Prepare old and new path patterns for text and binary replacements
    old_path_esc = re.escape(old_path)
    old_path_forward = old_path.replace('\\', '/')
    new_path_forward = new_path.replace('\\', '/')

    print("\n--- Repairing configuration files ---")
    
    # Files to update
    files_to_update = [
        ('pyvenv.cfg', pyvenv_cfg_path),
        ('activate', os.path.join(venv_dir, 'Scripts', 'activate')),
        ('activate.bat', os.path.join(venv_dir, 'Scripts', 'activate.bat')),
        ('Activate.ps1', os.path.join(venv_dir, 'Scripts', 'Activate.ps1')),
    ]

    for name, path in files_to_update:
        if os.path.isfile(path):
            try:
                # Read content
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Replace paths
                new_content = content
                # Replace backslash paths
                new_content = new_content.replace(old_path, new_path)
                # Replace forward slash paths
                new_content = new_content.replace(old_path_forward, new_path_forward)
                
                pass

                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {name}")
                else:
                    print(f"No changes needed: {name}")
            except Exception as e:
                print(f"Failed to update {name}: {e}")

    print("\n--- Repairing Scripts folder (executables & batch scripts) ---")
    scripts_dir = os.path.join(venv_dir, 'Scripts')
    if os.path.isdir(scripts_dir):
        for filename in os.listdir(scripts_dir):
            filepath = os.path.join(scripts_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            
            # 1. Text-based script files (.bat, .py, etc.)
            if ext in ['.bat', '.py', '.txt', '.json', '']:
                if filename in ['activate', 'activate.bat', 'Activate.ps1']:
                    continue
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    new_content = content
                    new_content = new_content.replace(old_path, new_path)
                    new_content = new_content.replace(old_path_forward, new_path_forward)
                    
                    if new_content != content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated script: {filename}")
                except Exception as e:
                    print(f"Failed to update script {filename}: {e}")
                    
            # 2. Binary executable launcher files (.exe)
            elif ext == '.exe':
                try:
                    with open(filepath, 'rb') as f:
                        data = f.read()
                    
                    # Search for shebang sequence: b'#!' + old python path + b'\n'
                    # Or simply search for the old python path bytes
                    old_py_path = os.path.join(old_path, 'venv', 'Scripts', 'python.exe')
                    old_py_path_bytes = old_py_path.encode('utf-8')
                    new_py_path = os.path.join(new_path, 'venv', 'Scripts', 'python.exe')
                    new_py_path_bytes = new_py_path.encode('utf-8')
                    
                    if old_py_path_bytes in data:
                        new_data = data.replace(old_py_path_bytes, new_py_path_bytes)
                        with open(filepath, 'wb') as f:
                            f.write(new_data)
                        print(f"Updated binary launcher: {filename}")
                    else:
                        # Try case-insensitive or general path check
                        old_path_bytes = old_path.encode('utf-8')
                        new_path_bytes = new_path.encode('utf-8')
                        if old_path_bytes in data:
                            new_data = data.replace(old_path_bytes, new_path_bytes)
                            with open(filepath, 'wb') as f:
                                f.write(new_data)
                            print(f"Updated binary launcher (general): {filename}")
                except Exception as e:
                    print(f"Failed to update binary launcher {filename}: {e}")

    print("\n--- Repairing pywin32 system DLLs path ---")
    try:
        # Run pywin32 post-installation script to register and copy DLLs to system32
        pywin_post = os.path.join(venv_dir, 'Scripts', 'pywin32_postinstall.py')
        if os.path.isfile(pywin_post):
            # We can run it using the venv python
            venv_python = os.path.join(venv_dir, 'Scripts', 'python.exe')
            import subprocess
            res = subprocess.run([venv_python, pywin_post, '-install'], capture_output=True, text=True)
            print("pywin32_postinstall.py output:")
            print(res.stdout)
            if res.stderr:
                print("pywin32_postinstall.py error:")
                print(res.stderr)
    except Exception as e:
        print(f"Failed to run pywin32 post-install: {e}")

    print("\nVirtual environment repair finished successfully!")
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Repair Python venv paths after renaming the project folder.")
    parser.add_argument('--venv', type=str, default=None, help="Path to venv directory")
    parser.add_argument('--force', action='store_true', help="Force repair even if path matches")
    args = parser.parse_args()
    
    fix_virtual_env(venv_dir=args.venv, force=args.force)
