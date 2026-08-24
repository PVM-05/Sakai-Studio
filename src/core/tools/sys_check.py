# coding: utf-8
import uuid
import hashlib
import os
import winreg

def get_system_mac():
    try:
        return str(uuid.getnode())
    except:
        return "sakai_studio_fallback_mac"

def make_hash(mac, count):
    salt = "SakaiStudioSecretSalt2026"
    raw = f"{mac}_{count}_{salt}".encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def read_registry_hash():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\SakaiStudio", 0, winreg.KEY_READ)
        val, _ = winreg.QueryValueEx(key, "SysToken")
        winreg.CloseKey(key)
        return val
    except:
        return None

def write_registry_hash(h):
    try:
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\SakaiStudio")
        winreg.SetValueEx(key, "SysToken", 0, winreg.REG_SZ, h)
        winreg.CloseKey(key)
    except:
        pass

def read_file_hash():
    try:
        path = os.path.join(os.environ.get('APPDATA', ''), 'SakaiStudio', 'win_sys.dat')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except:
        pass
    return None

def write_file_hash(h):
    try:
        dir_path = os.path.join(os.environ.get('APPDATA', ''), 'SakaiStudio')
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, 'win_sys.dat')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(h)
    except:
        pass

def get_count():
    mac = get_system_mac()
    reg_hash = read_registry_hash()
    file_hash = read_file_hash()
    
    reg_count = -1
    file_count = -1
    
    if reg_hash:
        for i in range(11):
            if make_hash(mac, i) == reg_hash:
                reg_count = i
                break
        if reg_count == -1:
            reg_count = 999
            
    if file_hash:
        for i in range(11):
            if make_hash(mac, i) == file_hash:
                file_count = i
                break
        if file_count == -1:
            file_count = 999
            
    if reg_count == -1 and file_count == -1:
        write_count(0)
        return 0
        
    if reg_count == -1:
        write_registry_hash(make_hash(mac, file_count))
        return file_count
    if file_count == -1:
        write_file_hash(make_hash(mac, reg_count))
        return reg_count
        
    max_count = max(reg_count, file_count)
    if reg_count != file_count:
        write_count(max_count)
    return max_count

def write_count(c):
    mac = get_system_mac()
    h = make_hash(mac, c)
    write_registry_hash(h)
    write_file_hash(h)

def check_limit():
    # Return True if limit reached, False if not
    return get_count() >= 3

def increment_usage_count():
    current = get_count()
    if current < 10:
        write_count(current + 1)
