"""
Kiểm thử tự động cho hệ thống xác thực người dùng Sakai Studio (Đăng ký, Đăng nhập, Quản lý phiên, Đăng xuất).
"""
import sys
import os
import pytest
import secrets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.web_api.api.main import app

client = TestClient(app)


def test_auth_full_cycle():
    # 1. Khởi tạo tài khoản ngẫu nhiên
    rand_suffix = secrets.token_hex(4)
    uname = f"test_{rand_suffix}"
    pword = "MatKhauBaoMat_123"
    email = f"test_{rand_suffix}@sakaistudio.com"
    fullname = "Người Dùng Kiểm Thử Tự Động"

    # 2. Đăng ký tài khoản
    res_reg = client.post("/auth/register", json={
        "username": uname,
        "password": pword,
        "email": email,
        "full_name": fullname
    })
    assert res_reg.status_code == 200
    reg_data = res_reg.json()
    assert reg_data["success"] is True
    assert reg_data["user"]["username"] == uname
    assert "token" in reg_data

    # 3. Đăng ký trùng tên tài khoản (bắt lỗi 400)
    res_dup = client.post("/auth/register", json={
        "username": uname,
        "password": pword
    })
    assert res_dup.status_code == 400

    # 4. Đăng nhập sai mật khẩu (bắt lỗi 401)
    res_fail = client.post("/auth/login", json={
        "identifier": uname,
        "password": "SaiMatKhauKhongDung"
    })
    assert res_fail.status_code == 401

    # 5. Đăng nhập đúng thông tin bằng tên tài khoản
    res_login = client.post("/auth/login", json={
        "identifier": uname,
        "password": pword
    })
    assert res_login.status_code == 200
    token = res_login.json()["token"]
    assert bool(token) is True

    # 6. Đăng nhập đúng thông tin bằng địa chỉ thư điện tử
    res_login_email = client.post("/auth/login", json={
        "identifier": email,
        "password": pword
    })
    assert res_login_email.status_code == 200

    # 7. Kiểm tra thông tin phiên đăng nhập (/auth/me)
    res_me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["user"]["username"] == uname
    assert me_data["user"]["full_name"] == fullname

    # 8. Đăng xuất tài khoản
    res_logout = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert res_logout.status_code == 200

    # 9. Xác nhận phiên làm việc đã bị thu hồi
    res_me_revoked = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me_revoked.status_code == 401
