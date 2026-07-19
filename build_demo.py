# coding: utf-8
import os
import sys
import shutil
import subprocess

def patch_file(filepath, target, replacement):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if target not in content:
        print(f"[WARNING] Target string not found in {filepath}!")
        return False
    content = content.replace(target, replacement)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return True

def main():
    print("=== SAKAI STUDIO DEMO BUILDER ===")
    
    # 1. Back up original files
    backups = {
        'ui/home_interface.py': 'ui/home_interface.py.bak',
        'backend/config.py': 'backend/config.py.bak',
        'SakaiStudio.spec': 'SakaiStudio.spec.bak'
    }
    
    print("[1/4] Creating backups of original files...")
    for orig, bak in backups.items():
        if os.path.exists(orig):
            shutil.copy2(orig, bak)
        else:
            print(f"[ERROR] Source file {orig} not found! Build aborted.")
            return

    try:
        print("[2/4] Applying demo modifications...")
        
        # Patch backend/config.py
        target_config_urls = """VERSION = "3.18.0"
PROJECT_HOME_URL = "https://github.com/PVM-05/Sakai-Studio"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/PVM-05/Sakai-Studio/releases/latest",
    "https://accelerate.xdow.net/api/repos/PVM-05/Sakai-Studio/releases/latest",
]"""
        replacement_config_urls = """VERSION = "3.18.0"
PROJECT_HOME_URL = ""
PROJECT_ISSUES_URL = ""
PROJECT_RELEASES_URL = ""
PROJECT_UPDATE_URLS = []"""
        patch_file('backend/config.py', target_config_urls, replacement_config_urls)
        
        # Patch checkUpdateOnStartup in backend/config.py
        target_update_startup = 'checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", True, BoolValidator())'
        replacement_update_startup = 'checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", False, BoolValidator())'
        patch_file('backend/config.py', target_update_startup, replacement_update_startup)

        # Patch ui/home_interface.py - Signals
        target_signals = """    # Tín hiệu trả kết quả inpaint preview từ background thread về UI thread
    mask_preview_result_signal = Signal(object, object)  # (result_frame, error_info)"""
        replacement_signals = """    # Tín hiệu trả kết quả inpaint preview từ background thread về UI thread
    mask_preview_result_signal = Signal(object, object)  # (result_frame, error_info)
    license_expired_signal = Signal()"""
        patch_file('ui/home_interface.py', target_signals, replacement_signals)

        # Patch ui/home_interface.py - Signal Connection
        target_connection = """        # Kết nối signal trả kết quả xem trước (background thread → UI thread)
        self.mask_preview_result_signal.connect(self._on_mask_preview_result)"""
        replacement_connection = """        # Kết nối signal trả kết quả xem trước (background thread → UI thread)
        self.mask_preview_result_signal.connect(self._on_mask_preview_result)
        self.license_expired_signal.connect(self.show_license_expired_dialog)"""
        patch_file('ui/home_interface.py', target_connection, replacement_connection)

        # Patch ui/home_interface.py - run_button_clicked
        target_run_btn = """    def run_button_clicked(self):
        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        try:
            # 获取所有待执行的任务
            pending_tasks = self.task_list_component.get_pending_tasks()
            if not pending_tasks:
                return

            self._stop_event.clear()
            self.toggle_buttons_signal.emit(False)"""
        replacement_run_btn = """    def show_license_expired_dialog(self):
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Phiên bản dùng thử - Sakai Studio")
        msg.setText("Bạn đang sử dụng phiên bản dùng thử (Trial Version) giới hạn 3 lần xuất video.\\n\\nBạn đã hết lượt dùng thử miễn phí. Vui lòng liên hệ Sakai Studio để mua bản quyền đầy đủ không giới hạn.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def run_button_clicked(self):
        from backend.tools.sys_check import check_limit
        if check_limit():
            self.show_license_expired_dialog()
            return

        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        try:
            # 获取所有待执行的任务
            pending_tasks = self.task_list_component.get_pending_tasks()
            if not pending_tasks:
                return

            self._stop_event.clear()
            self.toggle_buttons_signal.emit(False)"""
        patch_file('ui/home_interface.py', target_run_btn, replacement_run_btn)

        # Patch ui/home_interface.py - task limit check
        target_task_limit = """                            if self._stop_event.is_set():
                                break

                            pending_tasks = self.task_list_component.get_pending_tasks()"""
        replacement_task_limit = """                            if self._stop_event.is_set():
                                break

                            # Check usage limit
                            from backend.tools.sys_check import check_limit
                            if check_limit():
                                self.license_expired_signal.emit()
                                break

                            pending_tasks = self.task_list_component.get_pending_tasks()"""
        patch_file('ui/home_interface.py', target_task_limit, replacement_task_limit)

        # Patch ui/home_interface.py - task completion increment
        target_task_comp = """                            # 更新任务状态为已完成
                            task_obj = self.task_list_component.get_task(self.current_processing_task_index)
                            if process.exitcode == 0 and task_obj and task_obj.status == TaskStatus.PROCESSING:
                                self.progress_signal.emit(100, True)
                                # 任务完成, 更新输出路径为只读
                                task_obj.output_path = output_path
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.COMPLETED)
                            else:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)"""
        replacement_task_comp = """                            # 更新任务状态为已完成
                            task_obj = self.task_list_component.get_task(self.current_processing_task_index)
                            if process.exitcode == 0 and task_obj and task_obj.status == TaskStatus.PROCESSING:
                                self.progress_signal.emit(100, True)
                                # 任务完成, 更新输出路径为只读
                                task_obj.output_path = output_path
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.COMPLETED)
                                # Increment usage count for trial limit
                                try:
                                    from backend.tools.sys_check import increment_usage_count
                                    increment_usage_count()
                                except Exception as e:
                                    print("Licensing error:", e)
                            else:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)"""
        patch_file('ui/home_interface.py', target_task_comp, replacement_task_comp)

        # Patch SakaiStudio.spec - Binary Names
        patch_file('SakaiStudio.spec', "name='SakaiStudio',", "name='SakaiStudioDemo',")

        print("[3/4] Running PyInstaller compilation for SakaiStudioDemo...")
        venv_py = os.path.join('venv', 'Scripts', 'python.exe')
        pyinstaller_cmd = [venv_py, '-m', 'PyInstaller', 'SakaiStudio.spec', '--noconfirm']
        
        subprocess.run(pyinstaller_cmd, check=True)
        print("[SUCCESS] SakaiStudioDemo compilation completed successfully!")

    except Exception as e:
        print(f"[ERROR] Build process failed: {e}")
    finally:
        print("[4/4] Restoring original source files...")
        for orig, bak in backups.items():
            if os.path.exists(bak):
                if os.path.exists(orig):
                    os.remove(orig)
                shutil.move(bak, orig)
                print(f"Restored {orig}")

if __name__ == '__main__':
    main()
