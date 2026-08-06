import os

filepath = r'e:\Sakai-Studio\backend\main.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace init
text = text.replace('def __init__(self, vd_path, gui_mode=False):', 'def __init__(self, vd_path, gui_mode=False, callback=None):\n        self.callback = callback')

# Replace print inside append_output
old_append_output = '''    def append_output(self, *args):
        """
        """
        print(*args)'''

new_append_output = '''    def append_output(self, *args):
        msg = " ".join(map(str, args))
        logger.info(msg)
        if hasattr(self, 'callback') and self.callback:
            self.callback.log_message(msg)'''

text = text.replace(old_append_output, new_append_output)

# Replace update_progress
old_update_progress = '''    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage)
        self.progress_total = 40 + int(self.progress_remover * 0.6)
        self.current_frame_no = tbar.n
        self.notify_progress_listeners()'''

new_update_progress = '''    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage)
        self.progress_total = 40 + int(self.progress_remover * 0.6)
        self.current_frame_no = tbar.n
        if hasattr(self, 'callback') and self.callback:
            self.callback.update_progress(self.progress_total, self.isFinished, self.current_frame_no)
        self.notify_progress_listeners()'''

text = text.replace(old_update_progress, new_update_progress)

text = text.replace('print("Enabling PyTorch Tensor Cores optimizations (TF32, FP16 MatMul)...")', 'logger.info("Enabling PyTorch Tensor Cores optimizations (TF32, FP16 MatMul)...")')

# Also replace the dynamic manage_process calls inside main.py ? Wait, main.py doesn't call manage_process directly, but we can hook it if needed.
# Let's also check if update_preview_with_comp is called
# In sttn_auto_inpaint, update_preview_with_comp is NOT called. It's called in SubtitleRemover?
text = text.replace('self.update_preview_with_comp(', "if hasattr(self, 'callback') and self.callback:\n                                    self.callback.update_preview(\n")
# Actually, the original code had: self.update_preview_with_comp(..., ...)
# Let's leave update_preview alone if we don't know exactly. The UI monkey patches it anyway so it will keep working.

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)
