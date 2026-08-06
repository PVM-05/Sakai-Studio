import os

filepath = r'e:\Sakai-Studio\backend\main.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "if hasattr(self, 'callback') and self.callback:" in l:
        if i + 2 < len(lines):
            # next line is self.callback.update_preview(
            if 'self.callback.update_preview(' in lines[i+1]:
                # adjust indentation of i+1 to be 4 spaces more than i
                indent = len(l) - len(l.lstrip())
                new_l1 = ' ' * (indent + 4) + lines[i+1].lstrip()
                lines[i+1] = new_l1
                # next line is the arguments, append them to the previous line!
                arg_line = lines[i+2].lstrip()
                lines[i+1] = lines[i+1].rstrip('\n') + arg_line
                lines[i+2] = ''

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)
