import os
import shutil
import re

# Khai báo cấu trúc thư mục
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

DIRS = {
    'core': ['config.py', 'utils', 'tools', 'interface', 'i18n.py', 'language_support.py', 'import_safety.py', 
             'dependency_caps.py', 'update_check.py', 'crash_reporter.py', 'support_bundle.py', 
             'security_checks.py', 'release_verification.py', 'safe_image.py', 'io.py', 'processor.py', 
             'cli.py', 'main.py', 'proxy_workflow.py', 'batch_report.py', 'cache_inventory.py', 'quality.py', 
             'quality_gate.py', 'reference_corpus.py', 'presets.py', 'workers', 'output_quality_preflight.py',
             'mask_free_benchmark.py', 'static_logo_benchmark.py', 'resume_checkpoint.py', '__init__.py', 'a11y.py', 'adapter_manifest.py'],
    'ai_engines': ['models', 'inpaint', 'inpainters', 'inpainters_diffusion.py', 'inpainters_onnx.py', 
                   'inpainter_registry.py', 'ocr_engine.py', 'ocr_vlm.py', 'translator.py', 'detection.py', 
                   'segmentation.py', 'paddle_compat.py', 'onnx_model_info.py', 'onnxruntime_cuda.py', 
                   'tensorrt_compile.py', 'model_downloads.py', 'model_hashes.py', 'remote_model_policy.py', 
                   'whisper_fallback.py', 'voice_separator.py', 'tts_engine.py'],
    'video_ops': ['ffmpeg', 'ffmpeg_profiles.py', 'encoder.py', 'remux.py', 'preprocess.py', 'post_restore.py', 
                  'hdr.py', 'nle_sidecar.py', 'scenedetect', 'tracking.py', 'karaoke_flow.py', 
                  'vapoursynth_bridge.py', 'decode_accel.py']
}

def move_files():
    # 1. Rename root folders
    if os.path.exists('web'):
        os.rename('web', 'web_client')
    if os.path.exists('config'):
        os.rename('config', 'configs_data')
        
    # 2. Create src structure
    for d in ['core', 'ai_engines', 'video_ops', 'desktop', 'web_api']:
        os.makedirs(os.path.join('src', d), exist_ok=True)
        
    # 3. Move root stuff
    if os.path.exists('api'):
        shutil.move('api', 'src/web_api/api')
    if os.path.exists('ui'):
        shutil.move('ui', 'src/desktop/ui')
    if os.path.exists('gui.py'):
        shutil.move('gui.py', 'src/desktop/main.py')
        
    # 4. Split backend
    if os.path.exists('backend'):
        for cat, files in DIRS.items():
            for f in files:
                src = os.path.join('backend', f)
                if os.path.exists(src):
                    shutil.move(src, os.path.join('src', cat, f))
                    print(f"Moved {src} -> src/{cat}/{f}")
        
        # Move anything left in backend to core just in case
        for item in os.listdir('backend'):
            if item != '__pycache__':
                src = os.path.join('backend', item)
                shutil.move(src, os.path.join('src', 'core', item))
                print(f"Moved leftover {src} -> src/core/{item}")
                
        # Clean backend
        shutil.rmtree('backend', ignore_errors=True)

def update_imports():
    # Cập nhật đường dẫn import trong toàn bộ source
    # Mapping cũ -> mới
    import_map = [
        (r'from backend\.', r'from src.core.'),
        (r'import backend\.', r'import src.core.'),
        (r'from ui\.', r'from src.desktop.ui.'),
        (r'import ui\.', r'import src.desktop.ui.'),
        (r'from api\.', r'from src.web_api.api.'),
        (r'import api\.', r'import src.web_api.api.')
    ]
    
    # Do các file backend bị tách thành core, ai_engines, video_ops
    # Ta phải mapping cụ thể hơn cho ai_engines và video_ops
    advanced_map = []
    for f in DIRS['ai_engines']:
        mod = f.replace('.py', '')
        advanced_map.append((rf'from src\.core\.{mod}', rf'from src.ai_engines.{mod}'))
        advanced_map.append((rf'import src\.core\.{mod}', rf'import src.ai_engines.{mod}'))
        
    for f in DIRS['video_ops']:
        mod = f.replace('.py', '')
        advanced_map.append((rf'from src\.core\.{mod}', rf'from src.video_ops.{mod}'))
        advanced_map.append((rf'import src\.core\.{mod}', rf'import src.video_ops.{mod}'))

    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    orig_content = content
                    for old, new in import_map:
                        content = re.sub(old, new, content)
                        
                    for old, new in advanced_map:
                        content = re.sub(old, new, content)
                        
                    if content != orig_content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Updated imports in {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == '__main__':
    print("Moving files...")
    move_files()
    print("Updating imports...")
    update_imports()
    print("Done!")
