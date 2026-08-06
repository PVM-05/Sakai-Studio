import cv2
import sys
import os
from unittest.mock import MagicMock

# Ensure backend module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import SubtitleRemover

def test():
    video_path = 'C:/Users/ADMIN/Videos/2D/Minecraft/DVD Logo Hits Corner.mp4'
    if not os.path.exists(video_path):
        print(f"Test skipped, video not found: {video_path}")
        return
        
    sub_areas = [(0, 100, 0, 100)]
    
    class MockUpdateGui:
        def update(self, num):
            print(f"Progress: {num}")
            
    class MockSubRemover(SubtitleRemover):
        def update_gui(self):
            return MockUpdateGui()
            
    remover = MockSubRemover(video_path, sub_areas)
    remover.video_cap = cv2.VideoCapture(video_path)
    # mock to limit to 10 frames
    remover.frame_count = min(10, int(remover.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Disable audio extraction/merge for fast test
    remover.is_has_audio = False
    
    try:
        remover.run()
        print("Test passed without crashing!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with error: {e}")

if __name__ == '__main__':
    test()
