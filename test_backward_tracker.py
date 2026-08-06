import cv2
import sys
import os

# Ensure backend module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools.object_tracker import ObjectTracker


def test_backward_tracking():
    video_path = 'C:/Users/ADMIN/Videos/Mellstroy Content/Video by npc._.arc.mp4'
    if not os.path.exists(video_path):
        print(f"Test skipped, video not found: {video_path}")
        return
        
    sub_areas = [(0, 100, 0, 100)]
    
    class MockSubRemover:
        def append_output(self, msg):
            pass
            
    remover = MockSubRemover()
    
    # Init tracker starting at frame 62
    tracker = ObjectTracker(video_path, sub_areas, start_frame=62)
    
    print("Running find_subtitle_frame_no...")
    result = tracker.find_subtitle_frame_no(sub_remover=remover)
    
    # Check if backward tracking filled frames 1 to 61
    if result:
        print(f"Total frames tracked: {len(result)}")
        print(f"Frame 1 present: {1 in result}")
        print(f"Frame 61 present: {61 in result}")
        print(f"Frame 62 present: {62 in result}")
    
if __name__ == '__main__':
    test_backward_tracking()
