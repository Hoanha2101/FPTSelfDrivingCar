from utils import *
from config import *


while(True):
    
    start_time = time.time()
    _,frame = cap.read()
    
    ##########
    img_, direction_return, angle_str = pipeline_function(frame, INTEREST_BOX, paint = True, lane_paint = True, interest_box = True)
    
    ##########
    print("direction_return, angle_return:", direction_return, angle_str)
    
    # Hiển thị FPS lên hình
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    print(f"FPS: {fps:.2f}")
    
    cv2.imshow('Final_Lane_Detected',img_)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    
