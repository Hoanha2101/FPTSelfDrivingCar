import numpy as np
import cv2


cap = cv2.VideoCapture('videos/test2.mp4')

# video 1

# top_left = (450, 360)  # x,y
# top_right = (850, 360) 
# bottom_right = (1080, 650)
# bottom_left = (60, 650)

K = np.array([[1154.2, 0, 671.6], [0, 1148.2, 386.0], [0, 0, 1]])
d = np.array([-0.242, -0.048, -0.001, -0.00008, 0.022])

# video 2
top_left_set = (550, 460)
top_right_set = (760, 460)
bottom_right_set = (1280, 720)
bottom_left_set = (128, 720)

INTEREST_BOX = [top_left_set, top_right_set, bottom_right_set, bottom_left_set]