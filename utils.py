import numpy as np
from numpy import linalg as LA
from numpy import linalg as la
from matplotlib import pyplot as plt
import time
import math
import cv2
import time
from config import *

K = np.array([[1154.2, 0, 671.6], [0, 1148.2, 386.0], [0, 0, 1]])
d = np.array([-0.242, -0.048, -0.001, -0.00008, 0.022])

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)

def image_processing(img, sthreshold=(100, 255), sxthreshold=(15, 255)):
    img = undistort(img, K, d)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l, s = hls[:, :, 1], hls[:, :, 2]  # L channel & S channel
    
    sobel_x = np.abs(cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3))
    sobel_x = np.uint8(255 * sobel_x / np.clip(np.max(sobel_x), 1, None))  # Chuẩn hóa
    
    # Ánh xạ vào khoảng threshold
    binary = ((sxthreshold[0] <= sobel_x) & (sobel_x <= sxthreshold[1])) | \
             ((sthreshold[0] <= s) & (s <= sthreshold[1]))
    
    return binary.astype(np.uint8) * 255
    
def perspective_warp(img, INTEREST_BOX):
    
    top_left = INTEREST_BOX[0]
    top_right = INTEREST_BOX[1]
    bottom_right = INTEREST_BOX[2]
    bottom_left = INTEREST_BOX[3]

    # For calculating the perspective transform 
    h,w = img.shape[:2]
    pts_src = np.float32([[top_left, top_right, bottom_right, bottom_left]])
    pts_dst = np.float32([[0,0],[w, 0],[w,h],[0, h]])
    P = cv2.getPerspectiveTransform(pts_src, pts_dst) 
    warp = cv2.warpPerspective(img, P, (img.shape[1],img.shape[0]))
    return warp
    
def inv_perspective_warp(img, INTEREST_BOX):
    
    top_left = INTEREST_BOX[0]
    top_right = INTEREST_BOX[1]
    bottom_right = INTEREST_BOX[2]
    bottom_left = INTEREST_BOX[3]
    
    pts_src = np.array([[top_left, top_right, bottom_right, bottom_left]], np.int32)
    pts_src = pts_src.reshape((-1, 1, 2))
    
    # For calculating the inverse perspective transform 
    h,w = img.shape[:2]
    pts_src = np.float32([[top_left, top_right, bottom_right, bottom_left]])
    pts_dst = np.float32([[0,0],[w, 0],[w,h],[0, h]])
    P = cv2.getPerspectiveTransform(pts_src, pts_dst) 
    warp = cv2.warpPerspective(img, np.linalg.inv(P), (img.shape[1],img.shape[0]))
    return warp

def get_hist(img):
    # Calculating the histogram
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_p, left_q, left_r = [],[],[]
right_p, right_q, right_r = [],[],[]

def windows(img, min_pix = 1, margin=100,num_wind=9, windows_flag = True):
    global left_p, left_q, left_r,right_p, right_q, right_r 
    l_point= np.empty(3)
    r_point = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    mid_point = int(histogram.shape[0]/2)
    
    # Creating the base of bins/windows for  left and right lanes
    left_bin_base = np.argmax(histogram[:mid_point])
    right_bin_base = np.argmax(histogram[mid_point:]) + mid_point
    
    # Creating empty lists to receive left and right lane pixel indices
    leftlane_indices = []
    rightlane_indices = []
    
    # Setting the height of windows
    bin_h = int(img.shape[0] / num_wind)
    
    # Finding the x and y positions of all nonzero pixels
    pixel_indices = img.nonzero()
    pixel_y = np.array(pixel_indices[0])
    pixel_x = np.array(pixel_indices[1])
    
    # Current position to be updated for each window
    current_bin_left = left_bin_base
    current_bin_right = right_bin_base
    

    # Iterating over the bins/windows
    for w in range(num_wind):
        # Identify window boundaries in x and y (and right and left)
        w_y_bottom = img.shape[0] - (w +1)*bin_h
        w_y_top = img.shape[0] - w * bin_h
        
        w_xleft_bottom = current_bin_left - margin
        w_xleft_top = current_bin_left + margin
        
        w_xright_bottom = current_bin_right - margin
        w_xright_top = current_bin_right + margin
        
        # Draw the windows on the  image
        if windows_flag == True:
            cv2.rectangle(out_img,(w_xleft_bottom,w_y_bottom),(w_xleft_top,w_y_top),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(w_xright_bottom,w_y_bottom),(w_xright_top,w_y_top),
            (100,255,255), 3) 
            
        # Findding the nonzero pixels in x and y within the window
        req_left_pixels = ((pixel_y >= w_y_bottom) & (pixel_y < w_y_top) & 
        (pixel_x >= w_xleft_bottom) &  (pixel_x < w_xleft_top)).nonzero()[0]
        
        req_right_pixels = ((pixel_y >= w_y_bottom) & (pixel_y < w_y_top) & 
        (pixel_x >= w_xright_bottom) &  (pixel_x < w_xright_top)).nonzero()[0]
        
        # Append these indices to the corresponding lists
        leftlane_indices.append(req_left_pixels)
        rightlane_indices.append(req_right_pixels)
        
        # If we found > minpix pixels, recenter next window on their mean position
        if len(req_left_pixels) > min_pix:
            current_bin_left = int(np.mean(pixel_x[req_left_pixels]))
        if len(req_right_pixels) > min_pix:        
            current_bin_right = int(np.mean(pixel_x[req_right_pixels]))

   # Concatenate the arrays of left and right lane pixel indices
    leftlane_indices = np.concatenate(leftlane_indices)
    rightlane_indices = np.concatenate(rightlane_indices)

    # Calculating the left and right lane pixel positions
    leftlane_x_pixels = pixel_x[leftlane_indices]
    leftlane_y_pixels = pixel_y[leftlane_indices] 
    
    rightlane_x_pixels = pixel_x[rightlane_indices]
    rightlane_y_pixels = pixel_y[rightlane_indices] 

    # Fitting a second order polynomial to each lane
    leftlane_fit = np.polyfit(leftlane_y_pixels, leftlane_x_pixels, 2)
    rightlane_fit = np.polyfit(rightlane_y_pixels, rightlane_x_pixels, 2)
    
    left_p.append(leftlane_fit[0])
    left_q.append(leftlane_fit[1])
    left_r.append(leftlane_fit[2])
    
    right_p.append(rightlane_fit[0])
    right_q.append(rightlane_fit[1])
    right_r.append(rightlane_fit[2])
    
    l_point[0] = np.mean(left_p[-10:])
    l_point[1] = np.mean(left_q[-10:])
    l_point[2] = np.mean(left_r[-10:])
    
    r_point[0] = np.mean(right_p[-10:])
    r_point[1] = np.mean(right_q[-10:])
    r_point[2] = np.mean(right_r[-10:])
    
    # Generating x and y values for plotting
    y_values = np.linspace(0, img.shape[0]-1, img.shape[0] )
    leftlane_fit_x = l_point[0]*y_values**2 + l_point[1]*y_values + l_point[2]
    rightlane_fit_x = r_point[0]*y_values**2 + r_point[1]*y_values + r_point[2]

    out_img[pixel_y[leftlane_indices], pixel_x[leftlane_indices]] = [255, 0, 100]
    out_img[pixel_y[rightlane_indices], pixel_x[rightlane_indices]] = [0, 100, 255]
    
    return out_img, (leftlane_fit_x, rightlane_fit_x), (l_point, r_point), y_values

def calculate_angle_BAC(bottom_right, bottom_left, lane_center_pos, top_right, top_left):
    # Tọa độ điểm A, B, C
    A = ((bottom_right[0] - bottom_left[0]) / 2, bottom_right[1])
    B = (lane_center_pos, top_right[1])
    C = ((top_right[0] - top_left[0]) / 2, top_right[1])

    # Vector AB và AC
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    AC = np.array([C[0] - A[0], C[1] - A[1]])

    # Tích vô hướng AB . AC
    dot_product = np.dot(AB, AC)

    # Độ dài vector AB và AC
    norm_AB = np.linalg.norm(AB)
    norm_AC = np.linalg.norm(AC)

    # Tính cos(theta) và giới hạn giá trị tránh lỗi số học
    cos_theta = np.clip(dot_product / (norm_AB * norm_AC), -1.0, 1.0)

    # Góc theta (radian)
    theta_rad = np.arccos(cos_theta)

    # Chuyển sang độ
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg

def get_polynomial(img, leftlane_x_pixels, rightlane_x_pixels, INTEREST_BOX):
    
    top_left = INTEREST_BOX[0]
    top_right = INTEREST_BOX[1]
    bottom_right = INTEREST_BOX[2]
    bottom_left = INTEREST_BOX[3]
    
    y_values = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(y_values)
    y_unit_pix = 30.5/720 # meters per pixel in y direction
    x_unit_pix = 3.7/720 # meters per pixel in x direction

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(y_values*y_unit_pix, leftlane_x_pixels*x_unit_pix, 2)
    right_fit_cr = np.polyfit(y_values*y_unit_pix, rightlane_x_pixels*x_unit_pix, 2)
    
    # Calculate the new radii of curvature for lanes
    left_curve_rad = ((1 + (2*left_fit_cr[0]*y_eval*y_unit_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curve_rad = ((1 + (2*right_fit_cr[0]*y_eval*y_unit_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Getting the car center position
    car_center_pos = img.shape[1]/2      
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = + right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    
    # Getting the lane center position
    lane_center_pos = (r_fit_x_int + l_fit_x_int) /2
    center = (car_center_pos - lane_center_pos) * x_unit_pix / 10

    angle_BAC = calculate_angle_BAC(bottom_right, bottom_left, lane_center_pos, top_right, top_left)
    
    return (left_curve_rad, right_curve_rad, center), angle_BAC

def draw_lanes(img, INTEREST_BOX, leftlane_fit_x, rightlane_fit_x, lane_paint = True):
    # Plotting the x and y values
    y_values = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left_lane = np.array([np.transpose(np.vstack([leftlane_fit_x - 20, y_values]))])
    right_lane = np.array([np.flipud(np.transpose(np.vstack([rightlane_fit_x, y_values])))])
    lane_points = np.hstack((left_lane, right_lane))
    
    if lane_paint:
        cv2.fillPoly(color_img, np.int_(lane_points), (0,150,0))

    inv_perspective = inv_perspective_warp(color_img, INTEREST_BOX)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective


def pipeline_function(frame, INTEREST_BOX, paint = False, lane_paint = False, interest_box = False):
    
    direction_return = "L"
    angle_return = 0
    angle_str = "000"
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    warp = perspective_warp(img, INTEREST_BOX)

    dst = image_processing(warp)

    slide_img, lanes, curve, y_vales = windows(dst)

    curve_radius, angle_BAC = get_polynomial(dst, lanes[0],lanes[1], INTEREST_BOX)
    if paint :
        img_ = draw_lanes(frame, INTEREST_BOX, lanes[0], lanes[1], lane_paint)
    else:   
        img_ = frame
        
    offset = curve_radius[2]
    
    if offset <= 0.22:
        
        direction_return = "R"
        angle_return = int(angle_BAC)
        angle_str = f"{angle_return:03d}"
        
        if paint:
            direction = '--->>>> '
            cv2.putText(img_, direction, (INTEREST_BOX[1][0], 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img_,angle_str, (int(img_.shape[1]/2), 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        
    elif offset > 0.40:
        
        direction_return = "L"
        angle_return = int(angle_BAC)
        angle_str = f"{angle_return:03d}"
        
        if paint:
            direction = '<<<<--- '
            cv2.putText(img_, direction, (int(img_.shape[1]/2) - 300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img_,angle_str, (int(img_.shape[1]/2), 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    if paint == True and interest_box == True:
        top_left = INTEREST_BOX[0]
        top_right = INTEREST_BOX[1]
        bottom_right = INTEREST_BOX[2]
        bottom_left = INTEREST_BOX[3]
        pts_src = np.array([[top_left, top_right, bottom_right, bottom_left]], np.int32)
        pts_src = pts_src.reshape((-1, 1, 2))
        cv2.polylines(img_, [pts_src], isClosed=True, color=(255, 125, 0), thickness=3)
    
    return img_, direction_return, angle_str