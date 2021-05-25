import numpy as np
import cv2
import math
import os

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)
    
    return edges
    

def get_roi(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # define vertices by four points
    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (80, ysize)
    left_top = (xsize / 2 - 50, ysize / 2 + 50)
    right_bottom = (xsize - 80, ysize)
    right_top = (xsize / 2 + 50, ysize / 2 + 50)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image






def categorize_lines(img, lines):
    x_middle = img.shape[1] / 2
    all_left_lines = []
    all_right_lines = []
   
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x1-x2) > 2:
            # Fit polynomial, find intercept and slope 
            params = np.polyfit((x1, x2), (y1, y2), 1)  
            k = params[0] 
            y_intercept = params[1] 
            
            if x1 < x_middle and k < -0.5:
                all_left_lines.append(line[0])
            elif x2 > x_middle and k > 0.5:
                all_right_lines.append(line[0])
                
    all_left_lines.sort(key=lambda x: x[0])
    all_right_lines.sort(key=lambda x: x[0])
    
    
    
    return all_left_lines,all_right_lines
    


        
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    
    return hough_lines

def pipeline(img):
    
    roi = get_roi(canny(img))
    houghlines = hough_lines(roi, 2, np.pi / 180, 15, 5, 20)
    left,right = categorize_lines(img, houghlines )
    #img_lines = improved_lines(left,right)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left,[255, 0, 0], 10)
    draw_lines(line_img, right,[0, 255, 0], 10)
    result = cv2.addWeighted(img, 0.8, line_img, 1., 0.)
    return result


# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('../challenge.mp4')


  # Loop untill the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    cv2.imshow('Edge detect', pipeline(frame))
    
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
