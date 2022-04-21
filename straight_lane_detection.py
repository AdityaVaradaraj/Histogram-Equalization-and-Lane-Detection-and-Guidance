#!/usr/env python
import numpy as np
import cv2

if __name__ == '__main__':
    # Create a VideoCapture object
    cap = cv2.VideoCapture('whiteline.mp4') # input video
    
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frameSize = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # VideoWriter Objects are created to record videos
    out_2 = cv2.VideoWriter('output_2.mp4', fourcc, 50, frameSize)
    while(True):
        ret, frame = cap.read()
        if(ret == True):
            # To check for Horizontally flipped video,can uncomment this line below
            # frame = cv2.flip(frame, flipCode= 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale     
            
            ROI = np.zeros(frame.shape).astype(np.uint8)
            ROI[int(0.6*frame_height):frame_height, int(0.125*frame_width):int(0.875*frame_width)] = frame[int(0.6*frame_height):frame_height, int(0.125*frame_width):int(0.875*frame_width)]
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0,0,240], dtype=np.uint8)
            upper_white = np.array([145,145,255], dtype=np.uint8)

            # Threshold the HSV image to get only white colors
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            res = cv2.bitwise_and(ROI, ROI, mask=mask_white)            
            bi = cv2.bilateralFilter(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 5, 75, 75)
            edges = cv2.Canny(bi,100,200)
            lines = cv2.HoughLinesP(image=edges, rho=0.75,theta=np.pi/180,threshold=30, minLineLength=5, maxLineGap=10000)
            bi_left = np.copy(bi[int(0.6*frame_height):frame_height, :int(0.5*frame_width)+1])
            bi_right = np.copy(bi[int(0.6*frame_height):frame_height, int(0.5*frame_width):])
            left_count = np.count_nonzero(bi_left)
            right_count = np.count_nonzero(bi_right)
            
            left_line_count = 0
            right_line_count = 0
            for line in lines:
                x1= line[0][0]
                y1= line[0][1]
                x2= line[0][2]
                y2= line[0][3]
                if right_count >= left_count:
                    if ((y2-y1)/(x2-x1) < 0) and (left_line_count == 0):
                        cv2.line(frame,(x2,y2),(x1,y1),(0,0,255),3)
                        left_line_count += 1
                    elif ((y2-y1)/(x2-x1) >= 0) and (right_line_count == 0):         
                        cv2.line(frame,(x2,y2),(x1,y1),(0,255,0),3)
                        right_line_count += 1
                else:
                    if ((y2-y1)/(x2-x1) < 0) and (left_line_count == 0):
                        cv2.line(frame,(x2,y2),(x1,y1),(0,255,0),3)
                        left_line_count += 1
                    elif ((y2-y1)/(x2-x1) >= 0) and (right_line_count == 0):         
                        cv2.line(frame,(x2,y2),(x1,y1),(0,0,255),3)
                        right_line_count += 1
            cv2.imshow('frame', frame)               
            out_2.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    out_2.release()