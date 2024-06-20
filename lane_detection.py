#import OpenCv and Numpy
import cv2 as cv
import numpy as np

#This function is used to process the frames of our video to obtain lane
def detect_lane(frame):
    #Convert the image into grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #We blur the frames to remove any unwanted or weaker edges
    blurr = cv.GaussianBlur(gray, (7, 7), 0)

    #To detect edges we using the Canny function and store it in edges
    edges = cv.Canny(blurr, 50, 150)

    #Define a region of interest and apply mask to remove the unnecessary edges
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    
    #This region is defined specifically for this video. What this function does is that it creates a polygon from the given points
    # and defines a region within them. We use fillPoly function to mask the region. The points follow clockwise order. We then use
    # the bitwise and operator to superimpose this mask on the frame so that we only study the edges found inside the region and thus
    # removing all the unnecessary edges
    region = np.array([[(0, height - 50), (width // 2 - 25, height // 2 + 25), (width // 2 + 75, height // 2 + 25), (width, height - 100), (width // 2 + 25, height), (0, height)]], dtype=np.int32)
    cv.fillPoly(mask, region, 255)
    masked_edges = cv.bitwise_and(edges, mask)

    #The Houghlines function help us form lines by giving the end points of the line. It requires some initializations like
    #threshold which is the minimum number of intersections needed for a line to be considered
    #minLineLength which is the minimum number of pixel needed for a line to be considered as a line
    #maxLineGap which is the maximum gap between two lines to consider them as one line
    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap= 50)

    #Now from all the endpoints of lines we have recieved from Houghlines we will draw the lines using this loop.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 2) #This function creates lines on the frames which together represents the contours

    return frame #Return the processed frame to display



cap = cv.VideoCapture('road_test.mp4')  #Input the video file. If you are using Jupyter notebook then enter the full path of the video.

while True:
    ret, frame = cap.read() #This function takes the video and give us frames which can then be modified to give desired results
    
    if not ret: #If the video ends this will break the loop
        break
        
    lanes = detect_lane(frame) #use this function to detect and draw lanes

    cv.imshow('Lane Detection', lanes) #Display the obtained result in a window

    if cv.waitKey(25) & 0xFF == ord('e'): # The video stops playing if we press "e" on our keyboard
        break

cap.release()
cv.destroyAllWindows()
