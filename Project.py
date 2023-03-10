import numpy as np
import cv2 as cv

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]

def draw_all_lines(img, lines, color=[255, 0, 0], thickness=7):
    for line in lines:
        for x1, y1, x2, y2 in line:
                cv.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=[0, 0, 255], thickness=7):
    global Old_pts,result,distfromcenter
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = 740
    y_upper = 215

    slope = 0
    b = 0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                # test and filter values to slope
                if ((y2-y1)/(x2-x1)) > 0.5 and ((y2-y1)/(x2-x1)) < 0.8:
                    
                    slope = ((y2-y1)/(x2-x1))
                    b = y1 - slope*x1
                    
                    x_bottom_pos.append((y_bottom - b)/slope)
                    x_upper_pos.append((y_upper - b)/slope)
                                        
                elif ((y2-y1)/(x2-x1)) < -0.5 and ((y2-y1)/(x2-x1)) > -0.8:
                
                    slope = ((y2-y1)/(x2-x1))
                    b = y1 - slope*x1
                    
                    x_bottom_neg.append((y_bottom - b)/slope)
                    x_upper_neg.append((y_upper - b)/slope)

        # To be used for transparency when drawing the road
        overlay = img.copy()
        alpha = 0.3  # Transparency factor.

        try:
            # Find intersection
            intersect_points = findIntersection(int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper)), 
                                    int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper)))
            print("Intersect :")
            print(intersect_points)
            # a new 2d array with means 
            lines_mean = np.array([[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(intersect_points[0])), int(np.mean(intersect_points[1]))], 
                                    [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(intersect_points[0])), int(np.mean(intersect_points[1]))]])

            # Draw the road path
            for i in range(len(lines_mean-1)):

                pt1 = (lines_mean[i, 0], lines_mean[i, 1])
                pt2 = (lines_mean[i+1, 0], lines_mean[i+1, 1])
                pt3 = (lines_mean[i, 2], lines_mean[i, 3])

                midposX = int(abs(int(pt1[0]-pt2[0]))/2)
                center = int(img.shape[0])/2
                distfromcenter = center-midposX
                print("distfromcenteren cm :")
                print(distfromcenter)

                # draw a triangle
                vertices = np.array([pt1, pt2, pt3], np.int32)
                pts = vertices.reshape((-1, 1, 2))
                Old_pts = pts

                cv.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=5)
                cv.fillPoly(overlay, [pts], color=(255, 0, 0))

        except:
                # draw a triangle     
                if Old_pts.any():
                    cv.polylines(overlay, [Old_pts], isClosed=True, color=(255, 255, 255), thickness=5)
                    cv.fillPoly(overlay, [Old_pts], color=(255, 0, 0))

                    result = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                else:
                    pass


# Video Capture 
cap = cv.VideoCapture("project_video.mp4")
global Old_pts,result,distfromcenter
Old_pts = np.array([])
distfromcenter = 0
while (cv.waitKey(10)<0):
    # Capture frame-by-frame
    ret, frame = cap.read()
    result = frame
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame starts here
    # Here goes the treatement Canny Edge Detector & Hough Line Transform
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 200
    edges = cv.Canny(blur, low_t, high_t)

    # Create a set of vertices for the mask
    height = frame.shape[0]
    width = frame.shape[1]
    vertices = np.array([[(0,height),(5*width/10,6*height/10),(width,height)]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv.fillPoly(mask, vertices, color= (255,255,255))
    masked_edges = cv.bitwise_and(edges, mask)

    linesP = cv.HoughLinesP(masked_edges, 1, np.pi / 180, 50, None, 125, 60)
    draw_lines(frame,linesP)

    # Display the resulting frame
    cv.imshow("Road Detection", result)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()