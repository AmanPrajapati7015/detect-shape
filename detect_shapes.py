import cv2
import numpy as np

shape_map = {3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon", 7: "Heptagon"}

img = cv2.imread('shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(imgBlur, 50, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    
    num_sides = len(approx)
    
    if num_sides in shape_map:
        shape = shape_map[num_sides]
    elif num_sides > 7:
        shape = "Circle"

    (x, y, w, h) = cv2.boundingRect(approx)
    
    B,G,R=img[y+h//2][x+w//2]
    color = f'rgb({R}, {G}, {B})'
    cv2.putText(img, shape, (x + 10, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, color, (x + 10, y + h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)  #showing bounding box


    print((B,G,R))




cv2.imshow('Detected Shapes', img)

k = cv2.waitKey(0)
if(k == ord('q')):
    cv2.destroyAllWindows()
    cv2.imwrite('output.png', img)
