import cv2
from multi_tracking import MultiTracking

# cap = cv2.VideoCapture('C:/Users/VMC/Desktop/videoplayback.mp4')
# cap = cv2.VideoCapture('C:/Users/VMC/Desktop/Leading Pedestrian Interval.mp4')
cap = cv2.VideoCapture(0)
multiTracking = MultiTracking()

while True:
    ret, frame = cap.read()
    multiTracking.detectAndTrackMultipleFaces(frame)
    if cv2.waitKey(30) == 27:
        break