# system modules
import sys
sys.path.append('..')
import cv2
import time
import numpy as np
import threading

# local modules
from utils import saturation, draw_rectangle
import Person
from human_detection import HumanDetection

class MultiTracking():
    def __init__(self):
        self.humanDetection = HumanDetection()
        self.OUTPUT_SIZE_WIDTH = 640
        self.OUTPUT_SIZE_HEIGHT = 640
        self.rectangleColor = (0, 255, 0)
        self.rectangleColor_detect = (0, 255, 255)
        self.fps = 0

        # New Class
        self.PersonManager = []
        self.currentFaceID = 0                
        self.fidsToDelete = []
        
        self.baseImage = None
        self.gray = None    
        self.frameCounter = -1
        

    def doRecognizePerson(self, person):
        print('Start predict')
        pass


    def deleteRedundantPerson(self):
        for person in self.fidsToDelete:
            print("Removing fid " + str(person.getId()) + " from list of trackers")
            self.PersonManager.remove(person)
                

    def check_new_face(self):
        # self.gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)
        faces = self.humanDetection.detect(self.baseImage)

        for bbox in faces:
            (x, y, w, h) = bbox
            draw_rectangle(self.resultImage, x, y, x + w, y + h, self.rectangleColor_detect)

            #calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            #Variable holding information which faceid we matched with
            matchedFid = False

            #Now loop over all the trackers and check if the 
            #centerpoint of the person is within the box of a 
            #tracker
            for person in self.PersonManager:
                [t_x, t_y, t_w, t_h] = person.getPosition()

                #calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                # check if the centerpoint of the face is within the 
                # rectangle of a tracker region. Also, the centerpoint
                # of the tracker region must be within the region 
                # detected as a person. If both of these conditions hold
                # we have a match
                if (( t_x <= x_bar   <= (t_x + t_w)) and 
                    ( t_y <= y_bar   <= (t_y + t_h)) and 
                    ( x   <= t_x_bar <= (x   + w  )) and 
                    ( y   <= t_y_bar <= (y   + h  ))):
                    matchedFid = True
                    # Keep prediction on fid

#===============================================CREATE NEW FACE========================================#
            if matchedFid is False:
                print("Creating new tracker " + str(self.currentFaceID))

                #---------------------------------------------------------------------------------------#
                person = Person.Person(self.currentFaceID)
                person.startTrack(self.baseImage, bbox)

                self.PersonManager.append(person)
                #---------------------------------------------------------------------------------------#
                #Increase the currentFaceID counter
                self.currentFaceID += 1


#=====================================================================================================#
    def detectAndTrackMultipleFaces(self, frame):
        t0 = time.time()
        self.frameCounter += 1
        img2 = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        # Add 64x512 in image
        # padding = np.zeros(shape=(64, 512, 3))
        img2 = np.pad(img2, ((64, 64), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        self.baseImage = cv2.resize(img2, (416, 416))

        # self.gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)
        # while True:
        
        self.resultImage = self.baseImage.copy()        
        self.fidsToDelete.clear()
        
#=====================================================================================================#*            
        print('[DEBUG][MULTI-TRACKING] UPDATE POSITIONS')
        for person in self.PersonManager:
            # Update new position rely on tracker
            _, [t_x, t_y, t_w, t_h] = person.updatePosition(self.baseImage)

            t_x = saturation(t_x, 0, self.baseImage.shape[1])
            t_y = saturation(t_y, 0, self.baseImage.shape[0])
            t_w = int(t_w)
            t_h = int(t_h)

            draw_rectangle(self.resultImage, t_x, t_y, t_x + t_w, t_y + t_h, self.rectangleColor)

            # text_size = int(0.05*self.baseImage.shape[0])
            text_size = 0.6

            cv2.putText(self.resultImage, "Person: " + str(person.getId()) , 
                        (int(t_x), int(t_y)), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, (0, 255, 255), 2)


#=====================================================================================================#*
            #If the tracking quality is not good enough, we must delete
            #this tracker
            trackingQuality, _ = person.updatePosition(self.baseImage)
            if trackingQuality < 1:
                self.fidsToDelete.append(person)
   
        self.deleteRedundantPerson()

        #Every 10 frames, we will have to determine which faces
        #are present in the frame
        if (self.frameCounter % 20) == 0:
            # t2 = threading.Thread(target=self.check_new_face)
            # t2.start()
            print('[DEBUG][MULTI] CHECK NEW FACES')
            #Now use the FaceDetection detector to find all faces
            human_bboxes = self.humanDetection.detect(self.baseImage)

            for bbox in human_bboxes:
                (x, y, w, h) = bbox
                print(bbox)
                # [x, y, w, h] = bbox
                # if w >= self.baseImage.shape[0]/12:
                    #calculate the centerpoint
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                #Variable holding information which faceid we matched with
                matchedFid = False

                #Now loop over all the trackers and check if the 
                #centerpoint of the face is within the box of a 
                #tracker
                for person in self.PersonManager:
                    _, [t_x, t_y, t_w, t_h] = person.updatePosition(self.baseImage)
                    
                    print([t_x, t_y, t_w, t_h])
                    #calculate the centerpoint
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    #check if the centerpoint of the face is within the 
                    #rectangleof a tracker region. Also, the centerpoint
                    #of the tracker region must be within the region 
                    #detected as a face. If both of these conditions hold
                    #we have a match
                    if (( t_x <= x_bar   <= (t_x + t_w)) and 
                        ( t_y <= y_bar   <= (t_y + t_h)) and 
                        ( x   <= t_x_bar <= (x   + w  )) and 
                        ( y   <= t_y_bar <= (y   + h  ))):
                        matchedFid = True
                        # Keep prediction on fid

#===============================================CREATE NEW FACE========================================#
                if matchedFid is False:
                    print("Creating new tracker " + str(self.currentFaceID))

                    person = Person.Person(self.currentFaceID)
                    person.startTrack(self.baseImage, bbox)
                    self.PersonManager.append(person)
                    
                    #Increase the currentFaceID counter
                    self.currentFaceID += 1

        # Calculate Frames per second (FPS)
        t1 = time.time()
        self.fps = 1/(t1-t0)
        
        cv2.putText(self.resultImage, "FPS: " + str(int(self.fps)), 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
#================================================Visualizing=====================================================#*
        #Finally, we want to show the images on the screen
        cv2.imshow("result-image", self.resultImage)


# cap = cv2.VideoCapture('C:/Users/VMC/Desktop/videoplayback.mp4')
cap = cv2.VideoCapture('C:/Users/VMC/Desktop/Leading Pedestrian Interval.mp4')
multiTracking = MultiTracking()

while True:
    ret, frame = cap.read()
    multiTracking.detectAndTrackMultipleFaces(frame)
    if cv2.waitKey(30) == 27:
        break