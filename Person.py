import time
import dlib
from tracker import Tracker

class Person():
    def __init__(self, person_id, gender=None, age=None, bbox=None):
        self.person_id = person_id
        # self.person_tracker = dlib.correlation_tracker()
        self.person_tracker = Tracker(tracker_type_index=2)
        self.bbox = bbox


    def getId(self):
        return self.person_id


    def getFaceTracker(self):
        return self.person_tracker


    def setId(self, person_id):
        self.person_id = person_id


    def setFaceTracker(self, person_tracker):
        self.person_tracker = person_tracker


    def setFaceInfo(self, face_name):
        self.face_name = face_name


    def startTrack(self, original_image, bbox):
        # (x, y, w, h) = bbox
        # offset = int(0.05*w)
        # offset = 0
        self.person_tracker.startTrack(original_image, bbox)


    def getPosition(self):
        self.bbox = self.person_tracker.get_position()
        t_x = int(self.bbox.left())
        t_y = int(self.bbox.top())
        t_w = int(self.bbox.width())
        t_h = int(self.bbox.height())
        offset = int(0.05*t_w)
        t_x = t_x - offset
        t_y = t_y - offset
        t_w = t_w + 2*offset
        t_h = t_h + 2*offset

        return [t_x, t_y, t_w, t_h]


    def updatePosition(self, original_image):
        trackingQuality, [t_x, t_y, t_w, t_h] = self.person_tracker.update(original_image)
        return trackingQuality, [t_x, t_y, t_w, t_h]
